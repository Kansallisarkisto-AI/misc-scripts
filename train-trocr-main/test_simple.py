import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time
from collections import defaultdict

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
from evaluate import load
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation import GenerationConfig
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from collections import OrderedDict
from pathlib import Path
import re

# Args
parser = argparse.ArgumentParser("test_simple")

parser.add_argument("--experiment_name", type=str, default="experiment_name")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--root_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)

parser.add_argument("--img_height", type=int, default=182)
parser.add_argument("--img_width", type=int, default=1022)
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--retrained_tokenizer", type=bool, default=False)
parser.add_argument("--image_only", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--no_binning",
    action="store_false",
    dest="binning",
    help="Disable width binning"
)
parser.set_defaults(binning=True)

args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


Path(args.output_path).mkdir(parents=True, exist_ok=True)

# Monkey-patch ViT for variable width (same as training)
_original_embeddings_forward = ViTEmbeddings.forward

def universal_patch_forward(self, pixel_values, **kwargs):
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
    return embeddings

def universal_embeddings_forward(self, *args, **kwargs):
    kwargs["interpolate_pos_encoding"] = True
    return _original_embeddings_forward(self, *args, **kwargs)

ViTPatchEmbeddings.forward = universal_patch_forward
ViTEmbeddings.forward = universal_embeddings_forward

class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = self.image_processor
        self.chat_template = None

# Load processor + model
if args.binning:
    processor = TrOCRProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        do_resize=False,
        do_center_crop=False,
    )
else:
    processor = TrOCRProcessorCustom.from_pretrained(args.model_path,
                                        use_fast=True,
                                        do_resize=True, do_center_crop=False,
                                        size={'height': args.img_height, 'width': args.img_width})


model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

if args.retrained_tokenizer:
    bottleneck = model.config.decoder.hidden_size
    model.decoder.output_projection = nn.Linear(
        bottleneck, len(processor.tokenizer), bias=False
    )
    model.decoder.model.decoder.embed_tokens = nn.Embedding(
        len(processor.tokenizer),
        bottleneck,
        padding_idx=processor.tokenizer.pad_token_id,
    )

    state_dict = torch.load(args.model_path + "/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

model.config.interpolate_pos_encoding = True
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 128
model.config.num_beams = 1
#model.config.length_penalty = 2
#model.config.no_repeat_ngram_size = 3

gen_cfg = GenerationConfig.from_model_config(model.config)

model.to(device)
model.eval()

# Load metrics from Huggingface evaluate
cer_metric = load("cer")
wer_metric = load("wer")

# Load test data
test_df = pd.read_csv(args.test_data_path)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.head(128)

tqdm.write(f"Test samples: {len(test_df)}")

# Width binning (copied from training)
if args.binning:
    #BIN_W = [1024 - i * 192 for i in range(5)]
    BIN_W = sorted([args.img_width - i * 192 for i in range(3)])  # 3 bins at intervals of 192, down to w=384
    tqdm.write("Binning")
else:
    BIN_W = [args.img_width]
    tqdm.write("No binning")

BIN_H = args.img_height

def snap_width_to_bin(w: int) -> int:
    # clamp to valid range
    w = max(min(w, max(BIN_W)), min(BIN_W))

    # snap up to the nearest larger (or equal) bin
    for bw in BIN_W:
        if bw >= w:
            return bw

    # fallback (shouldn't be reached due to clamping)
    return BIN_W[-1]


def _compute_bin_w_one(args_tuple):
    root_dir, file_name, BIN_H = args_tuple
    
    with Image.open(root_dir + file_name) as img:
        w, h = img.size
    # divide ideal width by 3 to compress 3 times in width (more in line with original 384x384 compressed TrOCR pretraining)
    ideal_w = int(round((w / max(h, 1)) * BIN_H) / 3)
    return snap_width_to_bin(ideal_w)

tqdm.write("Computing width bins...")
test_df["bin_w"] = [
    _compute_bin_w_one((args.root_path, fn, BIN_H)) for fn in tqdm(test_df.file_name)
]
test_df["bin_id"] = test_df["bin_w"].apply(lambda w: BIN_W.index(w))
test_df.to_csv(os.path.join(args.output_path, "bins.csv"), index=False)

# Group indices by bin_id
groups = defaultdict(list)
for idx, row in test_df.iterrows():
    groups[row.bin_id].append(idx)

# Manual batched inference (WIDTH-SAFE)
processing_times = []

trans = str.maketrans({
        "ѣ": "е", "Ѣ": "Е",
        "і": "и", "І": "И"
    })

import re

def clean_russian_detokenized(text: str) -> str:
    text = re.sub(r"\s+", " ", text)

    # punctuation
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)

    # brackets
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)

    # quotes
    text = re.sub(r"\s+([»”\"'])", r"\1", text)
    text = re.sub(r"([«“\"'])\s+", r"\1", text)

    # hyphen compounds
    text = re.sub(r"\s*-\s*", "-", text)

    # em dash
    text = re.sub(r"\s*—\s*", " — ", text)

    # ---- protect paired slashes ----
    protected = []
    def protect(m):
        protected.append(m.group(1))
        return f"§{len(protected)-1}§"

    text = re.sub(r"/\s*([^/\s][^/]*?)\s*/", protect, text)

    # fractions
    text = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", text)

    # word separators
    text = re.sub(r"(?<=\w)\s*/\s*(?=\w)", "/", text)

    # restore paired slashes with spaces around
    for i, val in enumerate(protected):
        text = text.replace(f"§{i}§", f" /{val}/ ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def fix_slashes(text):
    text = re.sub(r"/\s*([^/]+?)\s*/", r"/\1/", text)   # / word / → /word/
    text = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", text)   # 7 / 29 → 7/29
    return text

for bin_id, indices in groups.items():
    bw = BIN_W[bin_id]
    tqdm.write(f"Running bin {bin_id} (width={bw}), samples={len(indices)}")

    for i in tqdm(range(0, len(indices), args.batch_size), leave=False):
        batch_idxs = indices[i : i + args.batch_size]
        batch_df = test_df.loc[batch_idxs]

        images = []
        for _, row in batch_df.iterrows():
            img = Image.open(args.root_path + row.file_name).convert("RGB")
            if args.binning:
                img = img.resize((row.bin_w, BIN_H), Image.BICUBIC)
            images.append(img)

        # Assert no mixed widths
        #widths = [img.size[0] for img in images]
        #assert len(set(widths)) == 1, f"Mixed widths in batch: {widths}"

        start = time.time()
        pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                generation_config=gen_cfg,
            )

        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        preds = [x.translate(trans) for x in preds]
        
        #preds = [line[:-2] + "-" if line.endswith(" -") else line for line in preds]

        test_df.loc[batch_idxs, "predicted"] = preds
        processing_times.append(time.time() - start)

# Calculate metrics
if not args.image_only:
    cer_overall = cer_metric.compute(
        predictions=test_df.predicted.tolist(),
        references=test_df.text.tolist(),
    )
    wer_overall = wer_metric.compute(
        predictions=test_df.predicted.tolist(),
        references=test_df.text.tolist(),
    )

    tqdm.write("\n\n\n")
    tqdm.write(f"CER: {cer_overall:.4f}")
    tqdm.write(f"WER: {wer_overall:.4f}")
    tqdm.write("\n\n\n")

    TEXT_LENGTH_BINS = [(i, i + 4) for i in range(0, 124, 5)]

    cer_metric_bin = load("cer")

    bin_cer = OrderedDict()
    bin_counts = OrderedDict()

    for lo, hi in TEXT_LENGTH_BINS:
        key = f"{lo}-{hi}"

        mask = test_df["text"].str.len().between(lo, hi)
        subset = test_df[mask]

        if len(subset) == 0:
            bin_cer[key] = 0.0
            bin_counts[key] = 0
            continue

        cer = cer_metric_bin.compute(
            predictions=subset["predicted"].tolist(),
            references=subset["text"].tolist(),
        )

        bin_cer[key] = cer
        bin_counts[key] = len(subset)
    
    labels = list(bin_cer.keys())
    values = list(bin_cer.values())

    non_empty = [v for v in values if v > 0]
    mean_cer = sum(non_empty) / max(len(non_empty), 1)

    plt.figure(figsize=(17, 14))
    plt.bar(labels, values)
    plt.ylabel("CER")
    plt.xlabel("Textline length (characters)")

    plt.title(f"{str(BIN_W)} x {args.img_height}: overall CER {cer_overall:.3f}; bin-mean CER {mean_cer:.3f}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.gca().set_aspect('equal')

    plot_path = os.path.join(args.output_path, "cer_vs_text_length.png")
    plt.savefig(plot_path, dpi=300)

    # Save version with y-axis capped at 1.0
    plt.ylim(0, 0.3)
    plot_path_capped = os.path.join(args.output_path, "cer_vs_text_length_ylim_0.30.png")
    plt.savefig(plot_path_capped, dpi=300)

    plt.close()
    
    bins_df = pd.DataFrame(
    {
        "text_length_bin": list(bin_cer.keys()),
        "cer": list(bin_cer.values()),
        "count": list(bin_counts.values()),
    }
    )

    bins_csv_path = "cer_vs_text_length_bins.csv"

    bins_df.to_csv(os.path.join(args.output_path, bins_csv_path), index=False)

    tqdm.write(f"Saved CER-vs-text-length plot to: {plot_path}")

# Save results
test_df.to_csv(os.path.join(args.output_path, "metrics.csv"), sep=";", index=False)

mean_time = np.mean(processing_times) / args.batch_size
tqdm.write(f"Mean processing time per image: {mean_time}")
