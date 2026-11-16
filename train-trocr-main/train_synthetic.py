import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set GPU 0 as the default device used for running the code
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, TrainerCallback, TrainingArguments, Trainer, TrainerState, TrainerControl

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    VisionEncoderDecoderConfig,
    AutoImageProcessor,
    RobertaModel,
    RobertaConfig
)

from transformers.generation import GenerationConfig
import safetensors
import torch.nn as nn
from torch.utils.data import Dataset 
from evaluate import load 
from PIL import Image 
import pandas as pd 
import argparse 
#import mlflow
import torch
from torch.utils.data import get_worker_info

from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist

from utils import load_annotations 

from augments import OwnAug
import torchvision.transforms.functional as F
import numpy as np

import pysam
from core import render_line, Config
from glob import glob
from pathlib import Path
import random
import pickle

parser = argparse.ArgumentParser('arguments for the code')

parser.add_argument('--output_path', type=str, default="/output/path",
                    help='Path for saving training results.')
parser.add_argument('--experiment_name', type=str, default="experiment2", help='Name for the MLflow experiment.') 
parser.add_argument('--model_name', type=str, default="model_name", help='Name of the model that is trained.') 
parser.add_argument('--model_path', type=str, default="microsoft/trocr-large-handwritten", help='Model to load (after preloading base model)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Training batch size.')
parser.add_argument('--eval_batch_size', type=int, default=16,
                    help='Evaluation batch size.')
parser.add_argument('--augment', type=bool, default=False,
                    help='Defines if image augmentations are used during training.')
parser.add_argument('--tf32', type=bool, default=False,
                    help='Use tf32. Do not combine with fp16/bf16')
parser.add_argument('--fp16', type=bool, default=False,
                    help='Use fp16. Do not combine with tf32/bf16')
parser.add_argument('--bf16', type=bool, default=False,
                    help='Use bf16. Do not combine with tf32/fp16')
parser.add_argument('--lr', type=float, default=5e-05,  #1e-04
                    help='Learning rate.')
parser.add_argument('--scheduler', type=str, default="linear", 
                    help='"linear"/"cosine"')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='How long to warmup')
parser.add_argument('--img_width', type=int, default=1024, help='Width of the input line image if not using binning')
parser.add_argument('--img_height', type=int, default=192, help='Height of the input line image')  # was 192 
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default="cuda", 
                    help='"device rocm or cuda"')
parser.add_argument('--resume', type=str, default="", 
                    help='"Path for checkpoint to resume from"')
parser.add_argument('--font_path', type=str, default="/tmp/fonts-filtered", help='Font root directory containing subdirectories of font families') 

args = parser.parse_args()


class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = self.image_processor


def get_processor(encoder_name, decoder_name):
    image_processor = AutoImageProcessor.from_pretrained(encoder_name, use_fast=True,
                                            do_resize=True,
                                            size={'height': args.img_height, 'width': args.img_width})
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessorCustom(image_processor, tokenizer)
    return processor


def get_model(encoder_name, decoder_name, max_length, num_decoder_layers=None):
    encoder_config = AutoConfig.from_pretrained(encoder_name)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False

    # BeiT normally uses relative positional embeddings, but we add absolute positional embeddings as per the TrOCR paper
    encoder_config.use_absolute_position_embeddings = True
    encoder_config.interpolate_pos_encoding = True  # this is probably useless, interpolation seems to always happen if use_absolute_position_embeddings=True
    #encoder = AutoModel.from_config(encoder_config)

    # add_pooling_layer=False is important because we want to directly translate a sequence of patch representations
    # into natural language (instead of pooling first into a single vector which would be better for classification)
    encoder = AutoModel.from_pretrained(encoder_name, config=encoder_config, add_pooling_layer=False)

    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder_config.tie_word_embeddings = False
    #decoder_config.use_cache = False

    #decoder = AutoModelForCausalLM.from_config(decoder_config)
    decoder = AutoModelForCausalLM.from_pretrained(decoder_name, config=decoder_config)

    if num_decoder_layers is not None:
        if decoder_config.model_type == "bert":
            print("BERT model")
            decoder.bert.encoder.layer = decoder.bert.encoder.layer[-num_decoder_layers:]
        elif decoder_config.model_type in ("roberta", "xlm-roberta"):
            print("Roberta model")
            decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[-num_decoder_layers:]
        else:
            raise ValueError(f"Unsupported model_type: {decoder_config.model_type}")
        
        decoder_config.num_hidden_layers = num_decoder_layers
        decoder.config.num_hidden_layers = num_decoder_layers

        for new_idx, layer in enumerate(decoder.roberta.encoder.layer):
            layer.attention.self.layer_idx = new_idx
            if hasattr(layer, "crossattention"):
                layer.crossattention.self.layer_idx = new_idx

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    config.tie_word_embeddings = False
    config.decoder.num_hidden_layers = num_decoder_layers
    config.decoder.tie_word_embeddings = False
    config.encoder.add_cross_attention = False
    config.encoder.interpolate_pos_encoding = True
    config.encoder.use_absolute_position_embeddings = True

    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    processor = get_processor(encoder_name, decoder_name)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.generation_config.max_length = max_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    # https://github.com/NielsRogge/Transformers-Tutorials/issues/70
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    return model, processor

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Path for saving the trained model and processor files
SAVE_MODEL_PATH = args.output_path + args.model_name + '/'

print("Loading model")
# LOAD MODELS
# microsoft/beit-large-patch16-224-pt22k is probably best for pure feature extraction (no bias for classification like in -ft22k)
model, processor = get_model("microsoft/beit-large-patch16-224-pt22k", "ai-forever/ruRoberta-large", 128, num_decoder_layers=12)

model.to(device)

# Initialize metrics
cer_metric = load("cer")
wer_metric = load("wer")

fontdir_paths = [Path(x) for x in glob(f"{args.font_path}/*/", recursive=True)]
fontfile_paths = (
    [Path(x) for x in glob(f"{args.font_path}/*/*.ttf", recursive=True)] +
    [Path(x) for x in glob(f"{args.font_path}/*/*.otf", recursive=True)]
)

print(f"Found {len(fontfile_paths)} font files.")

with open("/tmp/font_cmap.pkl", "rb") as f:
    font_to_supported_character = pickle.load(f)
print("Finished loading cmaps")

class SyntheticTextlineDataset(Dataset):
    def __init__(self, bgz_files, bgz_index_files, bgz_index_tuples, fontdir_paths, font_to_supported_character, processor, max_target_length=128, augment=False):
        #self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.augmentator = OwnAug()

        self.bgz_paths = bgz_files
        self.bgz_index_paths = bgz_index_files

        self.bgz_index_tuples = bgz_index_tuples

        # Will be created per worker
        self.bgz_files = None
        self.bgz_index_files = None

        self.fontdir_paths = fontdir_paths
        self.font_to_supported_character = font_to_supported_character

        cfg = Config()
        cfg.BACKGROUND_COLOR_VARIATION = 10
        cfg.MOLD_PROB = 0.04
        cfg.STAIN_PROB = 0.04
        cfg.FAINT_PROB = 0.04
        cfg.PARTIAL_FAINT_PROB = 0.04
        cfg.BOLD_PROB = 0.04
        cfg.BREAK_PROB = 0.04
        cfg.WAVE_PROB = 0.1
        cfg.WAVE_AMPLITUDE = (5.0, 60.0)
        cfg.WAVE_FREQUENCY = (0.1, 5.0)
        cfg.PAPER_TEXTURE_PROB = 0.04
        cfg.NOISE_STDDEV = 2.0
        cfg.FONT_COLOR_VARIATION = 30
        self.textgen_config = cfg

        #DEBUG
        self.save_debug_images = False  # or pass as argument
        self.base_debug_dir = Path("/flash/project_462001175/trocr/debug_images")
        self.worker_debug_dir = None  # will be set per worker

    def __len__(self):
        return len(self.bgz_index_tuples)

    def _ensure_open(self):
        if self.bgz_files is None:
            self.bgz_files = [pysam.BGZFile(p, "rb") for p in self.bgz_paths]
        if self.bgz_index_files is None:
            # memmap would be an option but uses up virtual memory
            #self.bgz_index_files = [np.memmap(p, dtype="<u8", mode="r") for p in self.bgz_index_paths]
            # so let's use file handles instead
            self.bgz_index_files = [open(filename, "rb") for filename in self.bgz_index_paths]

        # worker-specific debug directory
        if self.save_debug_images and self.worker_debug_dir is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0

            self.worker_debug_dir = self.base_debug_dir / f"worker_{worker_id}"
            self.worker_debug_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, idx):
        self._ensure_open()
        # get indexs in bgz lists and the location inside index file
        bgz_file_idx, index_location = self.bgz_index_tuples[idx]

        # convert to Python int for safety
        bgz_file_idx = int(bgz_file_idx)
        index_location = int(index_location)

        # select corresponding bgz file and get the virtual offset (at index_location) inside that file from the corresponding index file
        bgz_file = self.bgz_files[bgz_file_idx]
        # memmap option
        #virtual_offset = int(self.bgz_index_files[bgz_file_idx][index_location])
        index_file = self.bgz_index_files[bgz_file_idx]
        index_file.seek(index_location * 8)
        # file read option
        virtual_offset = int.from_bytes(index_file.read(8), "little")

        # seek to virtual offset inside bgz file and read line
        bgz_file.seek(virtual_offset)
        text = bgz_file.readline().decode().rstrip()

        # prepare image (i.e. resize + normalize if not variable_width, else just normalize)
        #image = Image.open(self.root_dir + file_name).convert("RGB")

        text_for_generation = text

        # generate text image
        tries = 0
        while True:
            try:
                image, font_name = render_line(
                    text_for_generation,
                    self.fontdir_paths,
                    self.font_to_supported_character,
                    self.textgen_config,
                    virtual_offset + tries
                )
                break
            except (OSError, Exception) as e:
                # In case of exception, replace with empty image
                # Be careful with this and watch out for Exception!
                if tries > 9:  # something must be wrong with the text, reset to whitespace
                    print(f"Textgen exception {str(e)}, overriding with empty: {text}")
                    random.seed(virtual_offset + tries)
                    text_for_generation = " " * random.randint(0,127)  # produce empty image of random width
                    text = ""  # empty label for empty image
            tries += 1

        # Add augmentations
        if self.augment:    
            w,h = image.size
            # Augmentations require minimum image size
            if (w > 25) and (h > 25):         
                image = self.augmentator(image)
        
        if self.save_debug_images:
            filename = self.worker_debug_dir / f"{idx}.png"
            image.save(filename)

        # get data in model input form from processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", truncation=True, 
                                          max_length=self.max_target_length).input_ids
        
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

        return encoding

# data files
bgz_files = ["/flash/project_462001175/trocr/corpus/historyrussia.bgz",
              "/flash/project_462001175/trocr/corpus/pravoslavnoe.bgz",
              "/flash/project_462001175/trocr/corpus/ru_deduped.bgz"]

# corresponding index files
bgz_index_files = [file + ".lidx" for file in bgz_files]

# virtual offsets are stored in binary files as uint64, so divide the file size by 8 bytes to get the number of indexed items in the file
voffset_counts = [os.path.getsize(file) // 8 for file in bgz_index_files]

bgz_index_tuples_train = []
bgz_index_tuples_eval = []

i = 0
for n in voffset_counts:
    sample = np.asarray(random.sample(range(n), 30), dtype=np.int64) #(n, size=11000, replace=False)
    result = np.column_stack((np.full_like(sample, i), sample))
    #bgz_index_tuples_train.append(result[0:10000])
    bgz_index_tuples_train.append(result)
    #bgz_index_tuples_eval.append(result[10000:])
    bgz_index_tuples_eval.append(result)
    i += 1

# Combine lists into single arrays
bgz_index_tuples_train = np.vstack(bgz_index_tuples_train)
bgz_index_tuples_eval = np.vstack(bgz_index_tuples_eval)

# Create train dataset
train_dataset = SyntheticTextlineDataset(
                           bgz_files=bgz_files, 
                           bgz_index_files=bgz_index_files, 
                           bgz_index_tuples=bgz_index_tuples_train, 
                           fontdir_paths=fontdir_paths, 
                           font_to_supported_character=font_to_supported_character,
                           processor=processor,
                           max_target_length=128,
                           augment=args.augment)

# Create eval dataset
eval_dataset = SyntheticTextlineDataset(
                           bgz_files=bgz_files, 
                           bgz_index_files=bgz_index_files, 
                           bgz_index_tuples=bgz_index_tuples_eval, 
                           fontdir_paths=fontdir_paths, 
                           font_to_supported_character=font_to_supported_character,
                           processor=processor,
                           max_target_length=128,
                           augment=False)  # do not augment eval data

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# Define model configuration 
# Configure the model to expect and handle custom image sizes
model.config.interpolate_pos_encoding = True
#model.config.use_absolute_position_embeddings = True
# set special tokens used for creating the decoder_input_ids from the labels 
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id 
model.config.pad_token_id = processor.tokenizer.pad_token_id 
# make sure vocab size is set correctly 
model.config.vocab_size = model.config.decoder.vocab_size 
# set beam search parameters 
model.config.eos_token_id = processor.tokenizer.sep_token_id 
model.generation_config.max_length = 128 
model.generation_config.early_stopping = True 
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4
#gen_cfg = GenerationConfig.from_model_config(model.config)

# Allow use of tf32 data type if args.tf32:   
if args.tf32:  
    torch.backends.cuda.matmul.allow_tf32 = True     
    torch.backends.cudnn.allow_tf32 = True 

# Set arguments for model training
# For all argumenst see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
eval_fp16 = True
eval_bf16 = False
if args.bf16:
    eval_fp16 = False
    eval_bf16 = True

training_args = Seq2SeqTrainingArguments(
    report_to='none',
    predict_with_generate=True,
    eval_on_start=True,
    eval_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    logging_strategy="steps",
    logging_steps=5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    num_train_epochs=args.epochs,
    save_total_limit=20,
    output_dir=SAVE_MODEL_PATH,
    #output_dir="/tmp/" + args.model_name + '/',
    optim='adamw_torch',
    fp16=args.fp16, 
    bf16=args.bf16,
    tf32=args.tf32, 
    dataloader_num_workers=6,  # 56/8=7
    dataloader_prefetch_factor=2,
    dataloader_drop_last=True,
    learning_rate=args.lr,
    lr_scheduler_type=args.scheduler,
    warmup_ratio=args.warmup_ratio,
    fp16_full_eval=eval_fp16,
    bf16_full_eval=eval_bf16,
    #generation_config=gen_cfg,
    average_tokens_across_devices=False,
    seed=args.seed,
    eval_accumulation_steps=1,
    #batch_eval_metrics=True
)

# Function for computing CER and WER metrics for the prediction results
def compute_metrics(pred):
    labels_ids = pred.label_ids.copy()
    pred_ids = pred.predictions

    invalid_mask_preds = (pred_ids < 0) | (pred_ids >= processor.tokenizer.vocab_size)
    pred_ids[invalid_mask_preds] = processor.tokenizer.pad_token_id 
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    #labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    invalid_mask_labels = (labels_ids < 0) | (labels_ids >= processor.tokenizer.vocab_size)
    labels_ids[invalid_mask_labels] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    if any(pred_str) and any(label_str):
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
    else:
        cer = 0
        wer = 0   

    return {"cer": cer, "wer": wer}

DEBUG_WIDTHS = False

def my_collate(ins):
    if DEBUG_WIDTHS:
        widths = [x["pixel_values"].shape[-1] for x in ins]
        w0 = widths[0]
        same = all(w == w0 for w in widths)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"[rank{rank}] batch widths={widths} same={same}")

        # hard assert
        assert same, f"Mixed widths in batch on rank {rank}: {widths}"

    thedict = default_data_collator(ins)
    thedict["interpolate_pos_encoding"] = True
    return thedict


# instantiate trainer
# For all parameters see: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=my_collate,
)

# Train the model
if len(args.resume) > 0:
    print("Resuming from " + args.resume)
    trainer.train(resume_from_checkpoint=args.resume)
else:
    trainer.train()

# Let's see if this guard is useful
if trainer.is_world_process_zero():
    # Save processor
    model.save_pretrained(SAVE_MODEL_PATH)
    processor.save_pretrained(SAVE_MODEL_PATH)
