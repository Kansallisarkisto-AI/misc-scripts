import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set GPU 0 as the default device used for running the code
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, TrainerCallback, TrainingArguments, Trainer, TrainerState, TrainerControl
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from grouped_batch_sampler import GroupedBatchSampler, DistributedGroupedBatchSampler, SyncGroupedBatchSampler, GlobalPaddedGroupedBatchSampler

from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist

from utils import load_annotations

from pathlib import Path

from augments import OwnAug
import torchvision.transforms.functional as F

from sd19_alphanumeric_line_generator_seek import generate_sample_image

from multiprocessing import Value

parser = argparse.ArgumentParser('arguments for the code')

parser.add_argument('--root_path', type=str, default="",
                    help='Root path to data files.')
parser.add_argument('--data_config', type=str, default="",
                    help='Path to .yaml file containing paths to train and valid annotation .txt files. OPTIONAL')
parser.add_argument('--tr_data_path', type=str, default="/path/to/train.csv",
                    help='Path to .csv file containing the training data.')
parser.add_argument('--val_data_path', type=str, default="/path/to/val.csv",
                    help='Path to .csv file containing the validation data.')
parser.add_argument('--output_path', type=str, default="/output/path",
                    help='Path for saving training results.')
parser.add_argument('--experiment_name', type=str, default="experiment2", help='Name for the MLflow experiment.') 
parser.add_argument('--model_name', type=str, default="model_name", help='Name of the model that is trained.') 
parser.add_argument('--model_path', type=str, default="microsoft/trocr-large-handwritten", help='Model to load (after preloading base model)') 
parser.add_argument('--retrain_small_tokenizer', type=bool, default=False,
                    help='Whether to retrain tokenizer as a smaller mostly character and bigram-level tokenizer.')
parser.add_argument('--inject_cyrillic', type=bool, default=False,
                    help='Whether to inject cyrillic characters into existing English tokenizer.')
parser.add_argument('--vocab_size', type=int, default=1000,
                    help='Vocabulary size for retraining tokenizer')
parser.add_argument('--freeze_until', type=int, default=-1,
                    help='Freeze model except for embed_tokens and output_projection layers, until (epochs), -1 = disabled')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Training batch size.')
parser.add_argument('--eval_batch_size', type=int, default=16,
                    help='Evaluation batch size.')
parser.add_argument('--augment', type=bool, default=False,
                    help='Defines if image augmentations are used during training.')
parser.add_argument('--use_custom_processor', action="store_true",
                    help='Defines if using custom (cyrillic) preprocessor.')
parser.add_argument('--variable_width', type=bool, default=False,
                    help='Defines if using variable width training with aspect ratio binning instead of static resize.')
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
parser.add_argument("--numeric_proportion", type=float, default=0.0, help="Proportion of synthetic handwritten numerals in training set")

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = self.image_processor
        self.chat_template = None


# Give a name to the MLflow experiment
#mlflow.set_experiment(args.experiment_name)

# Allow logging for GPU use, CPU use etc.
#mlflow.enable_system_metrics_logging()

# Path for saving the trained model and processor files
SAVE_MODEL_PATH = args.output_path + args.model_name + '/'


if args.variable_width:
    processor = TrOCRProcessor.from_pretrained(args.model_path,
                                           use_fast=True,
                                           do_resize=False,
                                           do_center_crop=False)
else:
    # Initialize processor and model
    # By default processor resizes image to 384 x 384
    if args.use_custom_processor:
        processor = TrOCRProcessorCustom.from_pretrained(args.model_path,
                                                use_fast=True,
                                                do_resize=True, do_center_crop=False,
                                                size={'height': args.img_height, 'width': args.img_width})
    else:
        processor = TrOCRProcessor.from_pretrained(args.model_path,
                                                use_fast=True,
                                                do_resize=True, do_center_crop=False,
                                                size={'height': args.img_height, 'width': args.img_width})

# Load annotations
if args.data_config == '':
    # Load train and validation data to Pandas dataframes
    train_df = pd.read_csv(args.tr_data_path)
    val_df = pd.read_csv(args.val_data_path)

    # Reset the dataframe row indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
else:
    # Load annotations from a config file
    train_df, val_df = load_annotations(args.data_config)

# Separate UOS numeric samples from the normal training data
uos_mask = train_df["file_name"].str.contains(
    "-lines/uos/",
    na=False,
    regex=False
)

train_additional_numeric_df = train_df.loc[uos_mask].copy()
train_df = train_df.loc[~uos_mask].copy()

# Reset indices
train_additional_numeric_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)

print(f"Separated UoS Data Rescue numeric samples: {len(train_additional_numeric_df)}")
print(f"Remaining training samples: {len(train_df)}")

shared_epoch = Value("i", 0)

model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

if args.inject_cyrillic:
    print("Injecting cyrillic characters as tokens...")
    characters_to_add = ["І", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ы", "Э", "Ю", "Я", "а", "б",
                "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", "ё", "і",
                "ѣ", "һ", "–", "“", "…", "№"]
    
    old_vocab_size = len(processor.tokenizer)
    print(f"Old vocab size {old_vocab_size}")

    added = processor.tokenizer.add_tokens(characters_to_add)
    print(f"Added {added} tokens.")
    model.decoder.model.decoder.embed_tokens = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    new_vocab_size = len(processor.tokenizer)
    print(f"New vocab size {new_vocab_size}")

    # New output projection with the updated vocab size
    model.decoder.output_projection = nn.Linear(
        model.config.decoder.hidden_size, new_vocab_size, bias=False
    )
    # Initialize with a copy of the embedding weights, not a shared reference
    model.decoder.output_projection.weight.data[:, :] = model.decoder.model.decoder.embed_tokens.weight.data


if args.retrain_small_tokenizer:
    print("Retraining a smaller ~bigram tokenizer...")
    characters_to_add = ["І", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ы", "Э", "Ю", "Я", "а", "б",
                    "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", "ё", "і",
                    "ѣ", "һ", "–", "“", "…", "№", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]

    #tokenizer_df = pd.read_csv(args.tokenizer_data)
    def get_training_corpus():
        return (
            train_df.iloc[i : i + 1000]["text"]
            for i in range(0, len(train_df), 1000)
        )
    
    training_corpus = get_training_corpus()

    # retrain tokenizer with a much simpler vocabulary
    tokenizer = processor.tokenizer.train_new_from_iterator(training_corpus, vocab_size=args.vocab_size, initial_alphabet=characters_to_add)
    #if args.vocab_size <= 300:
    #    print("Added special cyrillic tokens")
    #    tokenizer.add_tokens(characters_to_add)
    # optionally ensure some tokens are added
    #tokenizer.add_tokens(["–", "“", "…", "№"])

    # swap out tokenizer for custom one
    processor.tokenizer = tokenizer

    bottleneck_size = model.config.decoder.hidden_size
    # initialize output projection and embedding layers from scratch
    # https://github.com/huggingface/transformers/issues/19329
    new_output_projection = nn.Linear(bottleneck_size, len(processor.tokenizer), bias=False)
    #old_output_projection = model.decoder.output_projection
    #mean_output_projection = old_output_projection.weight.data.to(torch.float32).mean(dim=0)  # shape (bottleneck_size,)
    #new_output_projection.weight.data[:, :] = mean_output_projection[None, :].repeat(len(processor.tokenizer), 1).to(old_output_projection.weight.dtype)
    model.decoder.output_projection = new_output_projection

    # check here https://github.com/microsoft/unilm/blob/master/trocr/trocr_models.py
    new_embeddings = nn.Embedding(len(processor.tokenizer), bottleneck_size, padding_idx=processor.tokenizer.pad_token_id)
    #old_embeddings = model.decoder.model.decoder.embed_tokens  # weird access
    #mean_emb = torch.mean(old_embeddings.weight.data.to(torch.float32), axis=0)
    #new_embeddings.weight.data[:, :] = mean_emb[None, :].repeat(len(processor.tokenizer), 1).to(old_embeddings.weight.dtype)
    model.decoder.model.decoder.embed_tokens = new_embeddings

    print("New vocab size ", len(processor.tokenizer))
    model.config.decoder.vocab_size = len(processor.tokenizer) #processor.tokenizer.vocab_size
    model.config.vocab_size = model.config.decoder.vocab_size 

model.to(device)

# Initialize metrics
cer_metric = load("cer")
wer_metric = load("wer")

BIN_W = sorted([768 - i * 192 for i in range(3)])  # 3 bins at intervals of 192, down to w=384
#BIN_W = [1024]
BIN_H = args.img_height

'''def snap_width_to_bin(w: int) -> int:
    # clamp first
    w = max(min(w, BIN_W[0]), BIN_W[-1])
    # nearest bin
    return min(BIN_W, key=lambda bw: abs(bw - w))'''

def snap_width_to_bin(w: int) -> int:
    # clamp to valid range
    w = max(min(w, max(BIN_W)), min(BIN_W))

    # snap up to the nearest larger (or equal) bin
    for bw in sorted(BIN_W):
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

class TextlineDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128, augment=False, numeric_proportion=0.0, numeric_replaces_text=False, numeric_every_nth_synthetic=3, additional_numeric_df=None, shared_epoch=None):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.augmentator = OwnAug()
        self.count_numeric = 0
        self.numeric_replaces_text = numeric_replaces_text
        self.numeric_every_nth_synthetic = numeric_every_nth_synthetic

        self.shared_epoch = shared_epoch

        self.additional_numeric_df = additional_numeric_df
        if self.additional_numeric_df is not None:
            self.additional_numeric_df = self.additional_numeric_df.reset_index(drop=True)

        if args.variable_width:
            file_names = self.df["file_name"].tolist()

            # Threads should work best since PIL image open is mostly I/O bound.
            max_workers = 4

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                self.bin_w = list(ex.map(
                    _compute_bin_w_one,
                    [(self.root_dir, fn, BIN_H) for fn in file_names],
                    chunksize=256,  # helps reduce overhead
                ))

            self.bin_id = [BIN_W.index(w) for w in self.bin_w]

        if numeric_proportion > 0.0 and numeric_replaces_text:
            self.count_numeric = int(len(self.df)*numeric_proportion)
            print(f"Replacing with {self.count_numeric} numeric samples")
        elif numeric_proportion > 0.0:
            self.count_numeric = int((len(self.df)*numeric_proportion)/(1-numeric_proportion))
            print(f"Adding {self.count_numeric} numeric samples")

        # Number of synthetic numeric positions in one epoch
        self.count_synthetic_per_epoch = (
            self.count_numeric + self.numeric_every_nth_synthetic - 1
        ) // self.numeric_every_nth_synthetic

        # Number of UoS/additional numeric positions in one epoch
        self.count_additional_per_epoch = (
            self.count_numeric - self.count_synthetic_per_epoch
        )

    def __len__(self):
        if self.numeric_replaces_text:
            return len(self.df)
        else:
            return len(self.df) + self.count_numeric

    def __getitem__(self, idx):
        # get file name + text 
        if (self.numeric_replaces_text and idx < self.count_numeric) or idx >= len(self.df):
            numeric_idx = idx if self.numeric_replaces_text else idx - len(self.df)
            if numeric_idx % self.numeric_every_nth_synthetic == 0:  # every nth sample should be synthetic
                # generate synthetic numeric sample
                seed = None
                if not self.augment:
                    seed = idx
                image, text = generate_sample_image(seed=seed, sd19_index_file=Path("/tmp/sd19/sd19.pkl"), 
                                                    backgrounds_dir=Path("/tmp/gen/backgrounds"), 
                                                    curated_dir=Path("/tmp/gen/curated"), 
                                                    dida_dir=Path("/tmp/gen/dida_segmented"))
            else:
                if self.additional_numeric_df is None or len(self.additional_numeric_df) == 0:
                    raise ValueError("additional_numeric_df is required for non-synthetic numeric samples")
                # Remove the synthetic positions from the index
                additional_idx = numeric_idx - (numeric_idx // self.numeric_every_nth_synthetic) - 1

                epoch = self.shared_epoch.value
                # Continue through additional_numeric_df across epochs,
                # wrapping only after exhausting it
                additional_idx = (epoch * self.count_additional_per_epoch + additional_idx) % len(self.additional_numeric_df)

                row = self.additional_numeric_df.iloc[additional_idx]

                text = row["text"]
                if text is None or pd.isna(text):
                    text = ""
                else:
                    text = str(text)
                image = Image.open(
                    self.root_dir + row["file_name"]
                ).convert("RGB")
        else:
            # real data
            file_name = self.df['file_name'][idx]
            text = self.df['text'][idx]  #.replace('Ъ', 'ъ')  # normalize endings
            if text is None or pd.isna(text):
                text = ""

            # prepare image (i.e. resize + normalize if not variable_width, else just normalize)
            image = Image.open(self.root_dir + file_name).convert("RGB")

        # Add augmentations
        if self.augment:    
            w,h = image.size
            # Augmentations require minimum image size
            if (w > 25) and (h > 25):         
                image = self.augmentator(image)
        
        if args.variable_width:
            # resize/stretch to the bin (width varies, height fixed)
            bw = self.bin_w[idx]
            image = image.resize((bw, BIN_H), resample=Image.BICUBIC)

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

# Create train and validation datasets
train_dataset = TextlineDataset(root_dir=args.root_path,
                           df=train_df,
                           processor=processor,
                           max_target_length=128,
                           augment=args.augment,
                           numeric_proportion=args.numeric_proportion,
                           numeric_every_nth_synthetic=3,
                           additional_numeric_df=train_additional_numeric_df,
                           shared_epoch=shared_epoch)

eval_dataset = TextlineDataset(root_dir=args.root_path,
                           df=val_df,
                           processor=processor,
                           max_target_length=128,
                           augment=False,
                           shared_epoch=shared_epoch)

print("Number of training examples (excluding numeric):", len(train_df))
print("Whole training set size (including numeric):", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# Define model configuration 
# Configure the model to expect and handle custom image sizes
model.config.interpolate_pos_encoding = True
# set special tokens used for creating the decoder_input_ids from the labels 
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id 
model.config.pad_token_id = processor.tokenizer.pad_token_id 
# make sure vocab size is set correctly 
model.config.vocab_size = model.config.decoder.vocab_size 
# set beam search parameters 
model.config.eos_token_id = processor.tokenizer.sep_token_id 
model.generation_config.max_length = 128 
model.generation_config.early_stopping = False 
model.generation_config.no_repeat_ngram_size = 0
model.generation_config.length_penalty = 1.0
model.generation_config.num_beams = 1
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
    eval_on_start=False,
    eval_strategy="epoch",
    #eval_steps=1930,
    save_strategy="epoch",
    #save_steps=1930,
    logging_strategy="epoch",
    #logging_steps=1930,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    num_train_epochs=args.epochs,
    save_total_limit=2,
    output_dir=SAVE_MODEL_PATH,
    #output_dir="/tmp/" + args.model_name + '/',
    optim='adamw_torch',
    fp16=args.fp16, 
    bf16=args.bf16,
    tf32=args.tf32,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=2,
    dataloader_drop_last=True,
    learning_rate=args.lr,
    lr_scheduler_type=args.scheduler,
    warmup_steps=args.warmup_ratio,
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

    #if any(pred_str) and any(label_str):
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    #else:
    #    cer = 0
    #    wer = 0   

    return {"cer": cer, "wer": wer}



class CERWERAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cer = 0.0
        self.total_wer = 0.0
        self.total_n = 0

    def add_batch(self, preds, refs):
        if not preds or not refs:
            return

        cer = cer_metric.compute(predictions=preds, references=refs)
        wer = wer_metric.compute(predictions=preds, references=refs)

        n = len(preds)
        self.total_cer += cer * n
        self.total_wer += wer * n
        self.total_n += n

    def compute(self):
        if self.total_n == 0:
            return {"cer": 0.0, "wer": 0.0}
        return {
            "cer": self.total_cer / self.total_n,
            "wer": self.total_wer / self.total_n,
        }

metric_accumulator = CERWERAccumulator()

def compute_metrics_batched(pred, compute_result=False):
    global metric_accumulator
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    metric_accumulator.add_batch(pred_str, label_str)

    if not compute_result:
        # return empty dict for intermediate steps
        return {}
    else:
        metrics = metric_accumulator.compute()
        metric_accumulator.reset()
        return metrics

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
    # needed for ViT model
    if model.config.encoder.model_type == "vit":
        thedict["interpolate_pos_encoding"] = True
    return thedict

class FreezingCallback(TrainerCallback):
    """Callback to freeze model for freeze_until epochs, except for unfrozen_layers (parameter names match including substring)"""
    def __init__(self, unfrozen_layers, freeze_until, trainer: Trainer):
        self.trainer = trainer
        self.unfrozen_layers = unfrozen_layers
        self.current_step_idx = 0
        self.freeze_until = freeze_until

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch < self.freeze_until:  # freeze
            self.freeze_model_except_for(self.unfrozen_layers, int(state.epoch))
        else:  # unfreeze
            for name, param in self.trainer.model.named_parameters():
                param.requires_grad = True
        
        self.current_step_idx += 1

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for name, param in self.trainer.model.named_parameters():
            param.requires_grad = True

    def freeze_model_except_for(self, unfrozen_layers: list, epoch: int):
        print(f"\nEpoch {epoch}: Freezing model except for {unfrozen_layers}")
        for name, param in self.trainer.model.named_parameters():
            # freeze all layers up to layer freeze_to including embedding layers
            should_freeze = True
            for unfrozen_name in unfrozen_layers:
                if unfrozen_name in name:
                    should_freeze = False
                    break
            
            if should_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Didn't freeze {name}")


class SamplerEpochCallback(TrainerCallback):
    def __init__(self, sampler):
        self.sampler = sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        # DistributedSampler uses this for deterministic shuffling across ranks
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class DistributedSamplerEpochCallback(TrainerCallback):
    def __init__(self, sampler):
        self.sampler = sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.sampler.set_epoch(epoch)

class BinGroupedTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        # if not in variable width mode, just return normal dataloader
        if not args.variable_width:
            return super().get_train_dataloader()

        # Base sampler handles DDP sharding + shuffling
        '''self._base_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            seed=self.args.seed,
            drop_last=True,
        )'''

        # Group IDs: one per dataset index
        group_ids = self.train_dataset.bin_id  # list[int], len == len(dataset)

        #self._grouped_batch_sampler = GroupedBatchSampler(self._base_sampler, group_ids, self.args.train_batch_size, drop_last=False)
        '''self._grouped_batch_sampler = SyncGroupedBatchSampler(
            dataset_len=len(train_dataset),
            group_ids=group_ids,
            batch_size=self.args.per_device_eval_batch_size,
            include_partial_batches=False,
            shuffle=True,
            seed=self.args.seed,
            pad_by_repeating=True,          # IMPORTANT for equal steps
        )'''

        self._grouped_batch_sampler = GlobalPaddedGroupedBatchSampler(
            dataset_len=len(train_dataset),
            group_ids=group_ids,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            seed=self.args.seed,
            drop_last=False,
            fill_partial_by_repeating=True,
        )

        #self.add_callback(SamplerEpochCallback(self._base_sampler))
        self.add_callback(DistributedSamplerEpochCallback(self._grouped_batch_sampler))

        return DataLoader(
            self.train_dataset,
            batch_sampler=self._grouped_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        # if not in variable width mode, just return normal dataloader
        if not args.variable_width:
            return super().get_eval_dataloader(eval_dataset=eval_dataset)

        if eval_dataset:
            current_eval_dataset = eval_dataset
        else:
            current_eval_dataset = self.eval_dataset
        
        # Base sampler handles DDP sharding + shuffling
        '''self._base_sampler_eval = DistributedSampler(
            current_eval_dataset,
            shuffle=False,
            seed=self.args.seed,
            drop_last=False,
        )'''

        # Group IDs: one per dataset index
        group_ids = current_eval_dataset.bin_id  # list[int], len == len(dataset)

        #self._grouped_batch_sampler_eval = GroupedBatchSampler(self._base_sampler_eval, group_ids, self.args.eval_batch_size, is_eval=False, drop_last=False)

        '''self._grouped_batch_sampler_eval = GlobalPaddedGroupedBatchSampler(
            dataset_len=len(current_eval_dataset),
            group_ids=group_ids,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            seed=self.args.seed,
            drop_last=False,
            fill_partial_by_repeating=True,
        )'''

        self._grouped_batch_sampler_eval = SyncGroupedBatchSampler(
            dataset_len=len(current_eval_dataset),
            group_ids=group_ids,
            batch_size=self.args.per_device_eval_batch_size,
            include_partial_batches=False,
            shuffle=False,
            seed=self.args.seed,
            pad_by_repeating=True,          # IMPORTANT for equal steps
        )

        '''self._grouped_batch_sampler_eval = DistributedGroupedBatchSampler(
            dataset_len=len(current_eval_dataset),
            group_ids=group_ids,
            batch_size=self.args.per_device_eval_batch_size,
            include_partial_batches=False,       # do not keep leftovers, avoids freezing in AllGather
            drop_remainder_batches=True,        # do not drop remainder batches, instead pad by repeating batches, avoids freezing in AllGather
            shuffle=False,
            seed=self.args.seed,
        )'''

        dl = DataLoader(
            current_eval_dataset,
            batch_sampler=self._grouped_batch_sampler_eval,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        is_ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_ddp else 0
        print(f"[rank{rank}] eval len(dataloader)={len(dl)}")

        return dl

# instantiate trainer
# For all parameters see: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
'''trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,  
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=my_collate,
)'''

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

trainer = BinGroupedTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=my_collate,
    #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

if args.freeze_until >= 0:
    #freezing_callback = FreezingCallback(["output_projection", "embed_tokens"], args.freeze_until, trainer)
    freezing_callback = FreezingCallback(["encoder.encoder"], args.freeze_until, trainer)
    trainer.add_callback(freezing_callback)

class DatasetEpochCallback(TrainerCallback):
    def __init__(self, shared_epoch):
        self.shared_epoch = shared_epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0

        with self.shared_epoch.get_lock():
            self.shared_epoch.value = epoch

        print(f"Dataset epoch set to {epoch}")

trainer.add_callback(
    DatasetEpochCallback(shared_epoch)
)

# Train the model
if len(args.resume) > 0:
    print("Resuming from " + args.resume)
    trainer.train(resume_from_checkpoint=args.resume)
else:
    trainer.train()

# Let's see if this guard is useful
if trainer.is_world_process_zero():
    # Save model and processor
    model.save_pretrained(SAVE_MODEL_PATH)
    processor.save_pretrained(SAVE_MODEL_PATH)
