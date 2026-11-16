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

from utils import load_annotations 

from augments import OwnAug, OwnHeavyAug
import torchvision.transforms.functional as F

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
parser.add_argument('--img_width', type=int, default=1024, help='Width of the input line image')
parser.add_argument('--img_height', type=int, default=192, help='Height of the input line image')  # was 192 
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default="cuda", 
                    help='"device rocm or cuda"')
parser.add_argument('--resume', type=str, default="", 
                    help='"Path for checkpoint to resume from"')

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

# Initialize processor and model
# By default processor resizes image to 384 x 384
processor = TrOCRProcessorCustom.from_pretrained(args.model_path,
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

model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

model.to(device)

# Initialize metrics
cer_metric = load("cer")
wer_metric = load("wer")

trans = str.maketrans({
        "ѣ": "е", "Ѣ": "Е",
        "і": "и", "І": "И"
    })

class TextlineDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128, augment=False):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.augmentator = OwnAug()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx].translate(trans)  # Normalize old Russian characters

        # prepare image (i.e. resize + normalize if not variable_width, else just normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")

        # Add augmentations
        if self.augment:    
            w,h = image.size
            # Augmentations require minimum image size
            if (w > 25) and (h > 25):         
                image = self.augmentator(image)

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
                           augment=args.augment)

eval_dataset = TextlineDataset(root_dir=args.root_path,
                           df=val_df,
                           processor=processor,
                           max_target_length=128,
                           augment=False)

print("Number of training examples:", len(train_dataset))
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
#model.generation_config.early_stopping = True 
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2
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
    save_total_limit=5,
    output_dir=SAVE_MODEL_PATH,
    #output_dir="/tmp/" + args.model_name + '/',
    optim='adamw_torch',
    fp16=args.fp16, 
    bf16=args.bf16,
    tf32=args.tf32, 
    dataloader_num_workers=5,
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

trans = str.maketrans({
        "ѣ": "е", "Ѣ": "Е",
        "і": "и", "І": "И"
    })

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

    # NOTE
    pred_str = [x.translate(trans) for x in pred_str]
    label_str = [x.translate(trans) for x in label_str]  # CHARACTER SWAP HACK

    #if any(pred_str) and any(label_str):
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    #else:
    #    cer = 0
    #    wer = 0   

    return {"cer": cer, "wer": wer}


def my_collate(ins):
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
        if state.epoch <= self.freeze_until:  # freeze
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


if args.freeze_until >= 0:
    freezing_callback = FreezingCallback(["encoder.encoder"], args.freeze_until, trainer)
    trainer.add_callback(freezing_callback)

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
