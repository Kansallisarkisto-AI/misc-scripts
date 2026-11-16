import os

# Set GPU 0 as the default device used for running the code
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoFeatureExtractor, AutoModel, AutoConfig, AutoModelForCausalLM, VisionEncoderDecoderConfig, AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator 
from transformers.generation import GenerationConfig
from torch.utils.data import Dataset 
import torch.nn as nn
from evaluate import load 
from PIL import Image 
import pandas as pd 
import argparse 
import mlflow
import torch 

from utils import load_annotations 

from augments import OwnAug

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
parser.add_argument('--experiment_name', type=str, default="experiment name", help='Name for the MLflow experiment.') 
parser.add_argument('--model_name', type=str, default="model_name", help='Name of the model that is trained.') 
parser.add_argument('--encoder_path', type=str, default="facebook/deit-tiny-patch16-224", help='Path to encoder.') 
parser.add_argument('--decoder_path', type=str, default="ai-forever/ruBert-base", 
                    help='Path to decoder')  # xlm-roberta-large
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Training batch size.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='Evaluation batch size.')
parser.add_argument('--augment', type=int, default=0,
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
parser.add_argument('--img_height', type=int, default=192, help='Height of the input line image') 
parser.add_argument('--device', type=str, default="cuda", 
                    help='"device rocm or cuda"')

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Give a name to the MLflow experiment
mlflow.set_experiment(args.experiment_name)

# Allow logging for GPU use, CPU use etc.
mlflow.enable_system_metrics_logging()

# Path for saving the trained model and processor files
SAVE_MODEL_PATH = args.output_path + args.model_name + '/'

# get_processor and get_model under Apache 2.0 license in manga-ocr-license.txt, from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr_dev/training/train.py
def get_processor(encoder_name, decoder_name):
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessor(feature_extractor, tokenizer,
                                           use_fast=True,
                                           do_resize=True, 
                                           size={'height': args.img_height,'width': args.img_width})
    return processor

def get_model(encoder_name, decoder_name, max_length, num_decoder_layers=None):
    encoder_config = AutoConfig.from_pretrained(encoder_name)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder = AutoModel.from_config(encoder_config)

    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)

    if num_decoder_layers is not None:
        if decoder_config.model_type == "bert":
            decoder.bert.encoder.layer = decoder.bert.encoder.layer[-num_decoder_layers:]
        elif decoder_config.model_type in ("roberta", "xlm-roberta"):
            decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[-num_decoder_layers:]
        else:
            raise ValueError(f"Unsupported model_type: {decoder_config.model_type}")

        decoder_config.num_hidden_layers = num_decoder_layers

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    config.tie_word_embeddings = False
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    processor = get_processor(encoder_name, decoder_name)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor

model, processor = get_model(args.encoder_path, args.decoder_path, 128, num_decoder_layers=2)

model.to(device)

# Initialize metrics
cer_metric = load("cer")
wer_metric = load("wer")

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

# Torch dataset for processing and providing data to the model
class TextlineDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128, augment=False):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.augmentator = OwnAug()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # prepare image (i.e. resize + normalize)
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

# https://github.com/NielsRogge/Transformers-Tutorials/issues/70
model.config.decoder.is_decoder = True
model.config.decoder.add_cross_attention = True

# OPTIONAL: introduce a smaller (than in the original decoder model) bottleneck into the output projection and embedding layers
#bottleneck_size = model.config.decoder.hidden_size
# initialize output projection and embedding layers from scratch
# https://github.com/huggingface/transformers/issues/19329
#model.decoder.output_projection = nn.Linear(bottleneck_size, processor.tokenizer.vocab_size)

# check here https://github.com/microsoft/unilm/blob/master/trocr/trocr_models.py
#model.decoder.embed_tokens = nn.Embedding(processor.tokenizer.vocab_size, bottleneck_size, padding_idx=processor.tokenizer.pad_token_id)

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
model.config.max_length = 128 
model.config.early_stopping = True 
model.config.no_repeat_ngram_size = 3 
model.config.length_penalty = 2
model.config.num_beams = 4 
gen_cfg = GenerationConfig.from_model_config(model.config)

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
    report_to="mlflow",
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    #eval_accumulation_steps=8,  # NOTE: might need this?
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    num_train_epochs=args.epochs,
    save_total_limit=1,
    output_dir=SAVE_MODEL_PATH,
    optim='adamw_torch',
    fp16=args.fp16, 
    bf16=args.bf16,
    tf32=args.tf32, 
    dataloader_num_workers=1,
    dataloader_drop_last=True,
    learning_rate=args.lr,
    lr_scheduler_type=args.scheduler,
    warmup_ratio=args.warmup_ratio,
    fp16_full_eval=eval_fp16,
    bf16_full_eval=eval_bf16,
    generation_config=gen_cfg
)

# Function for computing CER and WER metrics for the prediction results
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    if any(pred_str) and any(label_str):
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
    else:
        cer = 0
        wer = 0   

    return {"cer": cer, "wer": wer}

def my_collate(ins):
    thedict = default_data_collator(ins)
    thedict["interpolate_pos_encoding"] = True
    return thedict

# instantiate trainer
# For all parameters see: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,  
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=my_collate,
)

# Train the model
trainer.train()
# Save model and processor
model.save_pretrained(SAVE_MODEL_PATH, safe_serialization=False)
processor.save_pretrained(SAVE_MODEL_PATH + "/processor")

