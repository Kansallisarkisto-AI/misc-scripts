import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set GPU 0 as the default device used for running the code
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator 
from transformers.generation import GenerationConfig
from torch.utils.data import Dataset 
import torch.nn as nn
from evaluate import load 
from PIL import Image 
import pandas as pd 
import argparse 
import mlflow
import torch

import torchvision.transforms.functional as F

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
parser.add_argument('--base_model_path', type=str, default="microsoft/trocr-large-handwritten", help='Path to base model.') 
parser.add_argument('--tokenizer_path', type=str, default="ai-forever/ruRoberta-large", 
                    help='Path to tokenizer')  # xlm-roberta-large
parser.add_argument('--num_decoder_layers', type=int, default=12,
                    help='Number of layers kept in the decoder language model. Can be e.g. 2-6 for bert and 4-12 for roberta')
parser.add_argument('--reset_embed_tokens', type=bool, default=True,
                    help='Whether to reset embed_tokens layer')
parser.add_argument('--retrain_small_tokenizer', type=bool, default=False,
                    help='Whether to retrain tokenizer as a smaller mostly character and bigram-level tokenizer.')
parser.add_argument('--tokenizer_data', type=str, default="corpus_cleaned.txt", 
                    help='Path to tokenizer training data')  # xlm-roberta-large
parser.add_argument('--vocab_size', type=int, default=5000,
                    help='Vocabulary size for retraining tokenizer')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Training batch size.')
parser.add_argument('--eval_batch_size', type=int, default=1,
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
parser.add_argument('--img_height', type=int, default=192, help='Height of the input line image') # was 128
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


# Initialize processor and model
# By default processor resizes image to 384 x 384
# images resized by custom padding function
processor = TrOCRProcessor.from_pretrained(args.base_model_path,
                                           use_fast=True,
                                           do_resize=True,
                                           size={'height': args.img_height, 'width': args.img_width})

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

# Load annotations
if args.data_config == '':
    # Load train and validation data to Pandas dataframes
    train_df = pd.read_csv(args.tr_data_path).dropna()
    if args.val_data_path == "/path/to/val.csv":
        # Take 10 000 random samples for validation
        val_df = train_df.sample(n=10_000, random_state=1)
        
        # Remove those samples from train_df
        train_df = train_df.drop(val_df.index)
    else:
        val_df = pd.read_csv(args.val_data_path).sample(n=10000, random_state=1)  # limit to 10 000 for now

    # Reset the dataframe row indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
else:
    # Load annotations from a config file
    train_df, val_df = load_annotations(args.data_config)

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
    '''with open(args.tokenizer_data, 'r') as f:
        corpus = f.readlines()

    def get_training_corpus():
        return (
            corpus[i : i + 1000]
            for i in range(0, len(corpus), 1000)
        )'''
    training_corpus = get_training_corpus()

    # retrain tokenizer with a much simpler vocabulary
    tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=args.vocab_size, initial_alphabet=characters_to_add)
    if args.vocab_size <= 500:
        tokenizer.add_tokens(characters_to_add)
    # optionally ensure some tokens are added
    #tokenizer.add_tokens(["–", "“", "…", "№"])

# swap out tokenizer for custom one
processor.tokenizer = tokenizer

# save processor now, also at end
processor.save_pretrained(SAVE_MODEL_PATH + "/processor")

model = VisionEncoderDecoderModel.from_pretrained(args.base_model_path)

decoder = AutoModelForCausalLM.from_pretrained(args.tokenizer_path, is_decoder=True, add_cross_attention=True, tie_word_embeddings=False)
decoder_type = decoder.config.model_type

# take only the last args.num_decoder_layers from the decoder language model
if args.num_decoder_layers is not None:
    if decoder_type == "bert":
        decoder.bert.encoder.layer = decoder.bert.encoder.layer[-args.num_decoder_layers:]
    elif decoder_type in ("roberta", "xlm-roberta"):
        decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[-args.num_decoder_layers:]
    else:
        raise ValueError(f"Unsupported model_type: {decoder_type}")

# swap out decoder for custom one
model.decoder = decoder
model.decoder.config.num_hidden_layers = args.num_decoder_layers
# update new decoder's config to match the original trocr checkpoints
model.decoder.config.tie_word_embeddings = False
#model.decoder.config.scale_embedding = False

model.to(device)

# Initialize metrics
cer_metric = load("cer")
wer_metric = load("wer")



# image padding and resize from https://github.com/pytorch/vision/issues/6236
class ResizeWithPad:
    def __init__(self, w=1024, h=192):
        self.w = w
        self.h = h

    def __call__(self, image):
        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        diff = max(ratio_1, ratio_f) / min(ratio_1, ratio_f) - 1.0
        # check if the original and final aspect ratios are the same within a margin
        # if round(ratio_1, 2) != round(ratio_f, 2):
        if diff > 1.0:  # if one of the images is more than 100% wider than the other
            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 255, "constant")
                return F.resize(image, [self.h, self.w])
            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 255, "constant")
                return F.resize(image, [self.h, self.w])
            else:  # fallback
                return F.resize(image, [self.h, self.w])
        else:
            return F.resize(image, [self.h, self.w])

# Torch dataset for processing and providing data to the model
class TextlineDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=192, augment=False):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.augmentator = OwnAug()
        self.resizer_padder = ResizeWithPad(w=args.img_width, h=args.img_height)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]  #.replace('Ъ', 'ъ')  # normalize endings, figure out something better

        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")

        # Add augmentations
        if self.augment:    
            w,h = image.size
            # Augmentations require minimum image size
            if (w > 25) and (h > 25):         
                image = self.augmentator(image)


        # DISABLED: resize and pad first
        #image = self.resizer_padder(image)
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

model.config.decoder.num_hidden_layers = args.num_decoder_layers

# possibility: introduce a smaller (than in the original decoder model) bottleneck into the output projection and embedding layers
bottleneck_size = model.config.decoder.hidden_size
# initialize output projection and embedding layers from scratch
# https://github.com/huggingface/transformers/issues/19329
model.decoder.output_projection = nn.Linear(bottleneck_size, len(processor.tokenizer))

if args.reset_embed_tokens:
    # check here https://github.com/microsoft/unilm/blob/master/trocr/trocr_models.py
    model.decoder.model.decoder.embed_tokens = nn.Embedding(len(processor.tokenizer), bottleneck_size, padding_idx=processor.tokenizer.pad_token_id)

# Define model configuration 
# Configure the model to expect and handle custom image sizes
model.config.interpolate_pos_encoding = True
# set special tokens used for creating the decoder_input_ids from the labels 
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
# maybe consider len(processor.tokenizer) instead: https://stackoverflow.com/questions/67412925/what-is-the-difference-between-lentokenizer-and-tokenizer-vocab-size
print("Vocab size ", len(processor.tokenizer))
model.config.decoder.vocab_size = len(processor.tokenizer) #processor.tokenizer.vocab_size
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
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    #eval_accumulation_steps=8,  # NOTE: might need this?
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    num_train_epochs=args.epochs,
    save_total_limit=10,
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
    generation_config=gen_cfg,
    save_safetensors=False
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
