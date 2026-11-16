
import os

# Set GPU 0 as the default device used for running the code
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator 
from optuna.storages import RDBStorage
from torch.utils.data import Dataset 
from evaluate import load 
from PIL import Image 
import pandas as pd 
import numpy as np
import argparse 
import optuna
import mlflow
import torch 

from augments_new import OwnAug

parser = argparse.ArgumentParser('arguments for the code') 

parser.add_argument('--root_path', type=str, default="", help='Root path to data files. When .csv contains complete path to files, this is empty string.') 
parser.add_argument('--data_config', type=str, default="", help='Path to .yaml file containing paths to train and valid annotation .txt files. OPTIONAL') 
parser.add_argument('--tr_data_path', type=str, default="/path/to/train.csv",
                    help='Path to .csv file containing the training data.')
parser.add_argument('--val_data_path', type=str, default="/path/to/val.csv",
                    help='Path to .csv file containing the validation data.')
parser.add_argument('--output_path', type=str, default="/output/path",
                    help='Path for saving training results.')
parser.add_argument('--experiment_name', type=str, default="experiment name", help='Name for the MLflow experiment.') 
parser.add_argument('--study_name', type=str, default="study_name", help='Name for the Optuna study.') 
parser.add_argument('--model_name', type=str, default="model_name", help='Name of the model that is trained.') 
parser.add_argument('--model_path', type=str, default="/path/to/model/", help='Path to base model.') 
parser.add_argument('--processor_path', type=str, default="/path/to/processor/", help='Path to trocr processor') 
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (per model).') 
parser.add_argument('--n_trials', type=int, default=20, help='Number of trials with different parameter values.') 
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.') 
parser.add_argument('--device', type=str, default="cuda:0", help='Device used for training.') 
parser.add_argument('--augment', type=int, default=1, help='Defines if image augmentations are used during training.') 
parser.add_argument('--tf32', type=bool, default=True, help='Use tf32') 
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.') 
parser.add_argument('--img_width', type=int, default=1024, help='Width of the input line image')
parser.add_argument('--img_height', type=int, default=192, help='Height of the input line image') 

args = parser.parse_args() 

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Give a name to the MLflow experiment
mlflow.set_experiment(args.experiment_name)

# Allow logging for GPU use, CPU use etc.
mlflow.enable_system_metrics_logging()

# Stores hyperparameter tuning data into a local database
storage = RDBStorage("sqlite:///optuna_trials.db")

# Create local database for saving optuna data
study = optuna.create_study(
    study_name=args.study_name, direction="minimize", storage=storage, load_if_exists=True
)

# Path for saving the trained model and processor files
SAVE_MODEL_PATH = args.output_path + args.model_name + '/'

# Initialize processor and model 
processor = TrOCRProcessor.from_pretrained(args.processor_path, 
                                           use_fast=True, 
                                           do_resize=True, 
                                           size={'height': args.img_height,'width': args.img_width})  
 
def model_init():
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path) 
    model.to(device)
    # Define model configuration 
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
    model.config.interpolate_pos_encoding = True
    if args.tf32:  
        torch.backends.cuda.matmul.allow_tf32 = True     
        torch.backends.cudnn.allow_tf32 = True 
    return model

# Initialize metrics 
cer_metric = load("cer") 
wer_metric = load("wer") 

# Load train and validation data to Pandas dataframes     
train_df = pd.read_csv(args.tr_data_path) 
val_df = pd.read_csv(args.val_data_path)
# Reset the dataframe row indices to start from zero     
train_df.reset_index(drop=True, inplace=True)     
val_df.reset_index(drop=True, inplace=True) 
    
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
        # add augmentations         
        if self.augment:    
            w,h = image.size
            if (w > 25) and (h > 25):         
                image = self.augmentator(image)    
        
        # get data in model input form from processor         
        pixel_values = self.processor(image, return_tensors="pt").pixel_values           
        # add labels (input_ids) by encoding the text         
        labels = self.processor.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_target_length).input_ids         
        # important: make sure that PAD tokens are ignored by the loss function         
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]         
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}         
        return encoding 
        
# Create train and validation datasets 
train_dataset = TextlineDataset(root_dir=args.root_path, df=train_df, processor=processor, max_target_length=128, augment=args.augment) 
eval_dataset = TextlineDataset(root_dir=args.root_path, df=val_df, processor=processor, max_target_length=128, augment=False) 
    
print("Number of training examples:", len(train_dataset)) 
print("Number of validation examples:", len(eval_dataset)) 
    
# Allow use of tf32 data type if args.tf32:   
if args.tf32:  
    torch.backends.cuda.matmul.allow_tf32 = True     
    torch.backends.cudnn.allow_tf32 = True 

# Set arguments for model training 
# For all arguments see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments 
training_args = Seq2SeqTrainingArguments(     
                            report_to="mlflow",
                            predict_with_generate=True,     
                            evaluation_strategy="epoch",     
                            save_strategy="epoch",     
                            logging_strategy="epoch",            
                            load_best_model_at_end=True,     
                            metric_for_best_model='cer',    
                            per_device_eval_batch_size=16,
                            greater_is_better=False,     
                            fp16=False,     
                            num_train_epochs=args.epochs,     
                            save_total_limit=1,     
                            output_dir=SAVE_MODEL_PATH,     
                            optim='adamw_torch',     
                            tf32=args.tf32,     
                            dataloader_num_workers=8,     
                            dataloader_drop_last=True,     
                            fp16_full_eval=True
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

def compute_objective(metrics):
    return metrics["eval_cer"]

def my_collate(ins):
    thedict = default_data_collator(ins)
    thedict["interpolate_pos_encoding"] = True
    return thedict

# Instantiate trainer 
# For all parameters see: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer 
trainer = Seq2SeqTrainer(model_init=model_init,     
                        tokenizer=processor.image_processor,    
                        args=training_args,     
                        compute_metrics=compute_metrics,     
                        train_dataset=train_dataset,     
                        eval_dataset=eval_dataset, 
                        data_collator=my_collate) 


def optuna_hp_space(trial):
    """Define hyperparameters for tuning."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "polynomial"]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }

best_run = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=args.n_trials,
    compute_objective=compute_objective,
    study_name=args.study_name,
    storage="sqlite:///optuna_trials.db",
    load_if_exists=True,

)

print(best_run)