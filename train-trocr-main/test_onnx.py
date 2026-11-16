from mlflow.data.sources import LocalArtifactDatasetSource
from optimum.onnxruntime import ORTModelForVision2Seq
from mlflow.data.pandas_dataset import PandasDataset
from transformers import TrOCRProcessor
from evaluate import load
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import onnxruntime
import argparse
import mlflow
import torch
import json
import time

parser = argparse.ArgumentParser('arguments for the code') 

parser.add_argument('--experiment_name', type=str, default="experiment_name", help='Name of the mode that is tested.') 
parser.add_argument('--model_path', type=str, default="/path/to/model", help='Path to base model.') 
parser.add_argument('--model_config_path', type=str, default="/path/to/model/config.json", help='Path to model config file.') 
parser.add_argument('--processor_path', type=str, default="/path/to/processor/", help='Path to trocr processor')
parser.add_argument('--test_data_path', type=str, default="/path/to/test/data.csv", help='Path to .csv file containing the test data.')  
parser.add_argument('--csv_path', type=str, default="/path/to/result/file.csv", help='Path for saving result .csv') 
parser.add_argument('--img_width', type=int, default=1024, help='Width of the input line image')
parser.add_argument('--img_height', type=int, default=192, help='Height of the input line image') 

args = parser.parse_args() 

# Give a name to the MLflow experiment
mlflow.set_experiment(args.experiment_name)

# Allow logging for GPU use, CPU use etc.
mlflow.enable_system_metrics_logging()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 0
sess_options.inter_op_num_threads = 0

processor = TrOCRProcessor.from_pretrained(args.processor_path)
model = ORTModelForVision2Seq.from_pretrained(args.model_path, session_options=sess_options, use_cache=False, use_io_binding=False, provider="CUDAExecutionProvider")

def open_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# Initialize metrics
cer_metric = load("cer")
wer_metric = load("wer")

# Load test data to dataframe
test_df = pd.read_csv(args.test_data_path)
# Reset the indices to start from zero
test_df.reset_index(drop=True, inplace=True)

print("Number of test examples: ", test_df.shape[0]) 

# Get prediction for each line image listed in dataframe
processing_times = []
for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    start_time = time.time()
    image = Image.open(row.file_name).convert("RGB")
    width, height = image.size     
    if (width, height) != (args.img_width, args.img_height):  
        image = image.resize((args.img_width, args.img_height))
    pixel_values = processor(image, return_tensors="pt").pixel_values 
    generated_ids = model.generate(pixel_values.to(device), max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    test_df.loc[idx, 'predicted'] = generated_text
    processing_times.append(time.time() - start_time)

# Calculate CER and WER metrics by comparing predicted and ground truth texts
cer_metric.add_batch(predictions=test_df.predicted, references=test_df.text)
wer_metric.add_batch(predictions=test_df.predicted, references=test_df.text)

cer_score = cer_metric.compute()
wer_score = wer_metric.compute()

print(f"CER score of model: {cer_score}")
print(f"WER score of model: {wer_score}")

test_df.to_csv(args.csv_path, sep=";", encoding='utf-8', index=False)

mean_processing_time = np.mean(processing_times)

# Create a MLflow Dataset object
mlflow_df: PandasDataset = mlflow.data.from_pandas(
    test_df, source=LocalArtifactDatasetSource(args.test_data_path), targets="text", predictions="predicted"
)

# Initiate the MLflow run context
with mlflow.start_run() as run:
     # Log dataset information
    mlflow.log_input(mlflow_df, context="testing")
    # Log metrics
    mlflow.log_metrics({"CER": cer_score, "WER": wer_score, "Mean processing time per image": mean_processing_time})
    # Load and log the parameters of the model
    param_dict = open_json(args.model_config_path)
    mlflow.log_params(param_dict)