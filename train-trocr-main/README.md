# Train TrOCR


This repository contains code for training a TrOCR handwritten text recognition model,
using the implementation available via the transformers library. Additional code files 
can be used for hyperparameter tuning and inference.

More information on TrOCR at [Hugging Face](https://huggingface.co/docs/transformers/model_doc/trocr).

Example code for fine-tuning a TrOCR model on [Google Colab](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb#scrollTo=mcQMbxi10SDm).

## Input data

The training code at the ```train.py``` file expects the model input as a .csv file. The data should have the following form: 

| file_name  | text |
| ------------- | ------------- |
| /path/to/line_image_1.jpg  | Text content of line_image_1.jpg  |
| /path/to/line_image_2.jpg  | Text content of line_image_2.jpg  |

The images should preferably contain individual text lines. 

## Alternative input data

An alternative way of loading input data is to input a config yaml file containing train paths inside 'train' argument and validation paths as a 'valid' argument. This way you can easily add or remove different datasets from training or validation. Supports both paths to txt and csv annotatation files. The txt files contain paths to images and their corresponding ground truth text per line separated by a comma. An example is presented below:

| /path/to/line_image_1.jpg,Text content of line_image_1.jpg  |

| /path/to/line_image_2.jpg,Text content of line_image_2.jpg  |

## Training parameters

There is a wide variety of model parameters that can be modified: more information is available for example [here](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments). The default values in the code are not necessarily best ones for every use case, so testing and parameter tuning is recommended.

## TF32 training

If you have access to an Ampere architecture or newer Nvidia gpu, you can use tf32 datatype in training. It almost halves the training time and produces similar accuracies as fp32 training. More information can be found [here](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/).

## Image augmentations

The ```augments.py``` file contains a class that implements some image augmentations that have generally improved training results.
It is good to test different augmentations and parameter values for the specific task and data at hand. 

### Hyperparameter tuning

The ```tune.py``` file contains code for hyperparameter tuning using the Optuna-library together with 
the Trainer-class. More information is available for example [here](https://huggingface.co/learn/cookbook/optuna_hpo_with_transformers).
The ```visualize_tune_results.py``` code can be used to generate plots of optimization history, parameter importance
and intermediate parameter values.

### Testing the model

File ```test.py``` contains code for testing the model in .pt form, while ```test_onnx.py``` has code for testing the .onnx version of the model.

### ONNX transform

Usually inference speeds up if the Pytorch model is transformed into .onnx format. Transformation can be done for instance by using the following code in the console:  

```optimum-cli export onnx --opset 14 --task image-to-text --model /path/to/pytorch/model /path/to/onnx/model```

The ```opset``` parameter sets the ONNX opset version to export the model with, and might have to be adjusted.

## Performance metrics

Character Error Rate (CER) has been used as the main metric for evaluating model performance. The code also measures Word Error Rate (WER), as well as training and evaluation loss.  

Training and testing results are logged using MLFlow. It will create ```mlruns``` folder for saving the logs, which can be visualized using the MLflow ui using command ```mlflow ui --host <host> --port <port>```.