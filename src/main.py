import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from huggingface_hub import login
from src.config import code
from src.eval import get_predictions
from src.utils import seed_everything
from src.preprocess import create_test_prompt, get_disease_name
from transformers import LlamaForCausalLM, LlamaTokenizer

login(token='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')


# Confirm that the GPU is detected
assert torch.cuda.is_available()

# Get the GPU device name.
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")
device = torch.device("cuda")
# Empty the cache
torch.cuda.empty_cache()

# Set seed for reproducibility
seed_everything()

# Set variables to be used for saving and loading results
text_column = "Text"
label_column = "Text_label"
prompt = "prompt2"
data_path = "./data/test_diabetes_filtered.csv"
output = "MIMIC_inference_small_"
chkpt="/checkpoint-2000"
save_name = output+"_"+code+"_"+prompt+'_yes_no_'+chkpt[1:]

# Load the data
df = pd.read_csv(data_path)

# Load the model and tokenizer
model_id = "meta-llama/Llama-2-7b-chat-hf"
model_path = "./models/Llama-2-7b-chat-hf"
model =  LlamaForCausalLM.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

model.to(device)

# Get full disease name
code_text = get_disease_name(code)

# Convert each row to a prompt
list_examples =[create_test_prompt(row, code_text, text_column, label_column) for i, row in df.iterrows()]
labels = list(df["Label"].values)

# Get predictions (between 0 and 1)
pred = get_predictions(list_examples, model, tokenizer, device)

# Store the results into a new df
result_df = pd.DataFrame(columns =["patientID","predictions","label"])
result_df["patientID"] = df["patientId"].values
result_df["predictions"] = pred
result_df["label"] = labels

# Show results
print(result_df)

# Save results to file
result_df.to_csv("Predictions_" +save_name+".csv", index=False)

# Calculate and show metrics
roc = roc_auc_score(labels, pred)
pr_auc = average_precision_score(labels, pred)
acc = accuracy_score(labels, [1 if i>=0.5 else 0 for i in pred])

print("ROC Area under curve: ", roc)
print("Avg Precision Score: ", pr_auc)
print("Accuracy: ", acc)
