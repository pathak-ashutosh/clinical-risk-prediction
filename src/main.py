import os
import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from src.utils import seed_everything
from transformers import LlamaForCausalLM, LlamaTokenizer

load_dotenv()
login(token=os.environ["HF_TOKEN"])

# Empty the cache
torch.cuda.empty_cache()

# Confirm that the GPU is detected
assert torch.cuda.is_available()

# Get the GPU device name.
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")
device = torch.device("cuda")

# Set seed for reproducibility
seed_everything()

# Load the data
data_path = "./data/test_diabetes_filtered.csv"
df = pd.read_csv(data_path) # Load only the first 4 columns 

# Load the model and tokenizer
model_id = "meta-llama/Llama-2-7b-chat-hf"
torch.no_grad()
model =  LlamaForCausalLM.from_pretrained(model_id)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

torch.save(model.state_dict(), f'../models/{model_id}.pth')