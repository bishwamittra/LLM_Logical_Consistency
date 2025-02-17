import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()


from transformers import AutoModelForCausalLM, AutoTokenizer

model_param = ['7b', '13b'][0]
# Step 0: Check directory
os.system("mkdir -p base_models/")
os.system(f"rm -r base_models/Llama-2-{model_param}-chat-hf") # remove the existing model
os.system(f"mkdir -p base_models/Llama-2-{model_param}-chat-hf")

# Step 1: Download the model
model_name = f"meta-llama/Llama-2-{model_param}-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Save the model locally
model_path = f"base_models/Llama-2-{model_param}-chat-hf"
model.save_pretrained(f"{model_path}")
tokenizer.save_pretrained(f"{model_path}")

# Step 3: Load the model from local files
# loaded_model = AutoModelForCausalLM.from_pretrained("/path/to/save/model")
# loaded_tokenizer = AutoTokenizer.from_pretrained("/path/to/save/tokenizer")
