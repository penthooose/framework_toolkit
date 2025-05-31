import torch
from transformers import TrainingArguments


path = r"c:\Users\Paul\.cache\training_output\FT 1.0\checkpoints_llama3_german_test_run\final_model\training_args.bin"

# Method 1: Add TrainingArguments to safe globals list
torch.serialization.add_safe_globals([TrainingArguments])
training_args = torch.load(path, weights_only=False, map_location="cpu")

# Method 2: Use context manager approach
with torch.serialization.safe_globals([TrainingArguments]):
    training_args = torch.load(path, weights_only=False, map_location="cpu")

# Print parameters
for key, value in vars(training_args).items():
    print(f"{key}: {value}")
