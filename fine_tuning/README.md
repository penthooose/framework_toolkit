# Fine-Tuning Module

## Overview

The Fine-Tuning module provides a bridge between Elixir and PyTorch for seamless model fine-tuning workflows. It uses ErlPort to establish communication between the Elixir application and Python-based PyTorch processes, enabling you to fine-tune AI models with configurable parameters defined in JSON files.

## Key Features

- **ErlPort Integration**: Seamless interoperability between Elixir and Python environments
- **JSON Configuration**: Flexible configuration of fine-tuning parameters via JSON files
- **PyTorch Backend**: Leverages PyTorch's powerful deep learning capabilities
- **Checkpoint Management**: Automatic saving and loading of model checkpoints
- **Metrics Tracking**: Monitor training progress with comprehensive evaluation metrics
- **Multi-model Support**: Compatible with various transformer-based models

## Requirements

### System Requirements

- Elixir 1.16 or higher
- Python 3.10 or higher
- PyTorch 1.10 or higher
- CUDA-compatible GPU (recommended for efficient training)

### Python Dependencies

```
torch
transformers
datasets
accelerate
tqdm
```

## Installation

1. Clone the repository and navigate to the fine_tuning directory:

```bash
cd fine_tuning
```

2. Install Elixir dependencies:

```bash
mix deps.get
mix compile
```

3. Set up Python environment with required dependencies:

```bash
python -m pip install torch transformers datasets accelerate tqdm
```

4. Ensure your Python path is correctly configured in the Elixir application.

## Usage

### Basic Workflow

1. Create a JSON configuration entry specifying your fine-tuning parameters:

```json
{
  "model_name": "safetensors_files/mistralai/Mistral-7B-v0.1",
  "training_parameters": {
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100
  },
  "dataset": {
    "path": "./data/training_data.jsonl",
    "validation_split": 0.1
  },
  "output_dir": "./fine_tuned_models/my_model",
  "quantization": {
    "bits": 4,
    "use_double_quant": true
  }
}
```

## How It Works

1. The Elixir application starts a Python process using ErlPort
2. Configuration parameters are sent from Elixir to Python
3. The Python process loads the model and dataset
4. Fine-tuning runs in Python with PyTorch
5. Progress and results are communicated back to Elixir
6. The fine-tuned model is saved to the specified output directory

## Model Merging

This module supports model merging capabilities to combine different models or integrate LoRA adapters. These capabilities are essential for creating optimized models that combine the strengths of multiple fine-tuned versions.

### LoRA Adapter Merging

Fine-tuned LoRA adapters can be merged back into base models to create standalone models. Here's a configuration example:

```json
{
  "base_model_path": "./models/hub/mistral-7b-instruct",
  "adapter_path": "./training_output/checkpoints_unsupervised_1/final_model",
  "output_path": "./merged_models/merged_unsupervised_1",
  "use_fp16": true
}
```

The merging process can be done sequentially to build on previous merges:

```json
{
  "base_model_path": "./merged_models/merged_unsupervised_1",
  "adapter_path": "./training_output/checkpoints_unsupervised_2/final_model",
  "output_path": "./merged_models/merged_unsupervised_2",
  "use_fp16": true
}
```

### MergeKit Integration

For more complex model merging strategies, this module uses MergeKit which requires a YAML configuration file. This is essential for techniques like SLERP (Spherical Linear Interpolation) that allow for sophisticated blending of model weights.

Example YAML configuration (`merge_models_params.yaml`):

```yaml
slices:
  - sources:
      - model: ./merged_models/unsupervised_model
        layer_range: [0, 32]
      - model: ./merged_models/supervised_model
        layer_range: [0, 32]
merge_method: slerp
base_model: ./merged_models/unsupervised_model
parameters:
  t:
    # Lower layers (0-10): 60% model1, 40% model2
    - filter: "self_attn.*(0|1|2|3|4|5|6|7|8|9|10)"
      value: 0.4
    - filter: "mlp.*(0|1|2|3|4|5|6|7|8|9|10)"
      value: 0.4

    # Middle layers (11-21): 50% each
    - filter: "self_attn.*(11|12|13|14|15|16|17|18|19|20|21)"
      value: 0.5
    - filter: "mlp.*(11|12|13|14|15|16|17|18|19|20|21)"
      value: 0.5

    # Upper layers (22-32): 30% model1, 70% model2
    - filter: "self_attn.*(22|23|24|25|26|27|28|29|30|31|32)"
      value: 0.7
    - filter: "mlp.*(22|23|24|25|26|27|28|29|30|31|32)"
      value: 0.7

    # Default for any unmatched layers
    - value: 0.5
dtype: float16
```

This YAML configuration allows for sophisticated merging with layer-specific blending parameters. The JSON configuration that references this YAML file looks like:

```json
{
  "yaml_config_path": "./external/merge_models_params.yaml",
  "output_dir": "./merged_models/merged_hybrid_model",
  "cpu_offload": true
}
```

## Troubleshooting

### Common Issues

- **ErlPort Connection Errors**: Ensure Python path is correctly set in your environment
- **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
- **Module Not Found Errors**: Check all Python dependencies are installed
