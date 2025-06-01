"""
Fine-Tuning with PyTorch

This script provides functionality for unsupervised/supervised fine-tuning of transformer models
using PyTorch and PEFT (Parameter-Efficient Fine-Tuning).
"""

import os
import json
import torch
import gc
import logging
import sys
import importlib
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch.serialization
from contextlib import contextmanager
import numpy as np

# Try to import 8-bit optimizer support
try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
    print("bitsandbytes available for 8-bit optimization")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not available - 8-bit optimization disabled")


# Import specific numpy components that might be in checkpoints
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype, generic


# Function to register all needed numpy components as safe globals
def register_numpy_safe_globals():
    """
    Register all commonly used numpy components for safe checkpoint loading.
    This is needed for PyTorch 2.6+ which has stricter security for unpickling.
    """
    # Register specific numpy classes and functions directly
    logger.info("Registering numpy component classes as safe globals")
    numpy_classes = [
        # Core numpy classes used in serialization
        np.ndarray,
        np.dtype,
        np.generic,
        # Core multiarray components
        np.core.multiarray._reconstruct,
        # Additional numpy types that might be in checkpoints
        np.int64,
        np.float32,
        np.float16,
        np.bool_,
        # Functions used in array creation
        np.array,
        np.zeros,
        # Additional numpy dtypes needed for checkpoints
        np.uint32,  # Add UInt32DType
        np.dtypes.UInt32DType,  # Direct reference to UInt32DType
        np.int32,
        np.uint8,
        np.uint16,
        np.uint64,
        np.int8,
        np.int16,
        # Add all numpy dtype classes
        *[dt for dt in np.sctypeDict.values() if isinstance(dt, type)],
    ]

    # Register the class objects directly
    torch.serialization.add_safe_globals(numpy_classes)

    # Using PyTorch's safe_registry for module names
    logger.info("Registering module names in the PyTorch safe registry")
    module_names = ["numpy", "numpy.core", "numpy.core.multiarray"]

    # Access the safe registry directly and add the module names
    if hasattr(torch.serialization, "safe_registry"):
        for module_name in module_names:
            if module_name not in torch.serialization.safe_registry:
                torch.serialization.safe_registry.add(module_name)
                logger.info(f"Added module name to safe registry: {module_name}")
    else:
        logger.warning(
            "torch.serialization.safe_registry not available, using alternative registration"
        )

        # Fixed approach: only register actual objects, not modules
        try:
            # Don't register the module itself, just its specific components
            logger.info("Registering specific numpy components")

            # Just ensure all the classes from numpy_classes get registered
            for cls in numpy_classes:
                if isinstance(cls, type) or callable(cls):
                    torch.serialization.add_safe_globals([cls])

        except Exception as e:
            logger.warning(f"Error in alternative registration: {e}")

    # Register common array construction patterns
    logger.info("Registering numpy array creation patterns")
    try:
        # This matches common pickle patterns for numpy arrays
        torch.serialization._get_safe_dtype = lambda dtype_str: np.dtype(dtype_str)
        # Don't register the module itself
        logger.info("Set up safe dtype handler for array reconstruction")
    except Exception as e:
        logger.warning(f"Error registering numpy array patterns: {e}")

    logger.info("Numpy components registered as safe globals")


class ProgressLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Only forward to callback, don't log again
            if _progress_callback:
                try:
                    _progress_callback(msg)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
            # If no callback registered, do nothing since the message
            # is already being output by the regular logger
        except Exception:
            self.handleError(record)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variable to store progress callback
_progress_callback = None
# Global variable for file logger
_file_logger = None


def setup_file_logger(
    log_dir: str, use_checkpoint: bool = False, checkpoint_path: str = None
):
    """
    Set up a file logger that writes to log_dir/logging.txt

    Args:
        log_dir: Directory to write log file
        use_checkpoint: Whether resuming from a checkpoint
        checkpoint_path: Path to the checkpoint being resumed from
    """
    global _file_logger

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "logging.txt")

    # Always use append mode to preserve previous logs
    file_mode = "a"

    # Set up a file handler with explicit UTF-8 encoding
    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Create a logger for file logging
    _file_logger = logging.getLogger("file_logger")
    _file_logger.setLevel(logging.INFO)
    # Remove any existing handlers to avoid duplicates
    for handler in _file_logger.handlers[:]:
        _file_logger.removeHandler(handler)
    _file_logger.addHandler(file_handler)
    _file_logger.propagate = False  # Don't propagate to parent logger

    logger.info(f"File logger setup to write to {log_file}")

    if use_checkpoint and checkpoint_path:
        # Add a clear separator when resuming training
        _file_logger.info("\n\n" + "=" * 50)
        _file_logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        _file_logger.info("=" * 50 + "\n")
    else:
        # Add a clear separator for a new training run
        _file_logger.info("\n\n\nStarting new Fine-tuning run:\n\n")

    logger.info(f"File logger setup to write to {log_file}")


def log_to_file(message):
    """
    Write a message to the log file.

    Args:
        message: Message to log
    """
    if _file_logger:
        _file_logger.info(message)


def register_progress_callback(callback_func):
    """
    Register a callback function to receive progress updates.

    Args:
        callback_func: Function that accepts a string message

    Returns:
        "ok" if successful
    """
    global _progress_callback
    _progress_callback = callback_func
    logger.info("Progress callback registered successfully")
    return "ok"


def send_progress(message):
    """
    Send a progress update through the callback if registered.

    Args:
        message: Progress message to send
    """
    if _progress_callback:
        try:
            _progress_callback(message)
        except Exception as e:
            # Don't use logger here to avoid potential recursion
            print(f"Error in progress callback: {e}")
    else:
        # Don't log again if message already came from logger
        # This prevents recursive logging
        if not message.startswith(("INFO:", "ERROR:", "WARNING:", "DEBUG:")):
            logger.info(message)
        else:
            # Just print to stdout instead of recursive logging
            print(message)

    # Also log important messages to file
    log_to_file(message)


# Custom logging handler to capture log messages
class ProgressLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            send_progress(msg)
        except Exception:
            self.handleError(record)


# Add our custom handler to the logger
progress_handler = ProgressLogHandler()
progress_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(progress_handler)


def test_connection():
    """
    Simple function to test if the connection to Python is working.
    """
    try:
        return "ok"
    except Exception as e:
        return f"error: {str(e)}"


def setup_environment() -> None:
    """Setup environment variables for better PyTorch performance."""
    # Set CUDA memory allocation to use expandable segments
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Log device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")


def clear_memory() -> None:
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("CUDA cache cleared")

    collected = gc.collect()
    logger.info(f"Garbage collector freed {collected} objects")


def load_finetuning_data(
    data_path: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load data from a file or directory.

    Args:
        data_path: Path to the data file or directory
        split: Dataset split to load
        text_column: Column name containing the text data
        max_samples: Maximum number of samples to load

    Returns:
        A HuggingFace Dataset object
    """
    try:
        # Check if data_path is a file or directory
        if os.path.isfile(data_path):
            # Load from JSONL file
            if data_path.endswith(".jsonl"):
                data = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            data.append(json.loads(line))

                if max_samples:
                    data = data[:max_samples]

                logger.info(f"Loaded {len(data)} examples from {data_path}")
                return Dataset.from_list(data)
            else:
                # Try using Hugging Face's load_dataset
                dataset = load_dataset(data_path, split=split)
        else:
            # Load from directory
            dataset = load_dataset(data_path, split=split)

        # Apply max_samples constraint if specified
        if max_samples and isinstance(dataset, Dataset):
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        logger.info(f"Loaded {len(dataset)} examples from {data_path}")
        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset from {data_path}: {e}")
        raise


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """
    Load and configure the tokenizer.

    Args:
        model_path: Path to the pre-trained model

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set padding token to EOS token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model max length: {tokenizer.model_max_length}")

    return tokenizer


def prepare_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    text_column: Union[str, List[str]] = "text",
    max_length: int = 2048,
    chunk_size: int = None,
) -> Dataset:
    """
    Prepare a dataset for learning.

    Args:
        dataset: The dataset to prepare
        tokenizer: Tokenizer to use
        text_column: Column(s) containing text to train on (str or list of str)
        max_length: Maximum sequence length
        chunk_size: Size of chunks to split texts (if None, use max_length)

    Returns:
        Processed dataset ready for training
    """
    # Handle multiple text columns (for mixed mode)
    if isinstance(text_column, list):
        available_columns = []
        for col in text_column:
            if col in dataset.column_names:
                available_columns.append(col)
                logger.info(f"Found text column: '{col}'")

        if not available_columns:
            columns = dataset.column_names
            logger.error(
                f"None of the text columns {text_column} found in dataset. Available columns: {columns}"
            )
            if "content" in columns:
                logger.info(f"Using 'content' column as fallback")
                available_columns = ["content"]
            elif len(columns) > 0:
                logger.info(f"Using '{columns[0]}' column as fallback")
                available_columns = [columns[0]]
            else:
                raise ValueError(
                    f"No valid text columns found in dataset with columns: {columns}"
                )

        # Creating a new combined text column
        def combine_available_columns(examples):
            result = {"text": []}
            for i in range(len(examples[dataset.column_names[0]])):
                # For each row, use the first available column that has content
                for col in available_columns:
                    if i < len(examples[col]) and examples[col][i]:
                        result["text"].append(examples[col][i])
                        break
                else:
                    # If no content found in any column, use empty string
                    result["text"].append("")
            return result

        dataset = dataset.map(
            combine_available_columns, batched=True, remove_columns=dataset.column_names
        )
        text_column = "text"  # Now use the combined column
        logger.info(f"Created combined 'text' column from {available_columns}")

    # Continue with the existing code for a single text column
    elif text_column not in dataset.column_names:
        columns = dataset.column_names
        logger.error(
            f"Text column '{text_column}' not found in dataset. Available columns: {columns}"
        )
        if "content" in columns:
            logger.info(f"Using 'content' column instead of '{text_column}'")
            text_column = "content"
        elif len(columns) > 0:
            logger.info(f"Using '{columns[0]}' column instead of '{text_column}'")
            text_column = columns[0]
        else:
            raise ValueError(
                f"Text column '{text_column}' not found and no alternative available"
            )

    # Use appropriate chunking approach
    if chunk_size is None:
        chunk_size = max_length

    def tokenize_function(examples):
        outputs = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != text_column],
    )

    logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def prepare_combined_dataset(dataset):
    """
    Combine input and output columns from a supervised dataset into a single text column.

    Args:
        dataset: Dataset with 'input' and 'output' columns

    Returns:
        Dataset with a combined 'text' column
    """

    def combine_input_output(examples):
        return {
            "text": [
                inp + "\n" + out
                for inp, out in zip(examples["input"], examples["output"])
            ]
        }

    return dataset.map(combine_input_output, batched=True)


def load_model(
    model_path: str,
    quantization_config: Dict[str, Any] = None,
    use_flash_attention: bool = True,
    device_map: str = "auto",
    freeze_partly: bool = False,
    freeze_partly_layers: int = 0,
    unfreeze_specific: bool = False,
    unfreeze_specific_layers: List[int] = None,
) -> AutoModelForCausalLM:
    """
    Load the pre-trained model with quantization.

    Args:
        model_path: Path to the pre-trained model
        quantization_config: Configuration for quantization
        use_flash_attention: Whether to use flash attention
        device_map: Device mapping strategy
        freeze_partly: Whether to freeze a portion of the model layers
        freeze_partly_layers: Number of layers to freeze from the beginning
        unfreeze_specific: Whether to unfreeze specific layers
        unfreeze_specific_layers: List of layer indices to unfreeze

    Returns:
        Loaded model
    """
    # Load model configuration
    config = AutoConfig.from_pretrained(model_path)
    if use_flash_attention:
        config.use_flash_attention_2 = True

    # Setup loading parameters
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device_map,
        "trust_remote_code": True,
        "config": config,
    }

    # Apply quantization config if provided
    if quantization_config is not None:
        bits_and_bytes_config = BitsAndBytesConfig(**quantization_config)
        model_kwargs["quantization_config"] = bits_and_bytes_config

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Set model to training mode
    model.train()

    # Prepare for k-bit training if using 4-bit or 8-bit quantization
    if quantization_config and (
        "load_in_4bit" in quantization_config or "load_in_8bit" in quantization_config
    ):
        model = prepare_model_for_kbit_training(model)
    else:
        # When not using quantization, ensure parameters require gradients
        logger.info("Setting requires_grad=True for full precision model parameters")
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Verify that parameters require gradients
        params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
        if params_requiring_grad == 0:
            logger.warning(
                "No parameters require gradients after setting requires_grad=True!"
            )

    # Implement partial freezing if requested
    if freeze_partly and freeze_partly_layers > 0:
        logger.info(
            f"Partially freezing model: first {freeze_partly_layers} layers will be frozen"
        )

        # Count total transformer layers for logging
        total_layers = len(model.model.layers)

        # Freeze specified number of layers from the beginning
        for i, layer in enumerate(model.model.layers):
            if i < freeze_partly_layers:
                logger.info(f"Freezing layer {i}/{total_layers}")
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                logger.info(f"Keeping layer {i}/{total_layers} trainable")

        # These are critical for all approaches - ensure LM head is always trainable
        logger.info("Ensuring LM head is trainable")
        for param in model.lm_head.parameters():
            param.requires_grad = True

        # Log freezing statistics
        frozen_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = frozen_params + trainable_params

        logger.info(
            f"Model partial freezing: {frozen_params}/{total_params} parameters frozen ({frozen_params/total_params:.2%})"
        )
        logger.info(
            f"Trainable: {trainable_params}/{total_params} parameters ({trainable_params/total_params:.2%})"
        )

        # Log specific information about frozen/unfrozen components
        logger.info(
            f"Frozen: first {freeze_partly_layers}/{total_layers} transformer layers"
        )
        logger.info(
            f"Trainable: last {total_layers - freeze_partly_layers}/{total_layers} transformer layers and LM head"
        )
    # Implement specific layer unfreezing if requested
    elif unfreeze_specific and unfreeze_specific_layers:
        # First freeze all layers
        total_layers = len(model.model.layers)
        logger.info(f"Freezing all {total_layers} layers first")
        for layer in model.model.layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Unfreeze only specified layers
        for i in unfreeze_specific_layers:
            # Convert to integer here - directly before using it
            try:
                layer_idx = int(i) if isinstance(i, str) else i
            except (ValueError, TypeError):
                logger.warning(f"Invalid layer index: {i}, skipping")
                continue

            if 0 <= layer_idx < total_layers:
                logger.info(f"Unfreezing specific layer {layer_idx}/{total_layers}")
                for param in model.model.layers[layer_idx].parameters():
                    param.requires_grad = True
            else:
                logger.warning(
                    f"Layer index {layer_idx} is out of range (0-{total_layers-1})"
                )

        # Ensure LM head is always trainable
        logger.info("Ensuring LM head is trainable")
        for param in model.lm_head.parameters():
            param.requires_grad = True

        # Log unfreezing statistics
        frozen_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = frozen_params + trainable_params

        logger.info(
            f"Specific layer unfreezing: {frozen_params}/{total_params} parameters frozen ({frozen_params/total_params:.2%})"
        )
        logger.info(
            f"Trainable: {trainable_params}/{total_params} parameters ({trainable_params/total_params:.2%})"
        )
        logger.info(
            f"Unfrozen layers: {unfreeze_specific_layers} out of {total_layers} layers plus LM head"
        )

    logger.info(f"Model loaded from {model_path}")

    # Log information about trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model has {all_params} parameters, {trainable_params} are trainable ({trainable_params/all_params:.2%})"
    )

    return model


def apply_peft_config(
    model: AutoModelForCausalLM, peft_config: Dict[str, Any] = None
) -> AutoModelForCausalLM:
    """
    Apply PEFT configuration to the model.

    Args:
        model: The model to configure
        peft_config: PEFT configuration parameters

    Returns:
        PEFT-configured model
    """
    # Default LoRA config if not provided
    if peft_config is None:
        peft_config = {
            "task_type": TaskType.CAUSAL_LM,
            "inference_mode": False,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }

    # Create LoRA config
    lora_config = LoraConfig(**peft_config)

    # Apply PEFT to model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def setup_training_args(
    output_dir: str, training_config: Dict[str, Any] = None
) -> TrainingArguments:
    """
    Set up training arguments.

    Args:
        output_dir: Directory to save model checkpoints
        training_config: Training configuration parameters

    Returns:
        TrainingArguments object
    """
    # Default training config if not provided
    default_config = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "warmup_steps": 100,  # Default to steps-based warmup
        "warmup_ratio": None,  # Optional ratio-based warmup (will override steps if set)
        "weight_decay": None,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "eval_strategy": "steps",
        "eval_steps": 300,
        "per_device_eval_batch_size": 1,
        "eval_accumulation_steps": 4,
        "fp16": True,
        "bf16": False,  # Add bf16 support
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "gradient_checkpointing": True,
        "report_to": "none",
        "disable_tqdm": False,
        "max_grad_norm": 1.0,
        "dataloader_num_workers": 2,
        "optim": "adamw_torch",  # Default optimizer
    }

    # Update with user-provided config
    if training_config:
        default_config.update(training_config)

    # Handle bf16 and fp16 mutual exclusivity
    if default_config.get("bf16", False):
        logger.info("Using bf16 (bfloat16) mixed precision training")
        # Disable fp16 if bf16 is enabled
        default_config["fp16"] = False

        # Check if bf16 is supported
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability[0] >= 8:  # Ampere architecture or newer
                logger.info(
                    f"BF16 supported on device with compute capability {device_capability}"
                )
            else:
                logger.warning(
                    f"BF16 may not be fully supported on device with compute capability {device_capability}"
                )
                logger.warning("Consider using fp16 instead for better compatibility")
        else:
            logger.warning("CUDA not available - bf16 training may not work properly")

    elif default_config.get("fp16", False):
        logger.info("Using fp16 (float16) mixed precision training")
        # Ensure bf16 is disabled when fp16 is used
        default_config["bf16"] = False
    else:
        logger.info("Using full precision (fp32) training")
        default_config["fp16"] = False
        default_config["bf16"] = False

    # Handle 8-bit optimizer configuration
    if default_config.get("optim") == "adamw_8bit":
        if not BITSANDBYTES_AVAILABLE:
            logger.error(
                "8-bit AdamW optimizer requested but bitsandbytes is not available"
            )
            logger.error("Please install bitsandbytes: pip install bitsandbytes")
            raise ImportError(
                "bitsandbytes is required for 8-bit optimization but is not installed"
            )

        logger.info("Using 8-bit AdamW optimizer from bitsandbytes")
        # Keep the optim setting as is - transformers will handle it

        # Log optimizer configuration
        logger.info("8-bit AdamW optimizer configuration:")
        logger.info(f"  - Learning rate: {default_config['learning_rate']}")
        logger.info(f"  - Weight decay: {default_config['weight_decay']}")
        if default_config.get("warmup_ratio"):
            logger.info(f"  - Warmup ratio: {default_config['warmup_ratio']}")
        else:
            logger.info(f"  - Warmup steps: {default_config['warmup_steps']}")

    elif default_config.get("optim") not in [None, "adamw_torch", "adamw_hf"]:
        # Validate other optimizer choices
        valid_optimizers = [
            "adamw_hf",
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_torch_xla",
            "adamw_apex_fused",
            "adafactor",
            "adamw_anyprecision",
            "sgd",
            "adagrad",
            "adamw_bnb_8bit",
            "adamw_8bit",
            "lion_8bit",
            "lion_32bit",
        ]
        if default_config["optim"] not in valid_optimizers:
            logger.warning(
                f"Unknown optimizer '{default_config['optim']}', using default 'adamw_torch'"
            )
            default_config["optim"] = "adamw_torch"

    # Create TrainingArguments object
    training_args = TrainingArguments(output_dir=output_dir, **default_config)

    return training_args


def safe_load_checkpoint(checkpoint_path):
    """
    Safely load a checkpoint with PyTorch 2.6+ compatibility.
    """
    try:
        # First register all numpy components needed for safe loading
        register_numpy_safe_globals()

        logger.info(f"Attempting to load checkpoint from {checkpoint_path}")
        # Use weights_only=False with warning about security implications
        logger.warning(
            "Using weights_only=False for checkpoint loading. This is less secure but needed for compatibility."
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logger.info(f"Successfully loaded checkpoint with weights_only=False")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise


def train_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    training_args: TrainingArguments = None,
    output_dir: str = None,
    pre_eval: bool = False,
    use_checkpoint: bool = False,
    checkpoint_path: str = None,
) -> None:
    """
    Train the model.

    Args:
        model: Model to train
        tokenizer: Tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Validation dataset for intermediate evaluation during training
        test_dataset: Test dataset for pre-training and final evaluation
        training_args: Training arguments
        output_dir: Directory to save model checkpoints
        pre_eval: Whether to run evaluation before training
        use_checkpoint: Whether to resume training from a checkpoint
        checkpoint_path: Path to the checkpoint to resume from
    """
    # Create default training args if not provided
    if training_args is None and output_dir is not None:
        training_args = setup_training_args(output_dir)
    elif training_args is None:
        raise ValueError("Either training_args or output_dir must be provided")

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create a custom callback to report progress
    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                send_progress(f"Training progress: {logs}")
                log_to_file(f"Training metrics: {logs}")

        def on_epoch_begin(self, args, state, control, **kwargs):
            send_progress(f"Starting epoch {state.epoch}/{args.num_train_epochs}")
            log_to_file(f"Starting epoch {state.epoch}/{args.num_train_epochs}")

        def on_save(self, args, state, control, **kwargs):
            send_progress(f"Saving checkpoint at step {state.global_step}")
            log_to_file(f"Saving checkpoint at step {state.global_step}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Use validation set for intermediate evaluation
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ProgressCallback()],  # Add our progress callback
    )

    # Try running a single evaluation if test_dataset is provided and pre_eval is True
    if test_dataset and pre_eval:
        try:
            logger.info("Testing evaluation with current settings...")
            logger.info(f"Test dataset size: {len(test_dataset)}")

            clear_memory()

            # Run a single evaluation pass on the test dataset
            eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
            clear_memory()

            # Use ASCII-friendly characters instead of Unicode emoji
            logger.info("\nEvaluation successful!")
            logger.info(f"Metrics: {eval_metrics}")
            send_progress(
                f"Pre-training evaluation successful. Metrics: {eval_metrics}"
            )
            log_to_file(f"Pre-training evaluation metrics: {eval_metrics}")

            # Check GPU memory usage after successful evaluation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info("\nGPU Memory Summary:")
                logger.info(f"Allocated: {allocated:.2f} GB")
                logger.info(f"Cached: {reserved:.2f} GB")
                send_progress(
                    f"GPU Memory: Allocated {allocated:.2f} GB, Cached {reserved:.2f} GB"
                )
                log_to_file(
                    f"GPU Memory: Allocated {allocated:.2f} GB, Cached {reserved:.2f} GB"
                )
        except RuntimeError as e:
            error_msg = f"Pre-training evaluation failed with error: {e}"
            logger.error(error_msg)
            send_progress(error_msg)
            log_to_file(error_msg)
            logger.warning("Continuing with training anyway...")

    # Start training
    send_progress("Starting training...")
    log_to_file("Starting training...")

    try:
        # Check if we should resume from checkpoint
        resume_from_checkpoint = None
        if use_checkpoint and checkpoint_path:
            if os.path.exists(checkpoint_path):
                if os.path.isdir(checkpoint_path):
                    resume_from_checkpoint = checkpoint_path
                    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                    send_progress(
                        f"Resuming training from checkpoint: {checkpoint_path}"
                    )
                    log_to_file(f"Resuming training from checkpoint: {checkpoint_path}")

                    # Register necessary numpy components for safe loading
                    register_numpy_safe_globals()
                else:
                    logger.warning(
                        f"Checkpoint path {checkpoint_path} is not a directory"
                    )
                    log_to_file(f"Checkpoint path {checkpoint_path} is not a directory")
                    send_progress(
                        f"Warning: Checkpoint path {checkpoint_path} is not a directory"
                    )
            else:
                logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
                log_to_file(f"Checkpoint path {checkpoint_path} does not exist")
                send_progress(
                    f"Warning: Checkpoint path {checkpoint_path} does not exist"
                )

        # Register numpy components as safe for checkpoint loading
        register_numpy_safe_globals()

        # Train the model, potentially resuming from checkpoint
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save the final model
        final_output_dir = os.path.join(training_args.output_dir, "final_model")
        send_progress(f"Training complete, saving model to {final_output_dir}")
        log_to_file(f"Training complete, saving model to {final_output_dir}")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        clear_memory()

        logger.info(
            f"Training completed successfully! Model saved to: {final_output_dir}"
        )
        logger.info(f"Training metrics: {train_result.metrics}")
        send_progress(f"Final training metrics: {train_result.metrics}")
        log_to_file(f"Final training metrics: {train_result.metrics}")

        # Evaluate on test dataset after training if provided
        if test_dataset:
            send_progress("Running final evaluation on test dataset...")
            log_to_file("Running final evaluation on test dataset...")
            final_metrics = trainer.evaluate(eval_dataset=test_dataset)
            clear_memory()
            logger.info(f"Final test metrics: {final_metrics}")
            send_progress(f"Final test metrics: {final_metrics}")
            log_to_file(f"Final test metrics: {final_metrics}")

    except Exception as e:
        error_msg = f"Training failed with error: {e}"
        logger.error(error_msg)
        send_progress(error_msg)
        log_to_file(error_msg)
        raise


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_config: Dict[str, Any] = None,
) -> str:
    """
    Generate text using the fine-tuned model.

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer to use
        prompt: Input prompt
        generation_config: Parameters for text generation

    Returns:
        Generated text
    """
    # Default generation config
    default_config = {
        "max_length": 500,
        "temperature": 0.3,
        "top_p": 0.7,
        "do_sample": True,
    }

    # Update with user config
    if generation_config:
        default_config.update(generation_config)

    # Prepare input
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            **default_config,
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def generate_text_from_controller(params):
    """
    Interface for generating text from the Elixir controller.

    Args:
        params: Dictionary containing:
            - prompt: The text prompt
            - model_path: Path to the model
            - generation_config: Optional configuration for generation

    Returns:
        Generated text
    """
    try:
        prompt = params["prompt"]
        model_path = params["model_path"]
        generation_config = params.get("generation_config")

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        tokenizer = load_tokenizer(model_path)

        # For fine-tuned models, we need to check if it's a PEFT model
        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_peft:
            from peft import PeftModel, PeftConfig

            # Load base model first
            config = PeftConfig.from_pretrained(model_path)
            base_model = load_model(
                config.base_model_name_or_path,
                quantization_config={"load_in_8bit": True},
            )

            # Then load PEFT adapter
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load regular model
            model = load_model(model_path, quantization_config={"load_in_8bit": True})

        # Generate text
        generated_text = generate_text(model, tokenizer, prompt, generation_config)

        return generated_text

    except Exception as e:
        logger.error(f"Error in generate_text_from_controller: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: {str(e)}"


def check_dataset_files(data_path):
    """
    Check for training, validation, and test JSONL files in the data_path.

    Args:
        data_path: Path to the data directory

    Returns:
        tuple: (training_file, validation_file, test_file) with full paths or None if not found
    """
    training_file = os.path.join(data_path, "training_set.jsonl")
    validation_file = os.path.join(data_path, "validation_set.jsonl")
    test_file = os.path.join(data_path, "test_set.jsonl")

    # Check if files exist
    train_exists = os.path.isfile(training_file)
    val_exists = os.path.isfile(validation_file)
    test_exists = os.path.isfile(test_file)

    logger.info(f"Dataset files in {data_path}:")
    logger.info(f"  - training_set.jsonl: {'Found' if train_exists else 'Not found'}")
    logger.info(f"  - validation_set.jsonl: {'Found' if val_exists else 'Not found'}")
    logger.info(f"  - test_set.jsonl: {'Found' if test_exists else 'Not found'}")

    return (
        training_file if train_exists else None,
        validation_file if val_exists else None,
        test_file if test_exists else None,
    )


def format_parameters(params):
    """
    Recursively format parameters coming from Erlang/Elixir to Python native types.
    Converts charlists (lists of integers) to strings, handles nested structures.

    Args:
        params: Parameters from Erlang/Elixir which may need conversion

    Returns:
        Python-native formatted parameters
    """
    if params is None:
        return None

    # Convert Erlang charlists (list of integers representing characters) to Python strings
    if isinstance(params, list) and all(
        isinstance(item, int) and 0 <= item <= 0x10FFFF for item in params
    ):
        try:
            return "".join(chr(c) for c in params)
        except (ValueError, TypeError, OverflowError):
            # If conversion fails, keep the original list
            pass

    # Handle dictionaries - recursively format all values
    elif isinstance(params, dict):
        # Create a new dictionary with formatted keys and values
        formatted_dict = {}
        for k, v in params.items():
            # Format the key - especially important for byte strings
            formatted_key = format_parameters(k)
            # If the key is a bytes object, decode it to a string
            if isinstance(formatted_key, bytes):
                formatted_key = formatted_key.decode("utf-8")

            # Special handling for unfreeze_specific_layers to ensure integers
            if formatted_key == "unfreeze_specific_layers" and isinstance(v, list):
                try:
                    # Convert all elements to integers
                    formatted_dict[formatted_key] = [
                        int(item) if isinstance(item, str) else item
                        for item in format_parameters(v)
                    ]
                    continue  # Skip the normal processing below
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Error converting unfreeze_specific_layers to integers: {e}"
                    )
                    # Fall back to normal processing

            formatted_dict[formatted_key] = format_parameters(v)
        return formatted_dict

    # Handle lists - recursively format all items
    elif isinstance(params, list):
        return [format_parameters(item) for item in params]

    # Handle tuples - recursively format and convert to tuple
    elif isinstance(params, tuple):
        return tuple(format_parameters(item) for item in params)

    # Handle bytes - decode to string
    elif isinstance(params, bytes):
        try:
            return params.decode("utf-8")
        except UnicodeDecodeError:
            # If it can't be decoded as UTF-8, return as is
            return params

    # Return other types unchanged
    return params


def initiate_finetuning(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to initiate fine-tuning.

    Args:
        params: Dictionary containing all parameters for fine-tuning

    Returns:
        Dictionary with training results and paths (only serializable data)
    """
    # Format parameters from Erlang/Elixir
    if params is not None:
        logger.info(f"Raw parameters before formatting: {params}")
        params = format_parameters(params)
        logger.info(f"Formatted parameters: {params}")

    # Default parameters
    default_params = {
        "mode": "unsupervised",
        "data_path": None,
        "text_column": "text",  # Default for unsupervised mode
        "use_checkpoint": False,
        "checkpoint_path": None,
        "max_samples": None,
        "pre_eval": True,
        "freeze_partly": False,
        "freeze_partly_layers": 0,
        "unfreeze_specific": False,
        "unfreeze_specific_layers": [],
        "eval_split": 0.1,  # Percentage of data to use for evaluation
        # Model parameters
        "model_path": None,
        "output_dir": "./ft_output",
        "logging_dir": None,
        "use_flash_attention": True,
        # Tokenizer parameters
        "max_length": 2048,
        "chunk_size": None,
        # Quantization parameters
        "quantization_config": 16,
        # PEFT parameters
        "peft_config": {
            "task_type": TaskType.CAUSAL_LM,
            "inference_mode": False,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        # Training parameters - include all possible parameters from setup_training_args
        "training_config": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "warmup_steps": 100,  # Default to steps-based warmup
            "warmup_ratio": None,  # Optional ratio-based warmup (will override steps if set)
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "eval_strategy": "steps",
            "eval_steps": 200,
            "per_device_eval_batch_size": 1,
            "eval_accumulation_steps": 4,
            "fp16": True,
            "bf16": False,  # Add bf16 support
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "gradient_checkpointing": True,
            "report_to": "none",
            "disable_tqdm": False,
            "max_grad_norm": 1.0,
            "dataloader_num_workers": 2,
            "optim": "adamw_torch",  # Default optimizer
        },
    }

    # Create a new merged parameters dictionary with properly formatted keys and values
    if params:
        # Direct update for top-level parameters
        for key, value in params.items():
            if (
                key in default_params
                and isinstance(default_params[key], dict)
                and isinstance(value, dict)
            ):
                # For nested dictionaries, update the nested dictionary
                default_params[key].update(value)
            else:
                # For other values, replace directly
                default_params[key] = value

    # If mode is "supervised", change the default text column
    if default_params["mode"] == "supervised":
        # Only override if user hasn't explicitly set it
        if "text_column" not in params:
            default_params["text_column"] = "input"
            logger.info("Setting text_column to 'input' for supervised mode")
    elif default_params["mode"] == "mixed":
        # For mixed mode, use both text column types if not explicitly set
        if "text_column" not in params:
            default_params["text_column"] = ["text", "input"]
            logger.info("Setting text_column to ['text', 'input'] for mixed mode")
        logger.info(
            f"Using mixed mode with text columns: {default_params['text_column']}"
        )

    if default_params["mode"] == "unsupervised":
        # Only override if user hasn't explicitly set it
        if "text_column" not in params:
            default_params["text_column"] = "text"
            logger.info("Setting text_column to 'text' for unsupervised mode")

    # Process quantization config
    quant_config = default_params["quantization_config"]
    if quant_config == 4 or quant_config == "4":
        default_params["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",  # Options: "nf4" or "fp4"
            "bnb_4bit_compute_dtype": torch.float16,
        }
        logger.info("Using 4-bit quantization with nf4 type")
    elif quant_config == 8 or quant_config == "8":
        default_params["quantization_config"] = {"load_in_8bit": True}
        logger.info("Using 8-bit quantization")
    elif quant_config == 16 or quant_config == "16":
        default_params["quantization_config"] = None
        logger.info("Using full precision (no quantization)")

        # For full precision, modify training config to avoid gradient issues
        if "training_config" in default_params:
            # Disable gradient checkpointing for full precision as it can cause issues
            default_params["training_config"]["gradient_checkpointing"] = False
            # Use a smaller learning rate for full model training
            if "learning_rate" not in params.get("training_config", {}):
                default_params["training_config"]["learning_rate"] = 1e-5
            logger.info("Disabled gradient checkpointing for full precision training")

    elif isinstance(quant_config, dict):
        # Keep the custom quantization config as is
        logger.info(f"Using custom quantization config: {quant_config}")
    else:
        logger.warning(
            f"Unknown quantization config '{quant_config}', defaulting to full precision"
        )
        default_params["quantization_config"] = None

    # Log the merged parameters for debugging
    logger.info(f"Final parameters after merging: {default_params}")

    # Required parameters validation
    if not default_params["data_path"]:
        raise ValueError("data_path must be specified")
    if not default_params["model_path"]:
        raise ValueError("model_path must be specified")

    # Setup environment
    setup_environment()

    # Check if data_path is a directory that contains JSONL files for training/validation/test
    train_file, val_file, test_file = None, None, None
    if os.path.isdir(default_params["data_path"]):
        train_file, val_file, test_file = check_dataset_files(
            default_params["data_path"]
        )

    # Set up file logger
    logging_dir = default_params.get("logging_dir") or default_params["output_dir"]
    setup_file_logger(
        logging_dir,
        use_checkpoint=default_params["use_checkpoint"],
        checkpoint_path=default_params["checkpoint_path"],
    )
    log_to_file(
        f"Starting {default_params['mode']} fine-tuning with parameters: {default_params}"
    )

    # Load datasets
    if train_file:  # If training file exists in the directory
        logger.info(f"Loading datasets from individual JSONL files")
        log_to_file(f"Loading datasets from individual JSONL files")
        train_dataset = load_finetuning_data(
            train_file,
            text_column=default_params["text_column"],
            max_samples=default_params["max_samples"],
        )

        # Load validation and test datasets if they exist
        eval_dataset = None
        if val_file:
            eval_dataset = load_finetuning_data(
                val_file,
                text_column=default_params["text_column"],
                max_samples=default_params["max_samples"],
            )
            logger.info(
                f"Loaded separate validation set with {len(eval_dataset)} examples"
            )
            log_to_file(
                f"Loaded separate validation set with {len(eval_dataset)} examples"
            )

        # Load test dataset for pre-evaluation and final evaluation
        test_dataset = None
        if test_file:
            test_dataset = load_finetuning_data(
                test_file,
                text_column=default_params["text_column"],
                max_samples=default_params["max_samples"],
            )
            logger.info(f"Loaded separate test set with {len(test_dataset)} examples")
            log_to_file(f"Loaded separate test set with {len(test_dataset)} examples")
    else:
        # Traditional approach - load dataset and split if needed
        logger.info(
            f"No specific dataset files found. Loading from {default_params['data_path']} and splitting"
        )
        log_to_file(
            f"No specific dataset files found. Loading from {default_params['data_path']} and splitting"
        )
        dataset = load_finetuning_data(
            default_params["data_path"],
            text_column=default_params["text_column"],
            max_samples=default_params["max_samples"],
        )

        # Split dataset if eval_split > 0
        if default_params["eval_split"] > 0:
            # Split into train, validation, and test sets (70%, 15%, 15% by default)
            splits = dataset.train_test_split(
                test_size=default_params["eval_split"] * 2
            )
            train_dataset = splits["train"]

            # Further split the test portion into validation and test sets
            eval_test_splits = splits["test"].train_test_split(test_size=0.5)
            eval_dataset = eval_test_splits["train"]  # validation set
            test_dataset = eval_test_splits["test"]  # test set

            logger.info(
                f"Split dataset into {len(train_dataset)} training, {len(eval_dataset)} validation, and {len(test_dataset)} test examples"
            )
            log_to_file(
                f"Split dataset into {len(train_dataset)} training, {len(eval_dataset)} validation, and {len(test_dataset)} test examples"
            )
        else:
            train_dataset = dataset
            eval_dataset = None
            test_dataset = None
            logger.info(
                f"Using all {len(train_dataset)} examples for training (no evaluation split)"
            )
            log_to_file(
                f"Using all {len(train_dataset)} examples for training (no evaluation split)"
            )

    # Check if we need to combine input and output columns for supervised mode
    if train_dataset is not None and default_params["mode"] == "supervised":
        if (
            "input" in train_dataset.column_names
            and "output" in train_dataset.column_names
        ):
            logger.info(
                "Supervised format detected, combining input and output columns"
            )
            log_to_file(
                "Supervised format detected, combining input and output columns"
            )
            train_dataset = prepare_combined_dataset(train_dataset)
            if (
                eval_dataset
                and "input" in eval_dataset.column_names
                and "output" in eval_dataset.column_names
            ):
                eval_dataset = prepare_combined_dataset(eval_dataset)
            if (
                test_dataset
                and "input" in test_dataset.column_names
                and "output" in test_dataset.column_names
            ):
                test_dataset = prepare_combined_dataset(test_dataset)
        else:
            logger.warning(
                "Supervised mode selected but 'input' and 'output' columns not found in dataset"
            )
            log_to_file(
                "Supervised mode selected but 'input' and 'output' columns not found in dataset"
            )

    # Load tokenizer
    tokenizer = load_tokenizer(default_params["model_path"])

    # Prepare datasets
    train_dataset = prepare_dataset(
        train_dataset,
        tokenizer,
        text_column=default_params["text_column"],
        max_length=default_params["max_length"],
        chunk_size=default_params["chunk_size"],
    )

    if eval_dataset:
        eval_dataset = prepare_dataset(
            eval_dataset,
            tokenizer,
            text_column=default_params["text_column"],
            max_length=default_params["max_length"],
            chunk_size=default_params["chunk_size"],
        )

    if test_dataset:
        test_dataset = prepare_dataset(
            test_dataset,
            tokenizer,
            text_column=default_params["text_column"],
            max_length=default_params["max_length"],
            chunk_size=default_params["chunk_size"],
        )

    # Clear memory before loading model
    clear_memory()

    # Load model
    model = load_model(
        default_params["model_path"],
        quantization_config=default_params["quantization_config"],
        use_flash_attention=default_params["use_flash_attention"],
        freeze_partly=default_params["freeze_partly"],
        freeze_partly_layers=default_params["freeze_partly_layers"],
        unfreeze_specific=default_params["unfreeze_specific"],
        unfreeze_specific_layers=default_params["unfreeze_specific_layers"],
    )

    # Apply PEFT
    model = apply_peft_config(model, peft_config=default_params["peft_config"])

    # Verify model is in training mode with trainable parameters after PEFT
    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        logger.warning(
            "No trainable parameters found after applying PEFT! Check your configuration."
        )
    else:
        logger.info(
            f"Model has {trainable_params} trainable parameters after PEFT configuration"
        )

    # Setup training arguments
    training_args = setup_training_args(
        default_params["output_dir"], training_config=default_params["training_config"]
    )

    # Train model
    log_to_file(f"Starting model training with {len(train_dataset)} training examples")
    if eval_dataset:
        log_to_file(
            f"Using {len(eval_dataset)} examples for validation during training"
        )
    if test_dataset:
        log_to_file(f"Using {len(test_dataset)} examples for pre/final evaluation")

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        training_args=training_args,
        pre_eval=default_params["pre_eval"],
        output_dir=default_params["output_dir"],
        use_checkpoint=default_params["use_checkpoint"],
        checkpoint_path=default_params["checkpoint_path"],
    )

    log_to_file("Training complete!")

    # Return only serializable data (paths and statistics) instead of model objects
    return {
        "status": "success",
        "output_dir": default_params["output_dir"],
        "final_model_path": os.path.join(default_params["output_dir"], "final_model"),
        "train_dataset_size": len(train_dataset) if train_dataset else 0,
        "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
        "test_dataset_size": len(test_dataset) if test_dataset else 0,
        "model_base_path": default_params["model_path"],
        "training_mode": default_params["mode"],
    }
