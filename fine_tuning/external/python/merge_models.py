import os
import time
import torch
import gc
import argparse
import shutil
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import mergekit
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge


def test_connection():
    """
    Simple function to test if the connection to Python is working.
    """
    try:
        return "ok"
    except Exception as e:
        return f"error: {str(e)}"


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
                try:
                    formatted_key = formatted_key.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    formatted_key = str(formatted_key)

            # Remove null bytes from string keys
            if isinstance(formatted_key, str):
                formatted_key = formatted_key.replace("\x00", "")

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
            result = params.decode("utf-8", errors="replace")
            # Remove null bytes
            return result.replace("\x00", "")
        except UnicodeDecodeError:
            # If it can't be decoded as UTF-8, return as is
            return params

    # Handle strings - remove null bytes
    elif isinstance(params, str):
        return params.replace("\x00", "")

    # Return other types unchanged
    return params


def merge_adapters_into_base_model(params=None):
    """
    Entry point for merging adapter models into a base model from Elixir.

    Args:
        params: Dictionary containing all parameters
            - base_model_path: Path to the base model
            - adapter_path: Path to the adapter model
            - output_path: Directory to save the merged model
            - use_fp16: Whether to use FP16 precision
            - verbose: Whether to print verbose output
    """
    # Format parameters from Elixir
    params = format_parameters(params) or {}

    # Extract required parameters with defaults
    base_model_path = params.get("base_model_path")
    adapter_path = params.get("adapter_path")
    output_path = params.get("output_path")

    # Set up configuration
    config = {
        "device_map": "auto",
        "offload_folder": (
            os.path.join(os.path.dirname(output_path), "temp_offload")
            if output_path
            else "./temp_offload"
        ),
        "offload_buffers": True,
        "use_fp16": params.get("use_fp16", True),  # Default to True if not specified
        "use_8bit": False,
        "verbose": params.get("verbose", True),  # Default to True if not specified
    }

    if config["verbose"]:
        print(f"Starting merge with parameters:")
        print(f"  Base model: {base_model_path}")
        print(f"  Adapter: {adapter_path}")
        print(f"  Output: {output_path}")
        print(f"  Using FP16: {config['use_fp16']}")

    # Perform the merge operation
    _merge_adapters_into_base_model(base_model_path, adapter_path, output_path, config)

    return "Merge completed successfully"


def _merge_adapters_into_base_model(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Merge adapter models into a base model and save the result.

    Args:
        base_model_path: Path to the base model
        adapter_path: Path to the adapter model
        output_dir: Directory to save the merged model
        config: Optional configuration parameters
    """
    # Default configuration
    default_config = {
        "device_map": "auto",
        "offload_folder": os.path.join(os.path.dirname(output_dir), "temp_offload"),
        "offload_buffers": True,
        "use_fp16": True,
        "use_8bit": False,
        "verbose": True,
    }

    # Override defaults with provided config
    if config:
        default_config.update(config)

    config = default_config

    # Create output and offload directories if they don't exist
    os.makedirs(config["offload_folder"], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Force garbage collection before loading models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if config["verbose"]:
        print(f"Loading base model from {base_model_path}...")

    # Load base model with specified configuration
    model_kwargs = {
        "device_map": config["device_map"],
        "offload_folder": config["offload_folder"],
        "offload_buffers": config["offload_buffers"],
    }

    if config["use_fp16"]:
        model_kwargs["torch_dtype"] = torch.float16

    if config["use_8bit"]:
        model_kwargs["load_in_8bit"] = True

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    if config["verbose"]:
        print(f"Loading adapter from {adapter_path}...")

    # Load adapter with same offload settings
    adapter_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        offload_folder=config["offload_folder"],
        offload_buffers=config["offload_buffers"],
    )

    # Merge adapter into base model
    if config["verbose"]:
        print("Starting merge operation...")
    merged_model = adapter_model.merge_and_unload()

    # Free memory before saving
    del base_model
    del adapter_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if config["verbose"]:
        print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)

    # Copy important files from adapter_path to output_dir
    important_files = [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
    ]

    if config["verbose"]:
        print(f"Copying important tokenizer and configuration files...")

    for filename in important_files:
        source_file = os.path.join(adapter_path, filename)
        target_file = os.path.join(output_dir, filename)

        # Check if the file exists in the adapter path
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, target_file)
                if config["verbose"]:
                    print(f"  Copied {filename}")
            except Exception as e:
                print(f"  Warning: Failed to copy {filename}: {str(e)}")
        else:
            # If file doesn't exist in adapter_path, check the parent directory
            # (sometimes files are stored in the parent directory of the adapter)
            parent_source = os.path.join(os.path.dirname(adapter_path), filename)
            if os.path.exists(parent_source):
                try:
                    shutil.copy2(parent_source, target_file)
                    if config["verbose"]:
                        print(f"  Copied {filename} from parent directory")
                except Exception as e:
                    print(
                        f"  Warning: Failed to copy {filename} from parent directory: {str(e)}"
                    )
            else:
                print(
                    f"  Warning: {filename} not found in adapter path or its parent directory"
                )

    if config["verbose"]:
        print("Model successfully merged and saved!")


def _merge_models(config, output_dir=None, cpu_offload=None, dtype=None):
    """
    Wrapper for mergekit that adds ETA functionality

    Args:
        config: MergeKit configuration dictionary or path to YAML
        output_dir: Output directory for merged model (overrides config's output_dir if provided)
        cpu_offload: Whether to use CPU offloading (overrides config's cpu_offload if provided)
        dtype: Data type for merging (overrides config's dtype if provided)
    """
    # Handle case where config contains output_dir, cpu_offload and dtype
    if isinstance(config, dict):
        # Extract parameters from config if they exist there
        output_dir = output_dir or config.pop("output_dir", None)
        cpu_offload = (
            cpu_offload if cpu_offload is not None else config.pop("cpu_offload", True)
        )
        dtype = dtype or config.pop("dtype", "float16")

        # Create proper mergekit config format if coming from our JSON format
        if "models" in config and "merge_method" in config:
            merged_config = {
                "slices": [
                    {
                        "sources": [
                            {"model": model_info["model"], "layer_range": [0, 32]}
                            for model_info in config["models"]
                        ]
                    }
                ],
                "merge_method": config["merge_method"],
                "parameters": config.get("merge_parameters", {}),
                "dtype": dtype,
            }

            # Set base_model if available
            if len(config["models"]) > 0:
                merged_config["base_model"] = config["models"][0]["model"]

            # Convert standard SLERP parameters if needed
            if config["merge_method"] == "slerp":
                # Format t parameter correctly
                if "t" in merged_config["parameters"]:
                    merged_config["parameters"]["t"] = float(
                        merged_config["parameters"]["t"]
                    )
                # Handle layer-specific weights
                elif "weights_by_layer" in merged_config["parameters"]:
                    # Convert to proper format for YAML-style config
                    layer_weights = merged_config["parameters"]["weights_by_layer"]
                    t_filters = []

                    # Group layers by weight value
                    weight_groups = {}
                    for layer, weight in layer_weights.items():
                        layer = int(layer)
                        weight = float(weight)
                        if weight not in weight_groups:
                            weight_groups[weight] = []
                        weight_groups[weight].append(layer)

                    # Create filter entries for each weight group
                    for weight, layers in weight_groups.items():
                        # Create regex pattern for these layers
                        layers_str = "|".join([str(l) for l in layers])
                        t_filters.append(
                            {"filter": f"self_attn.*({layers_str})", "value": weight}
                        )
                        t_filters.append(
                            {"filter": f"mlp.*({layers_str})", "value": weight}
                        )

                    # Add default value
                    t_filters.append({"value": 0.5})

                    # Replace weights_by_layer with t filters
                    merged_config["parameters"] = {"t": t_filters}

            config = merged_config

    if not output_dir:
        raise ValueError(
            "output_dir must be provided either in config or as a parameter"
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Print the config structure
    print(f"Final merge configuration structure:")
    import json

    print(json.dumps(config, indent=2))

    # Convert to MergeConfiguration object
    merge_config = MergeConfiguration.model_validate(config)

    # Set up options
    options = MergeOptions(
        cuda=torch.cuda.is_available(), copy_tokenizer=True, low_cpu_memory=cpu_offload
    )

    # Start timer
    start_time = time.time()

    try:
        # Run the merge
        run_merge(merge_config, out_path=output_dir, options=options)
    except Exception as e:
        print(f"Error during model merging: {e}")
        raise
    finally:
        print(
            f"Merge operation completed in {(time.time() - start_time)/60:.1f} minutes"
        )

    return "Merge completed successfully"


def merge_models(params=None):
    """
    Entry point for merging models with mergekit from Elixir/JSON configuration.

    Args:
        params: Parameters formatted as a dictionary matching the JSON structure
    """
    try:
        # Format parameters from Elixir/JSON
        params = format_parameters(params) or {}

        # Check if we're using YAML config
        if "yaml_config_path" in params:
            yaml_path = params["yaml_config_path"]
            output_dir = params.get("output_dir")
            cpu_offload = params.get("cpu_offload", True)

            print(f"Using YAML configuration from: {yaml_path}")

            # Validate required parameters
            if not output_dir:
                raise ValueError("Missing 'output_dir' parameter in configuration")

            # Format paths properly (handle strlists from Elixir)
            yaml_path = str(yaml_path).replace("\x00", "")
            output_dir = str(output_dir).replace("\x00", "")

            # Load YAML configuration
            import yaml

            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)
            except Exception as e:
                raise ValueError(f"Failed to load YAML configuration: {e}")

            # Convert to MergeKit configuration
            merge_config = MergeConfiguration.model_validate(yaml_config)

            # Set up merge options
            options = MergeOptions(
                cuda=torch.cuda.is_available(),
                copy_tokenizer=True,
                low_cpu_memory=cpu_offload,
            )

            # Start timing and show progress
            start_time = time.time()
            print(f"Starting merge with YAML configuration...")
            print(f"  Output directory: {output_dir}")
            print(f"  CPU offload: {cpu_offload}")
            print(
                f"  Models to merge: {len(yaml_config.get('slices', [{}])[0].get('sources', []))}"
            )

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            try:
                # Run merge with MergeKit
                run_merge(merge_config, out_path=output_dir, options=options)
                print(f"Merge completed in {(time.time() - start_time)/60:.1f} minutes")
                return "Merge completed successfully"
            except Exception as e:
                print(f"Error during model merging: {e}")
                raise

        # Original code for JSON-based configuration follows:

        # Extract and validate required parameters
        if not params.get("models"):
            raise ValueError("Missing 'models' parameter in configuration")

        output_dir = params.get("output_dir")
        if not output_dir:
            raise ValueError("Missing 'output_dir' parameter in configuration")

        # Check for advanced SLERP configuration
        if params.get("merge_method") == "slerp":
            if params.get("advanced_slerp", False):
                print("Using Advanced SLERP with layer-dependent weights...")

                # Extract layer groupings and their weights
                layer_weights = params.get(
                    "layer_weights",
                    {
                        "lower": {"weight": 0.4, "layers": [0, 10]},
                        "middle": {"weight": 0.5, "layers": [11, 21]},
                        "upper": {"weight": 0.7, "layers": [22, 32]},
                    },
                )

                # Configure layer-specific weights
                if "merge_parameters" not in params:
                    params["merge_parameters"] = {}

                # Set up layer-specific weights in the format mergekit expects
                params["merge_parameters"]["weight_by_layer"] = True

                # Create the weight map with proper type conversions
                weights_by_layer = {}
                for group_name, group_config in layer_weights.items():
                    try:
                        # Ensure weight is a float
                        weight = float(group_config["weight"])

                        # Get layer boundaries, handle potential type issues
                        layers = group_config["layers"]
                        if isinstance(layers, list) and len(layers) >= 2:
                            # Clean and convert layer indices
                            start_layer = int(str(layers[0]).replace("\x00", ""))
                            end_layer = int(str(layers[1]).replace("\x00", ""))

                            # Assign weights to each layer in this group
                            for layer in range(start_layer, end_layer + 1):
                                # Convert layer number to string for dictionary key
                                layer_key = str(layer)
                                weights_by_layer[layer_key] = weight
                    except (ValueError, TypeError, IndexError) as e:
                        print(
                            f"Warning: Error processing layer weight group '{group_name}': {e}"
                        )
                        continue

                # Add weights to configuration
                params["merge_parameters"]["weights_by_layer"] = weights_by_layer
                print(f"Configured layer-specific weights: {weights_by_layer}")
            else:
                # Regular SLERP case - make sure 't' parameter is set
                print("Using regular SLERP with global weight...")
                if "merge_parameters" not in params:
                    params["merge_parameters"] = {}

                # Use default t=0.5 if not specified
                if "t" not in params["merge_parameters"]:
                    params["merge_parameters"]["t"] = float(0.5)
                else:
                    # Ensure 't' is a float, with error handling
                    try:
                        t_value = params["merge_parameters"]["t"]
                        if isinstance(t_value, str):
                            t_value = t_value.replace("\x00", "")
                        params["merge_parameters"]["t"] = float(t_value)
                    except (ValueError, TypeError):
                        print("Warning: Invalid 't' value, using default 0.5")
                        params["merge_parameters"]["t"] = 0.5

                print(f"Using SLERP weight t={params['merge_parameters']['t']}")

        # Ensure all model paths are strings
        if "models" in params:
            for i, model_config in enumerate(params["models"]):
                if "model" in model_config:
                    model_path = str(model_config["model"]).replace("\x00", "")
                    params["models"][i]["model"] = model_path

        # Call the merge function with properly formatted parameters
        return _merge_models(
            config=params,
            output_dir=str(output_dir).replace("\x00", ""),
            cpu_offload=bool(params.get("cpu_offload", True)),
            dtype=str(params.get("dtype", "float16")).replace("\x00", ""),
        )
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Error in merge_models: {e}\n{error_details}")
        raise
