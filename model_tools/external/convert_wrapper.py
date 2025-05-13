import subprocess
import os
import sys
import glob


def convert_model(input_path, output_path, quantization="f16"):
    """
    Executes the conversion script with the provided input and output paths.

    Args:
        input_path: Path to the input model (safetensors)
        output_path: Path where the output GGUF model will be saved
        quantization: Quantization type (default: "fp16")

    Returns:
        dict: Process results with returncode, stdout, stderr
    """
    # Convert Erlang character lists (list of integers) to Python strings if needed
    if isinstance(input_path, list) and all(isinstance(x, int) for x in input_path):
        input_path = "".join(chr(x) for x in input_path)

    if isinstance(output_path, list) and all(isinstance(x, int) for x in output_path):
        output_path = "".join(chr(x) for x in output_path)

    if isinstance(quantization, list) and all(isinstance(x, int) for x in quantization):
        quantization = "".join(chr(x) for x in quantization)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to find the conversion script in several possible locations
    script_path = None
    possible_locations = [
        os.path.join(current_dir, "llama_cpp", "convert_hf_to_gguf.py"),
        os.path.join(current_dir, "convert_hf_to_gguf.py"),
        # Look for any file with similar name
        *glob.glob(
            os.path.join(current_dir, "**", "*convert*gguf*.py"), recursive=True
        ),
    ]

    for location in possible_locations:
        if os.path.isfile(location):
            script_path = location
            break

    if not script_path:
        # Script not found, return error
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Could not find conversion script. Searched in: {possible_locations}",
        }

    # Debug info about the environment
    debug_info = f"Using script at: {script_path}\nCurrent directory: {current_dir}\n"
    debug_info += f"Python executable: {sys.executable}\n"
    debug_info += f"Converting model with quantization: {quantization}\n"

    cmd = [
        sys.executable,
        script_path,
        input_path,
        "--outfile",
        output_path,
        "--outtype",
        quantization,
    ]

    print(cmd)

    try:
        # Force UTF-8 encoding for both stdin and stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid characters instead of failing
        )
        stdout, stderr = process.communicate()

        print("=== STDOUT ===")
        print(stdout)
        print("=== STDERR ===")
        print(stderr)

        # Debug info in stdout
        stdout = debug_info + stdout

        # Use simple strings for better Erlang compatibility
        return {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr if stderr else "",
        }
    except Exception as e:
        return {"returncode": -1, "stdout": debug_info, "stderr": str(e)}


def convert_lora(
    base_model_path, lora_path, output_path, quantization="q8_0", base_model_id=None
):
    """
    Executes the convert_lora_to_gguf script to merge LoRA adapters with a base model and convert to GGUF.

    Args:
        lora_path: Path to the LoRA adapter model
        output_path: Path where the output merged GGUF model will be saved
        base_model_path: Path to the base model (optional)
        base_model_id: HuggingFace model ID for the base model (optional)
        quantization: Quantization type (default: "q8_0")

    Returns:
        dict: Process results with returncode, stdout, stderr
    """
    # Convert Erlang character lists (list of integers) to Python strings if needed
    # Convert Erlang character lists (list of integers) to Python strings if needed
    if isinstance(quantization, list) and all(isinstance(x, int) for x in quantization):
        quantization = "".join(chr(x) for x in quantization)

    # Convert other parameters as needed
    if isinstance(lora_path, list) and all(isinstance(x, int) for x in lora_path):
        lora_path = "".join(chr(x) for x in lora_path)

    if isinstance(output_path, list) and all(isinstance(x, int) for x in output_path):
        output_path = "".join(chr(x) for x in output_path)

    if (
        base_model_path
        and isinstance(base_model_path, list)
        and all(isinstance(x, int) for x in base_model_path)
    ):
        base_model_path = "".join(chr(x) for x in base_model_path)

    # Handle byte strings for base_model_id
    if isinstance(base_model_id, bytes):
        base_model_id = base_model_id.decode("utf-8")
    elif (
        base_model_id
        and isinstance(base_model_id, list)
        and all(isinstance(x, int) for x in base_model_id)
    ):
        base_model_id = "".join(chr(x) for x in base_model_id)

    # Handle "nil" strings after all conversions
    if base_model_id == "nil":
        base_model_id = None

    if base_model_path == "nil":
        base_model_path = None

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to find the conversion script in several possible locations
    script_path = None
    possible_locations = [
        os.path.join(current_dir, "llama_cpp", "convert_lora_to_gguf.py"),
        os.path.join(current_dir, "convert_lora_to_gguf.py"),
        # Look for any file with similar name
        *glob.glob(
            os.path.join(current_dir, "**", "*convert*lora*gguf*.py"), recursive=True
        ),
    ]

    for location in possible_locations:
        if os.path.isfile(location):
            script_path = location
            break

    if not script_path:
        # Script not found, return error
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Could not find LoRA conversion script. Searched in: {possible_locations}",
        }

    # Debug info about the environment
    debug_info = f"Using script at: {script_path}\nCurrent directory: {current_dir}\n"
    debug_info += f"Python executable: {sys.executable}\n"
    debug_info += f"Converting LoRA: {lora_path} to {output_path} with quantization {quantization}\n"
    debug_info += f"Base model path: {base_model_path}\n"
    debug_info += f"Base model ID: {base_model_id}\n"

    print(debug_info)

    # Build the command with all necessary arguments
    cmd = [
        sys.executable,
        script_path,
        lora_path,
        "--outfile",
        output_path,
        "--outtype",
        quantization,
    ]

    # Add optional base model parameters if provided
    if base_model_path:
        cmd.extend(["--base", base_model_path])

    if base_model_id:
        cmd.extend(["--base-model-id", base_model_id])

    try:
        # Force UTF-8 encoding for both stdin and stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid characters instead of failing
        )
        stdout, stderr = process.communicate()

        # Debug info in stdout
        stdout = debug_info + stdout

        # Use simple strings for better Erlang compatibility
        return {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr if stderr else "",
        }
    except Exception as e:
        return {"returncode": -1, "stdout": debug_info, "stderr": str(e)}
