import subprocess
import os
import sys


def quantize_model(input_path, output_path, quant_type):
    """
    Quantize a GGUF model using llama.cpp's quantization tool.

    Args:
        input_path: Path to input GGUF model
        output_path: Path for quantized output model
        quant_type: Quantization type (e.g., 'Q4_0', 'Q8_0')
    """
    # Convert Erlang charlists (list of ints) to strings if needed
    if isinstance(input_path, list) and all(isinstance(x, int) for x in input_path):
        input_path = "".join(chr(x) for x in input_path)
    if isinstance(output_path, list) and all(isinstance(x, int) for x in output_path):
        output_path = "".join(chr(x) for x in output_path)
    if isinstance(quant_type, list) and all(isinstance(x, int) for x in quant_type):
        quant_type = "".join(chr(x) for x in quant_type)

    # Updated quantization type string to numeric code mapping
    quant_map = {
        "q4_0": "2",
        "q4_1": "3",
        "q5_0": "8",
        "q5_1": "9",
        "iq2_xxs": "19",
        "iq2_xs": "20",
        "iq2_s": "28",
        "iq2_m": "29",
        "iq1_s": "24",
        "iq1_m": "31",
        "tq1_0": "36",
        "tq2_0": "37",
        "q2_k": "10",
        "q2_k_s": "21",
        "iq3_xxs": "23",
        "iq3_s": "26",
        "iq3_m": "27",
        "q3_k": "12",  # alias for Q3_K_M
        "iq3_xs": "22",
        "q3_k_s": "11",
        "q3_k_m": "12",
        "q3_k_l": "13",
        "iq4_nl": "25",
        "iq4_xs": "30",
        "q4_k": "15",  # alias for Q4_K_M
        "q4_k_s": "14",
        "q4_k_m": "15",
        "q5_k": "17",  # alias for Q5_K_M
        "q5_k_s": "16",
        "q5_k_m": "17",
        "q6_k": "18",
        "q8_0": "7",
        "f16": "1",
        "bf16": "32",
        "f32": "0",
        "copy": "COPY",
    }
    qt = str(quant_type).lower()
    qt = qt.replace("-", "_")
    ftype_code = quant_map.get(qt)
    if ftype_code is None:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Quantization type '{quant_type}' is not supported or not mapped to a file type code.",
        }

    try:
        quantize_path = os.path.join(
            os.path.dirname(__file__),
            "llama_cuda",
            "llama-quantize.exe",
        )

        print(
            f"Quantizing model from {input_path} to {output_path} with type {quant_type} (ftype {ftype_code})"
        )

        if not os.path.exists(quantize_path):
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Quantize executable not found at path: {quantize_path}",
            }

        cmd = [
            quantize_path,
            str(input_path),
            str(output_path),
            ftype_code,
        ]

        process = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except subprocess.CalledProcessError as e:
        return {"returncode": e.returncode, "stdout": e.stdout, "stderr": e.stderr}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": str(e)}


def quantize_safetensors_model(input_path, output_path, quant_type):
    """
    Quantize a safetensors model by converting it to GGUF with quantization.
    Calls convert_model from convert_wrapper.py.
    """
    try:
        # Try to import convert_model from convert_wrapper.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from convert_wrapper import convert_model
    except ImportError as e:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Could not import convert_model from convert_wrapper.py: {e}",
        }

    # Call convert_model with quantization type
    return convert_model(input_path, output_path, quant_type)


def quantize_safetensors_inplace(input_path, output_path, quant_type):
    """
    Quantize a safetensors model and save as safetensors.
    Only standard PyTorch quantization (e.g., int8) is supported.
    Returns error for unsupported quantization types (e.g., Q4_0, Q5_0, etc.).
    """
    # Map llama.cpp quantization names to torch dtypes
    quant_map = {
        "q8_0": "int8",
        "q8": "int8",
        "int8": "int8",
        "qint8": "int8",
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "copy": None,
    }
    qt = str(quant_type).lower().replace("-", "_")
    # Accept both uppercase and lowercase quant types
    qt = qt.strip()
    if qt not in quant_map:
        # Try uppercase mapping (e.g., Q8_0)
        qt_uc = qt.upper()
        qt_lc = qt.lower()
        # Accept Q8_0, F16, etc.
        if qt_uc in [k.upper() for k in quant_map]:
            # Map to lower-case key
            qt = [k for k in quant_map if k.upper() == qt_uc][0]
        elif qt_lc in quant_map:
            qt = qt_lc
        else:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Quantization type '{quant_type}' is not supported for safetensors in-place quantization. Only int8/f16/bf16/f32/copy are supported.",
            }
    torch_quant = quant_map.get(qt)
    if torch_quant is None and qt in quant_map:
        # "copy" or similar, just copy tensors
        torch_quant = None
    elif torch_quant is None:
        # Not supported (e.g., Q4_0, Q5_0, etc.)
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Quantization type '{quant_type}' is not supported for safetensors in-place quantization. Only int8/f16/bf16/f32/copy are supported.",
        }

    try:
        import torch
        from safetensors.torch import load_file, save_file

        tensors = load_file(input_path)
        quantized_tensors = {}

        for name, tensor in tensors.items():
            if torch_quant == "int8":
                quantized = torch.quantize_per_tensor(
                    tensor, scale=0.1, zero_point=0, dtype=torch.qint8
                )
                quantized_tensors[name] = quantized.dequantize()
            elif torch_quant == "float16":
                quantized_tensors[name] = tensor.to(torch.float16)
            elif torch_quant == "bfloat16":
                quantized_tensors[name] = tensor.to(torch.bfloat16)
            elif torch_quant == "float32":
                quantized_tensors[name] = tensor.to(torch.float32)
            else:
                # "copy" or unknown, just copy
                quantized_tensors[name] = tensor

        save_file(quantized_tensors, output_path)
        return {
            "returncode": 0,
            "stdout": f"Quantized and saved to {output_path}",
            "stderr": "",
        }
    except Exception as e:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": str(e),
        }
