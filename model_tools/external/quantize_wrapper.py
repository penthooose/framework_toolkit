import subprocess
import os


def quantize_model(input_path, output_path, quant_type):
    """
    Quantize a GGUF model using llama.cpp's quantization tool.

    Args:
        input_path: Path to input GGUF model
        output_path: Path for quantized output model
        quant_type: Quantization type (e.g., 'Q4_0', 'Q8_0')
    """
    try:
        # Path to llama.cpp quantize executable
        quantize_path = os.path.join(
            os.path.dirname(__file__),
            "llama.cpp",
            "examples",
            "quantize",
            "quantize.cpp",
        )

        print(
            f"Quantizing model from {input_path} to {output_path} with type {quant_type}"
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
            f"--type={str(quant_type)}",
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
    return
