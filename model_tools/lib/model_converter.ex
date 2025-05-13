defmodule ModelConverter do
  @moduledoc """
  API for certain operations on (LLM) models, e.g. converting models from one format to another.
  Uses the llama.cpp library for conversion via Erlport.
  """

  @load_params_from_file true
  @use_param_entry "2"
  @conv_params_file_path Path.expand(Path.join([__DIR__, "..", "external", "conv_params.json"]))
  @path_output_gguf "N:/LLMs/gguf_models/llama3_german_instruct_FT_stage1.gguf"

  # specify path of safetensors model and path of output gguf model
  @path_safetensors_model "N:/LLMs/huggingface_models/models--DiscoResearch--Llama3-DiscoLeo-Instruct-8B-v0.1/snapshots/0d7cf14a066a65f037d184d00eddc9d284225d93"
  @path_safetensors_lora "N:/LLMs/finetuned_models/checkpoints_llama3_german_instruct_stage1/final_model"

  # original command for converting:
  # python convert-hf-to-gguf.py @path_safetensors_model --outfile @path_output_gguf

  # parameters and values explained: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
  @modelfile_contents """
    PARAMETER temperature 0.2
    PARAMETER top_p 0.5
    PARAMETER top_k 50
    PARAMETER num_predict: 3072
    PARAMETER num_ctx 8192

    PARAMETER stop "\n\n"

    SYSTEM Du bist ein Assistent zur Erstellung von Gutachten über Medizinprodukte.

  """

  # path of the wrapper for the running the Python converter script (convert-hf-to-gguf.py)
  @path_gguf_converter Path.join([
                         File.cwd!(),
                         "external"
                       ])

  def convert_safetensors_to_gguf(file_param_entry \\ nil)
      when is_binary(file_param_entry) or is_nil(file_param_entry) or is_integer(file_param_entry) do
    param_entry =
      if is_integer(file_param_entry),
        do: Integer.to_string(file_param_entry),
        else: file_param_entry

    params = get_conversion_params(param_entry)

    convert_safetensors_to_gguf_impl(
      params["path_safetensors_model"],
      params["path_output_gguf"],
      params["quantization"],
      params["integrate_in_ollama"],
      params["modelfile_contents"]
    )
  end

  def convert_safetensors_to_gguf_impl(
        path_safetensors_model,
        path_output_gguf,
        quantization \\ "f16",
        integrate_in_ollama \\ true,
        modelfile_contents \\ nil
      ) do
    {:ok, pid} =
      :python.start_link([
        {:python, ~c"python"},
        {:python_path, String.to_charlist(@path_gguf_converter)}
      ])

    try do
      result =
        :python.call(pid, :convert_wrapper, :convert_model, [
          String.to_charlist(path_safetensors_model),
          String.to_charlist(path_output_gguf),
          String.to_charlist(quantization)
        ])

      result = convert_charlists_in_map(result)

      conversion_result =
        cond do
          is_map_key(result, "returncode") && result["returncode"] == 0 ->
            IO.puts("Conversion successful.")
            {:ok, "Model converted successfully."}

          is_map_key(result, "returncode") ->
            {:error, "Exit code: #{result["returncode"]}, Error: #{result["stderr"] || ""}"}

          true ->
            {:error, "Unexpected result structure: #{inspect(result)}"}
        end

      final_result =
        if integrate_in_ollama do
          try do
            case conversion_result do
              {:ok, _} ->
                integrate_gguf_in_ollama_impl(path_output_gguf, true, modelfile_contents)

              error ->
                error
            end
          catch
            kind, error ->
              IO.puts(
                "Error while integrating GGUF file in Ollama: #{inspect(kind)} - #{inspect(error)}"
              )

              {:error, error}
          end
        else
          conversion_result
        end

      final_result
    catch
      kind, error ->
        IO.puts("Error: #{inspect(kind)} - #{inspect(error)}")
        :python.stop(pid)
        {:error, error}
    after
      :python.stop(pid)
    end
  end

  def integrate_gguf_in_ollama(file_param_entry \\ nil)
      when is_binary(file_param_entry) or is_nil(file_param_entry) or is_integer(file_param_entry) do
    param_entry =
      if is_integer(file_param_entry),
        do: Integer.to_string(file_param_entry),
        else: file_param_entry

    params = get_conversion_params(param_entry)

    integrate_gguf_in_ollama_impl(
      params["path_output_gguf"],
      true,
      params["modelfile_contents"]
    )
  end

  def integrate_gguf_in_ollama_impl(
        path_output_gguf,
        overwrite \\ true,
        modelfile_contents \\ nil
      ) do
    raw_name = Path.basename(path_output_gguf, ".gguf")
    model_name = sanitize_model_name(raw_name)

    modelfile_contents = modelfile_contents || @modelfile_contents

    model_exists =
      case System.cmd("ollama", ["list"], stderr_to_stdout: true) do
        {output, 0} ->
          String.contains?(output, model_name)

        _ ->
          false
      end

    if model_exists and overwrite do
      case System.cmd("ollama", ["rm", model_name], stderr_to_stdout: true) do
        {_, 0} ->
          IO.puts("Removed existing model #{model_name} for replacement.")

        {error, code} ->
          IO.puts("Warning: Failed to remove existing model. Exit code: #{code}")
          IO.puts("Error: #{error}")
      end
    end

    temp_dir = Path.join(System.tmp_dir(), "ollama_integration_#{:os.system_time(:millisecond)}")
    File.mkdir_p!(temp_dir)
    modelfile_path = Path.join(temp_dir, "Modelfile")

    complete_modelfile_content = """
    FROM #{path_output_gguf}
    #{modelfile_contents}
    """

    File.write!(modelfile_path, complete_modelfile_content)

    case System.cmd("ollama", ["create", model_name, "-f", modelfile_path],
           stderr_to_stdout: true
         ) do
      {output, 0} ->
        IO.puts("Successfully integrated model into Ollama:")
        IO.puts(output)
        File.rm_rf!(temp_dir)
        {:ok, "Model #{model_name} successfully integrated into Ollama"}

      {error, code} ->
        IO.puts("Failed to integrate model into Ollama. Exit code: #{code}")
        IO.puts("Error: #{error}")
        File.rm_rf!(temp_dir)
        {:error, "Failed to integrate model with error: #{error}"}
    end
  end

  def convert_lora_adapter_to_gguf(file_param_entry \\ nil)
      when is_binary(file_param_entry) or is_nil(file_param_entry) or is_integer(file_param_entry) do
    param_entry =
      if is_integer(file_param_entry),
        do: Integer.to_string(file_param_entry),
        else: file_param_entry

    params = get_conversion_params(param_entry)

    convert_lora_adapter_to_gguf_impl(
      params["path_safetensors_model"],
      params["path_safetensors_lora"],
      params["path_output_gguf"],
      params["quantization"],
      params["base_model_id"],
      params["integrate_in_ollama"]
    )
  end

  def convert_lora_adapter_to_gguf_impl(
        base_model_path,
        path_lora_model,
        path_output_gguf,
        quantization \\ "f16",
        base_model_id \\ nil,
        integrate_in_ollama \\ false
      ) do
    {:ok, pid} =
      :python.start_link([
        {:python, ~c"python"},
        {:python_path, String.to_charlist(@path_gguf_converter)}
      ])

    try do
      result =
        :python.call(pid, :convert_wrapper, :convert_lora, [
          if(base_model_path, do: String.to_charlist(base_model_path), else: nil),
          String.to_charlist(path_lora_model),
          String.to_charlist(path_output_gguf),
          String.to_charlist(quantization),
          if(base_model_id, do: String.to_charlist(base_model_id), else: nil)
        ])
        |> convert_charlists_in_map()

      conversion_result =
        case result do
          %{"returncode" => 0, "stdout" => stdout} ->
            {:ok, stdout}

          %{"returncode" => code, "stderr" => stderr} ->
            {:error, "Exit code: #{code}, Error: #{stderr}"}
        end

      final_result =
        if integrate_in_ollama do
          try do
            case conversion_result do
              {:ok, _} -> integrate_gguf_in_ollama_impl(path_output_gguf)
              error -> error
            end
          catch
            kind, error ->
              IO.puts(
                "Error while integrating GGUF file in Ollama: #{inspect(kind)} - #{inspect(error)}"
              )

              {:error, error}
          end
        else
          conversion_result
        end

      final_result
    after
      :python.stop(pid)
    end
  end

  def get_conversion_params(file_param_entry \\ nil) do
    param_entry = file_param_entry || @use_param_entry

    if @load_params_from_file do
      load_params_from_json(param_entry)
    else
      %{
        "path_safetensors_model" => @path_safetensors_model,
        "path_safetensors_lora" => @path_safetensors_lora,
        "path_output_gguf" => @path_output_gguf,
        "modelfile_contents" => @modelfile_contents
      }
    end
  end

  def load_params_from_json(param_entry \\ nil) do
    param_entry = param_entry || @use_param_entry

    try do
      with {:ok, json_content} <- File.read(@conv_params_file_path),
           {:ok, json_parsed} <- Jason.decode(json_content),
           params_entry = json_parsed[param_entry] || json_parsed["1"] || %{} do
        modelfile_contents =
          case params_entry["modelfile_contents"] do
            nil -> @modelfile_contents
            contents when is_map(contents) -> convert_modelfile_contents(contents)
            contents when is_binary(contents) -> contents
          end

        %{
          "path_safetensors_model" =>
            params_entry["path_safetensors_model"] || @path_safetensors_model,
          "path_safetensors_lora" =>
            params_entry["path_safetensors_lora"] || @path_safetensors_lora,
          "path_output_gguf" => params_entry["path_output_gguf"] || @path_output_gguf,
          "quantization" => params_entry["quantization"] || "f16",
          "base_model_id" => params_entry["base_model_id"],
          "integrate_in_ollama" => params_entry["integrate_in_ollama"] || false,
          "modelfile_contents" => modelfile_contents
        }
      else
        error ->
          IO.puts("Error loading params from JSON: #{inspect(error)}")

          %{
            "path_safetensors_model" => @path_safetensors_model,
            "path_safetensors_lora" => @path_safetensors_lora,
            "path_output_gguf" => @path_output_gguf,
            "modelfile_contents" => @modelfile_contents
          }
      end
    rescue
      e ->
        IO.puts("Exception loading params from JSON: #{inspect(e)}")

        %{
          "path_safetensors_model" => @path_safetensors_model,
          "path_safetensors_lora" => @path_safetensors_lora,
          "path_output_gguf" => @path_output_gguf,
          "modelfile_contents" => @modelfile_contents
        }
    end
  end

  def sanitize_model_name(name) do
    name
    |> String.downcase()
    |> String.replace(~r/[^A-Za-z0-9_-]/u, "_")
    |> String.replace(~r/_+/u, "_")
    |> String.replace(~r/^_|_$/u, "")
    |> then(fn s -> if s == "", do: "model", else: s end)
  end

  defp convert_charlists_in_map(map) when is_map(map) do
    Enum.reduce(map, %{}, fn
      {key, value}, acc when is_list(key) and key != [] and is_integer(hd(key)) ->
        key_str = List.to_string(key)
        acc |> Map.put(key_str, convert_charlists_in_map(value))

      {key, value}, acc ->
        acc |> Map.put(key, convert_charlists_in_map(value))
    end)
  end

  defp convert_charlists_in_map(value)
       when is_list(value) and value != [] and is_integer(hd(value)) do
    List.to_string(value)
  end

  defp convert_charlists_in_map(value) when is_list(value) do
    Enum.map(value, &convert_charlists_in_map/1)
  end

  defp convert_charlists_in_map(value), do: value

  defp convert_modelfile_contents(contents_map) when is_map(contents_map) do
    contents_map
    |> Enum.sort_by(fn {k, _} -> k end)
    |> Enum.map(fn {key, value} ->
      prefix =
        cond do
          String.starts_with?(key, "PARAMETER_") -> "PARAMETER "
          String.starts_with?(key, "SYSTEM_") -> "SYSTEM "
          true -> "#{key} "
        end

      "#{prefix}#{value}"
    end)
    |> Enum.join("\n")
    |> Kernel.<>("\n")
  end

  defp convert_modelfile_contents(_), do: nil

  def run_erlport_test do
    {python_path, 0} = System.cmd("python", ["-c", "import sys; print(sys.prefix)"])
    python_home = "D:/SDKs/Python310"
    modules_dir = "#{python_home}/my_modules"

    {:ok, pid} =
      :python.start_link([
        {:python, ~c"python"},
        {:python_path, String.to_charlist(modules_dir)}
      ])

    IO.puts("Python started successfully.")

    try do
      python_module = :test

      result = :python.call(pid, python_module, :print_success, [])
      :python.stop(pid)
      result
    catch
      kind, error ->
        IO.puts("Error: #{inspect(kind)} - #{inspect(error)}")
        :python.stop(pid)
        {:error, error}
    end
  end
end
