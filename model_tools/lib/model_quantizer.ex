defmodule ModelQuantizer do
  @moduledoc """
  API for quantizing LLM models.
  Uses the llama.cpp library for quantization via Python.
  """

  @load_params_from_file true
  @use_param_entry "2"
  @quant_params_file_path Path.expand(Path.join([__DIR__, "..", "external", "quant_params.json"]))

  # Path of the wrapper for running the Python quantizer script
  @path_quantizer Path.join([
                    File.cwd!(),
                    "external"
                  ])

  @quant_types %{
    "q4_0" => "Q4_0",
    "q4_1" => "Q4_1",
    "q5_0" => "Q5_0",
    "q5_1" => "Q5_1",
    "iq2_xxs" => "IQ2_XXS",
    "iq2_xs" => "IQ2_XS",
    "iq2_s" => "IQ2_S",
    "iq2_m" => "IQ2_M",
    "iq1_s" => "IQ1_S",
    "iq1_m" => "IQ1_M",
    "tq1_0" => "TQ1_0",
    "tq2_0" => "TQ2_0",
    "q2_k" => "Q2_K",
    "q2_k_s" => "Q2_K_S",
    "iq3_xxs" => "IQ3_XXS",
    "iq3_s" => "IQ3_S",
    "iq3_m" => "IQ3_M",
    "q3_k" => "Q3_K",
    "iq3_xs" => "IQ3_XS",
    "q3_k_s" => "Q3_K_S",
    "q3_k_m" => "Q3_K_M",
    "q3_k_l" => "Q3_K_L",
    "iq4_nl" => "IQ4_NL",
    "iq4_xs" => "IQ4_XS",
    "q4_k" => "Q4_K",
    "q4_k_s" => "Q4_K_S",
    "q4_k_m" => "Q4_K_M",
    "q5_k" => "Q5_K",
    "q5_k_s" => "Q5_K_S",
    "q5_k_m" => "Q5_K_M",
    "q6_k" => "Q6_K",
    "q8_0" => "Q8_0",
    "f16" => "F16",
    "bf16" => "BF16",
    "f32" => "F32",
    "copy" => "COPY"
  }

  # Function to normalize quantization type
  def normalize_quant_type(quant_type) when is_binary(quant_type) do
    # Convert to lowercase for consistent comparison
    lowercase_type = String.downcase(quant_type)

    cond do
      Map.has_key?(@quant_types, lowercase_type) ->
        @quant_types[lowercase_type]

      # Try with underscores replaced by hyphens
      Map.has_key?(@quant_types, String.replace(lowercase_type, "-", "_")) ->
        @quant_types[String.replace(lowercase_type, "-", "_")]

      # Try with hyphens replaced by underscores
      Map.has_key?(@quant_types, String.replace(lowercase_type, "_", "-")) ->
        @quant_types[String.replace(lowercase_type, "_", "-")]

      # Try without any separators - capture the value in a variable first
      true ->
        stripped_type =
          lowercase_type
          |> String.replace(~r/[_\-]/, "")
          |> String.replace(~r/^q/, "q_")
          |> String.replace(~r/^iq/, "iq_")
          |> String.replace(~r/^tq/, "tq_")

        # Find matching key by comparing without underscores
        matching_entry =
          Enum.find(@quant_types, fn {k, _v} ->
            String.replace(k, "_", "") == stripped_type
          end)

        case matching_entry do
          {_k, v} ->
            v

          nil ->
            IO.puts("Warning: Unknown quantization type '#{quant_type}', defaulting to q8_0")
            "q8_0"
        end
    end
  end

  def normalize_quant_type(nil), do: "q8_0"

  # Quantize model using parameters from file_param_entry
  def quantize_model(file_param_entry \\ nil)
      when is_binary(file_param_entry) or is_nil(file_param_entry) or is_integer(file_param_entry) do
    param_entry =
      if is_integer(file_param_entry),
        do: Integer.to_string(file_param_entry),
        else: file_param_entry

    params = get_quantization_params(param_entry)

    quantize_model_impl(
      params["path_input_model"],
      params["path_output_dir"],
      params["quantization"],
      params["integrate_in_ollama"],
      params["modelfile_contents"],
      params["use_safetensors"]
    )
  end

  # Implementation function for quantization
  def quantize_model_impl(
        path_input_model,
        path_output_dir,
        quantization \\ "q8_0",
        integrate_in_ollama \\ false,
        modelfile_contents \\ nil,
        use_safetensors \\ false
      ) do
    # Normalize the quantization type
    normalized_quant = normalize_quant_type(quantization)

    {:ok, pid} =
      :python.start_link([
        {:python, ~c"python"},
        {:python_path, String.to_charlist(@path_quantizer)}
      ])

    # Determine input type
    is_dir = File.dir?(path_input_model)
    is_gguf = is_binary(path_input_model) and String.ends_with?(path_input_model, ".gguf")

    is_safetensors =
      (is_binary(path_input_model) and String.ends_with?(path_input_model, ".safetensors")) or
        (is_dir and
           File.ls!(path_input_model) |> Enum.any?(&String.ends_with?(&1, ".safetensors")))

    file_basename =
      cond do
        is_dir ->
          Path.basename(path_input_model)

        is_gguf or is_safetensors ->
          Path.basename(path_input_model, Path.extname(path_input_model))

        true ->
          "model"
      end

    output_filename = "#{file_basename}_#{normalized_quant}.gguf"
    path_output = Path.join(path_output_dir, output_filename)

    IO.inspect(
      %{
        is_dir: is_dir,
        is_gguf: is_gguf,
        is_safetensors: is_safetensors,
        use_safetensors: use_safetensors,
        path_input_model: path_input_model
      },
      label: "Quantize input info"
    )

    result =
      try do
        cond do
          is_gguf and use_safetensors ->
            raise ArgumentError,
                  "Cannot use 'use_safetensors=true' when input is already a GGUF file."

          is_gguf ->
            # Quantize GGUF file to another GGUF file
            :python.call(pid, :quantize_wrapper, :quantize_model, [
              String.to_charlist(path_input_model),
              String.to_charlist(path_output),
              String.to_charlist(normalized_quant)
            ])
            |> convert_charlists_in_map()

          (is_safetensors or is_dir) and use_safetensors ->
            # Quantize safetensors model and keep as safetensors
            :python.call(pid, :quantize_wrapper, :quantize_safetensors_inplace, [
              String.to_charlist(path_input_model),
              String.to_charlist(path_output),
              String.to_charlist(normalized_quant)
            ])
            |> convert_charlists_in_map()

          (is_safetensors or is_dir) and not use_safetensors ->
            # Quantize safetensors model (convert to GGUF with quantization)
            :python.call(pid, :quantize_wrapper, :quantize_safetensors_model, [
              String.to_charlist(path_input_model),
              String.to_charlist(path_output),
              String.to_charlist(normalized_quant)
            ])
            |> convert_charlists_in_map()

          true ->
            {:error, "Unknown input type for quantization: #{inspect(path_input_model)}"}
        end
      catch
        kind, error ->
          IO.puts("Error during Python call: #{inspect(kind)} - #{inspect(error)}")
          {:error, "Python error: #{inspect(error)}"}
      end

    try do
      quantization_result =
        case result do
          %{"returncode" => 0, "stdout" => stdout} ->
            IO.puts("Quantization successful.")
            {:ok, stdout}

          %{"returncode" => code, "stderr" => stderr} ->
            {:error, "Exit code: #{code}, Error: #{stderr}"}

          _ ->
            {:error, "Unexpected result structure: #{inspect(result)}"}
        end

      final_result =
        if integrate_in_ollama do
          try do
            case quantization_result do
              {:ok, _} ->
                ModelConverter.integrate_gguf_in_ollama_impl(
                  path_output,
                  true,
                  modelfile_contents
                )

              error ->
                error
            end
          catch
            kind, error ->
              IO.puts(
                "Error while integrating quantized model in Ollama: #{inspect(kind)} - #{inspect(error)}"
              )

              {:error, error}
          end
        else
          quantization_result
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

  # Get quantization parameters from JSON file or use defaults
  def get_quantization_params(file_param_entry \\ nil) do
    param_entry = file_param_entry || @use_param_entry

    if @load_params_from_file do
      load_params_from_json(param_entry)
    else
      %{}
    end
  end

  # Load parameters from JSON file
  def load_params_from_json(param_entry \\ nil) do
    param_entry = param_entry || @use_param_entry

    try do
      with {:ok, json_content} <- File.read(@quant_params_file_path),
           {:ok, json_parsed} <- Jason.decode(json_content),
           params_entry = json_parsed[param_entry] || json_parsed["1"] || %{} do
        # Get quantization and normalize it
        raw_quant = params_entry["quantization"] || "q8_0"
        normalized_quant = normalize_quant_type(raw_quant)

        # Construct final params map
        %{
          "path_input_model" => params_entry["path_input_model"],
          "path_output_dir" => params_entry["path_output_dir"],
          "quantization" => normalized_quant,
          "integrate_in_ollama" => params_entry["integrate_in_ollama"] || false,
          "modelfile_contents" => params_entry["modelfile_contents"],
          "use_safetensors" => params_entry["use_safetensors"] || false
        }
      else
        error ->
          IO.puts("Error loading params from JSON: #{inspect(error)}")
          %{}
      end
    rescue
      e ->
        IO.puts("Exception loading params from JSON: #{inspect(e)}")
        %{}
    end
  end

  # Helper function to convert charlists to strings in maps
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
end
