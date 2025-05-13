defmodule FineTuningController do
  alias GatewayAPI, as: GW
  require Logger

  @load_params_from_file true
  @use_param_entry "11"
  @ft_params_file_path Path.expand(Path.join([__DIR__, "..", "external", "ft_params.json"]))
  @merge_params_file_path Path.expand(Path.join([__DIR__, "..", "external", "merge_params.json"]))

  # This parameters are used if the loading from file is disabled
  @ft_params %{
    # Data parameters
    "mode" => "unsupervised",
    "data_path" =>
      "N:/Thesis/data_prepare/datasets_ready/unsupervised/single_chapters/combined_datasets",
    # change to "input" for supervised (as entries start with "input")
    "text_column" => "text",
    "pre_eval" => true,
    "eval_split" => 0,
    "max_length" => 3000,

    # Model parameters
    "model_path" =>
      "C:/Users/Paul/.cache/huggingface/hub/models--DiscoResearch--Llama3-DiscoLeo-Instruct-8B-v0.1/snapshots/0d7cf14a066a65f037d184d00eddc9d284225d93",
    "output_dir" =>
      "C:/Users/Paul/.cache/training_output/checkpoints_llama3_german_unsupervised_1",
    "use_flash_attention" => true,
    "quantization_config" => 8,

    # Training parameters
    "training_config" => %{
      "num_train_epochs" => 3,
      "learning_rate" => 0.00002,
      "per_device_train_batch_size" => 1,
      # set higher in case of memory insufficiency
      "gradient_accumulation_steps" => 8,
      # set higher for more stability in eval (less memory spikes)
      "eval_accumulation_steps" => 4
    },

    # PEFT parameters
    "peft_config" => %{
      "r" => 16,
      "lora_alpha" => 32,
      "lora_dropout" => 0.05
    }
  }

  @merge_params %{
    "base_model_path" => "C:/Users/Paul/.cache/merged_models/llama3_german_merged_unsupervised_1",
    "adapter_path" =>
      "C:/Users/Paul/.cache/training_output/checkpoints_llama3_german_unsupervised_2/final_model",
    "output_path" => "C:/Users/Paul/.cache/merged_models/llama3_german_merged_unsupervised_2",
    "use_fp16" => true
  }

  @merge_models_param %{}

  @doc """
  Initiates unsupervised fine-tuning using the Python module with progress reporting.

  Args:
    params: Optional map of parameters to override the defaults.
    progress_callback: Optional function to handle progress updates (receives string messages)

  Returns:
    {:ok, response} on success, {:error, formatted_reason} on failure.
  """
  def fine_tune(params \\ nil, progress_callback \\ nil) do
    # Start by testing the connection
    if GW.test_connection(:pytorch_finetuning) do
      Logger.info("Connection to Python module established successfully.")

      # Set up process to receive progress updates
      if progress_callback do
        # Start a process to handle callbacks
        parent = self()
        callback_pid = spawn_link(fn -> progress_receiver(parent, progress_callback) end)
        GW.register_progress_callback(callback_pid, :pytorch_finetuning)
      else
        # Just register this process to receive updates
        GW.register_progress_callback(nil, :pytorch_finetuning)
      end

      # Get base parameters (either from file or module attribute)
      base_params =
        if @load_params_from_file do
          case load_params_from_file() do
            {:ok, file_params} ->
              file_params

            {:error, reason} ->
              Logger.warning(
                "Failed to load parameters from file: #{reason}. Using default parameters."
              )

              @ft_params
          end
        else
          @ft_params
        end

      # Merge provided params with base params
      final_params =
        if params do
          Map.merge(base_params, params)
        else
          base_params
        end

      Logger.info("Starting unsupervised fine-tuning with parameters: #{inspect(final_params)}")

      # Call the Python function with better error formatting
      case GW.call_with_formatted_errors(
             :pytorch_finetuning,
             :initiate_finetuning,
             [final_params],
             %{restart: true},
             # 20 hours timeout
             72_000_000
           ) do
        {:ok, result} ->
          Logger.info("Fine-tuning completed successfully.")
          {:ok, result}

        {:error, reason} ->
          Logger.error("Fine-tuning failed: #{reason}")
          {:error, reason}
      end
    else
      error_msg = "Connection to Python module failed. Make sure the module is available."
      Logger.error(error_msg)
      {:error, error_msg}
    end
  end

  def merge_models(params \\ nil) do
    # Start by testing the connection
    if GW.test_connection(:merge_models) do
      Logger.info("Connection to Python module established successfully.")

      # Get base parameters (either from file or module attribute)
      base_params =
        if @load_params_from_file do
          case load_merge_params_from_file() do
            {:ok, file_params} ->
              file_params

            {:error, reason} ->
              Logger.warning(
                "Failed to load merge parameters from file: #{reason}. Using default parameters."
              )

              @merge_models_param
          end
        else
          @merge_models_param
        end

      # Merge provided params with base params
      final_params =
        if params do
          Map.merge(base_params, params)
        else
          base_params
        end

      IO.inspect(final_params, label: "Final Parameters for Model Merging")

      Logger.info("Starting model merging with parameters: #{inspect(final_params)}")

      # Call the Python function with better error formatting
      case GW.call_with_formatted_errors(
             :merge_models,
             :merge_models,
             [final_params],
             %{restart: true},
             # 2 hours timeout
             7_200_000
           ) do
        {:ok, result} ->
          Logger.info("Model merging completed successfully.")
          {:ok, result}

        {:error, reason} ->
          Logger.error("Model merging failed: #{reason}")
          {:error, reason}
      end
    else
      error_msg = "Connection to Python module failed. Make sure the module is available."
      Logger.error(error_msg)
      {:error, error_msg}
    end
  end

  def merge_base_model_and_adapter(params \\ nil) do
    # Start by testing the connection
    if GW.test_connection(:merge_models) do
      Logger.info("Connection to Python module established successfully.")

      # Get base parameters (either from file or module attribute)
      base_params =
        if @load_params_from_file do
          case load_merge_params_from_file() do
            {:ok, file_params} ->
              file_params

            {:error, reason} ->
              Logger.warning(
                "Failed to load merge parameters from file: #{reason}. Using default parameters."
              )

              @merge_params
          end
        else
          @merge_params
        end

      # Merge provided params with base params
      final_params =
        if params do
          Map.merge(base_params, params)
        else
          base_params
        end

      Logger.info("Starting model merging with parameters: #{inspect(final_params)}")

      # Call the Python function with better error formatting
      case GW.call_with_formatted_errors(
             :merge_models,
             :merge_adapters_into_base_model,
             [final_params],
             %{restart: true},
             # 2 hours timeout
             7_200_000
           ) do
        {:ok, result} ->
          Logger.info("Model merging completed successfully.")
          {:ok, result}

        {:error, reason} ->
          Logger.error("Model merging failed: #{reason}")
          {:error, reason}
      end
    else
      error_msg = "Connection to Python module failed. Make sure the module is available."
      Logger.error(error_msg)
      {:error, error_msg}
    end
  end

  def get_pre_info do
    # get number of steps and training time, analyse memory usage
  end

  @doc """
  Loads merge parameters from the JSON file based on the entry specified by @use_param_entry.

  Returns:
    {:ok, params} on success, {:error, reason} on failure.
  """
  def load_merge_params_from_file do
    try do
      with {:ok, file_content} <- File.read(@merge_params_file_path),
           {:ok, json_data} <- Jason.decode(file_content),
           {:ok, params} <- extract_params_entry(json_data, @use_param_entry) do
        Logger.info("Successfully loaded merge parameters from file: #{@merge_params_file_path}")
        {:ok, params}
      else
        {:error, %Jason.DecodeError{} = error} ->
          error_msg = "Failed to parse JSON file: #{inspect(error)}"
          Logger.error(error_msg)
          {:error, error_msg}

        {:error, :file_read_error, reason} ->
          error_msg = "Failed to read parameters file: #{inspect(reason)}"
          Logger.error(error_msg)
          {:error, error_msg}

        {:error, :missing_entry, entry} ->
          error_msg = "Entry '#{entry}' not found in parameters file"
          Logger.error(error_msg)
          {:error, error_msg}

        {:error, reason} ->
          error_msg = "Error loading parameters: #{inspect(reason)}"
          Logger.error(error_msg)
          {:error, error_msg}
      end
    rescue
      e ->
        error_msg = "Unexpected error loading parameters: #{inspect(e)}"
        Logger.error(error_msg)
        {:error, error_msg}
    end
  end

  # Process to receive progress updates and forward to callback function
  defp progress_receiver(parent, callback_fn) when is_function(callback_fn, 1) do
    receive do
      {:python_progress, message} when is_binary(message) ->
        # Forward to callback and log
        callback_fn.(message)
        Logger.debug("Progress: #{message}")
        progress_receiver(parent, callback_fn)

      {:python_progress, message} when is_list(message) ->
        # Convert charlist to binary string
        str_message = List.to_string(message)
        callback_fn.(str_message)
        Logger.debug("Progress: #{str_message}")
        progress_receiver(parent, callback_fn)

      {:EXIT, ^parent, _reason} ->
        # Parent process exited, stop receiver
        :ok

      other ->
        Logger.debug("Unexpected message in progress_receiver: #{inspect(other)}")
        progress_receiver(parent, callback_fn)
    end
  end

  @doc """
  Generates text using a fine-tuned model with better error formatting.

  Args:
    prompt: Text prompt for generation.
    model_path: Path to the fine-tuned model.
    generation_config: Optional parameters for text generation.

  Returns:
    {:ok, generated_text} on success, {:error, formatted_reason} on failure.
  """
  def generate_text(prompt, model_path, generation_config \\ nil) do
    if GW.test_connection(:pytorch_finetuning) do
      params = %{
        "prompt" => prompt,
        "model_path" => model_path,
        "generation_config" => generation_config
      }

      # Use the new formatted error handling
      GW.call_with_formatted_errors(
        :pytorch_finetuning,
        :generate_text_from_controller,
        [params],
        %{reload: false},
        30_000
      )
    else
      {:error, "Connection to Python module failed"}
    end
  end

  @doc """
  Interactive progress demo - prints progress to console.
  Useful for testing progress reporting.
  """
  def interactive_tuning(params \\ nil) do
    IO.puts("Starting interactive fine-tuning with console progress reporting...")

    # Define callback that prints to console
    callback = fn message ->
      IO.puts("\n[PROGRESS] #{message}")
    end

    # Start fine-tuning with console callback
    result = fine_tune(params, callback)

    # Print final result
    case result do
      {:ok, _} ->
        IO.puts("\n✅ Fine-tuning completed successfully!")

      {:error, reason} ->
        IO.puts("\n❌ Fine-tuning failed: #{reason}")
    end

    result
  end

  @doc """
  Loads parameters from the JSON file based on the entry specified by @use_param_entry.

  Returns:
    {:ok, params} on success, {:error, reason} on failure.
  """
  def load_params_from_file do
    try do
      with {:ok, file_content} <- File.read(@ft_params_file_path),
           {:ok, json_data} <- Jason.decode(file_content),
           {:ok, params} <- extract_params_entry(json_data, @use_param_entry) do
        Logger.info("Successfully loaded parameters from file: #{@ft_params_file_path}")
        {:ok, params}
      else
        {:error, %Jason.DecodeError{} = error} ->
          error_msg = "Failed to parse JSON file: #{inspect(error)}"
          Logger.error(error_msg)
          {:error, error_msg}

        {:error, :file_read_error, reason} ->
          error_msg = "Failed to read parameters file: #{inspect(reason)}"
          Logger.error(error_msg)
          {:error, error_msg}

        {:error, :missing_entry, entry} ->
          error_msg = "Entry '#{entry}' not found in parameters file"
          Logger.error(error_msg)
          {:error, error_msg}

        {:error, reason} ->
          error_msg = "Error loading parameters: #{inspect(reason)}"
          Logger.error(error_msg)
          {:error, error_msg}
      end
    rescue
      e ->
        error_msg = "Unexpected error loading parameters: #{inspect(e)}"
        Logger.error(error_msg)
        {:error, error_msg}
    end
  end

  defp extract_params_entry(json_data, entry) do
    case Map.fetch(json_data, entry) do
      {:ok, params} when is_map(params) and map_size(params) > 0 ->
        # Convert keys to Elixir-style strings and ensure nested maps are handled
        converted_params = convert_json_to_elixir_map(params)
        {:ok, converted_params}

      {:ok, _} ->
        {:error, :missing_entry, entry}

      :error ->
        {:error, :missing_entry, entry}
    end
  end

  # Converts JSON map format to Elixir map format (handles nested structures)
  defp convert_json_to_elixir_map(map) when is_map(map) do
    Enum.reduce(map, %{}, fn {k, v}, acc ->
      # If value is a map, recursively convert it
      converted_value = if is_map(v), do: convert_json_to_elixir_map(v), else: v
      # Put the key as string and converted value
      Map.put(acc, k, converted_value)
    end)
  end
end
