defmodule SSE.SubSymbolicExtractor do
  alias LangChain.Message
  alias LangChain.ChatModels.ChatOllamaAI
  alias LangChain.PromptTemplate
  alias SSE.PromptCreation

  @llm_parameters %{
    # "leo_mistral_german",
    # "llama3.2:3b-instruct-fp16",
    # "leo_llama_german_13b_q8",
    model: "llama3_german_instruct",
    temperature: 0.1,
    top_p: 0.5,
    top_k: 40,
    # max_tokens equal to num_predict in ollama modelfile
    max_tokens: 1024
    # repeat_penalty: 1.2
    # repeat_last_n: 256
  }

  @filename_categories "N:/Thesis/data_prepare/statistics/cluster_filename_categories.json"
  @temp_path "N:/Thesis/temp"

  # Load filename categories from JSON file
  defp load_filename_categories do
    case File.read(@filename_categories) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, categories} -> categories
          {:error, _} -> %{}
        end

      {:error, _} ->
        %{}
    end
  end

  # Check if filename is in the "Technische_Daten" category
  defp is_technical_data_file?(filename, categories) do
    tech_data_files = Map.get(categories, "Technische_Daten", [])
    Enum.member?(tech_data_files, filename)
  end

  # extract function that accepts a map of files (arbitrary many files, JSON or text-based (txt / md))
  # Example: %{"full_report" => "content...", "extracted_meta_info" => "content..."}
  def extract(files_map, single_chapters \\ true, only_summary \\ true)
      when is_map(files_map) and map_size(files_map) > 0 do
    # Initialize Ollama client with timeout
    # 2-minute timeout
    client = Ollama.init(receive_timeout: 120_000)

    # Process the files_map to ensure proper formatting
    processed_files_map_initial = Map.new(files_map, fn {k, v} -> {k, format_input(v)} end)

    # Get system prompt
    system_prompt = PromptCreation.extract_prompt_element("system_prompt")
    # Initialize system prompt message
    system_message = [%Message{role: "system", content: system_prompt}]

    # Load the filename categories
    categories = load_filename_categories()

    extraction_results =
      if single_chapters do
        # Get extracted meta info if available
        extracted_meta_info = Map.get(processed_files_map_initial, "extracted_meta_info", "{}")
        # Remove extracted_meta_info key from processed_files_map
        processed_files_map = Map.delete(processed_files_map_initial, "extracted_meta_info")

        # Extract files for processing
        processable_files =
          processed_files_map
          |> Enum.sort_by(fn {key, _} ->
            key_str = to_string(key)

            # Check if the filename starts with a number (e.g., "1. Vorwort")
            case Regex.run(~r/^(\d+)[\.\s]/, key_str) do
              [_, num_str] ->
                # If it starts with a number, sort numerically
                {0, String.to_integer(num_str), key_str}

              nil ->
                # If not a numbered chapter, sort after numbered chapters
                {1, 0, key_str}
            end
          end)

        # Process each file individually and aggregate results into extraction_results
        Enum.reduce(processable_files, %{}, fn {key, content}, results ->
          # Get proper filename from the key
          filename = to_string(key)

          # Skip empty content
          if String.trim(content) == "" do
            results
          else
            # Check if the filename is in the "Technische_Daten" category
            is_technical_data =
              is_technical_data_file?(PromptCreation.sanitize_filename(filename), categories)

            # For each file, construct prompts using the filename
            # Returns list of maps with messages, category, and questions
            prompt_data =
              try do
                cond do
                  # Skip processing technical data files when only_summary is true
                  is_technical_data and only_summary ->
                    []

                  # For technical data files (when not only_summary), include extracted_meta_info
                  is_technical_data ->
                    PromptCreation.construct_prompt(filename, {extracted_meta_info, content})

                  # For regular files with only_summary
                  only_summary ->
                    [PromptCreation.construct_summary_prompt(filename, content)]

                  # For regular files with full processing
                  true ->
                    category_prompt_data = PromptCreation.construct_prompt(filename, content)
                    summary_chapter = PromptCreation.construct_summary_prompt(filename, content)
                    [summary_chapter | category_prompt_data]
                end
              rescue
                e ->
                  IO.puts("Error constructing prompt for #{filename}: #{inspect(e)}")
                  # Fallback to a basic prompt when the template processing fails
                  construct_fallback_prompt(content)
              end

            # Skip files with empty prompt_data (technical files in only_summary mode)
            if prompt_data == [] do
              results
            else
              file_results =
                Enum.reduce(prompt_data, %{}, fn %{
                                                   messages: messages,
                                                   category: category,
                                                   questions: questions
                                                 },
                                                 file_acc ->
                  # Add system message to the message list
                  full_messages = system_message ++ messages

                  # count words of message and calculate roughly the number of tokens
                  result = PromptCreation.count_words_and_tokens(messages)
                  words = result.word_count
                  tokens = result.estimated_tokens

                  IO.puts("\nWord count: #{words}, Estimated tokens: #{tokens}")
                  IO.inspect(full_messages, label: "Full Messages")

                  # Process with LLM
                  llm_response = process_with_llm(client, full_messages)
                  IO.inspect(llm_response, label: "LLM Response")

                  # Parse response into appropriate format based on mode
                  response_content =
                    if only_summary do
                      # For summary, join the list items into a single string with newlines
                      parse_response_to_list_items(llm_response) |> Enum.join("\n")
                    else
                      # For full extraction mode, keep the list format
                      parse_response_to_list_items(llm_response)
                    end

                  # Create different entry structure based on whether only_summary is true
                  if only_summary do
                    # For summary-only mode, use the new structure where "included_meta_data" and "summary" are top-level
                    Map.put(file_acc, "included_meta_data", %{})
                    |> Map.put("summary", response_content)
                  else
                    # For full extraction mode, use the original structure
                    # Check if the category already exists in accumulator
                    existing_entry = Map.get(file_acc, category)

                    # Create or update category entry with questions and parsed response
                    category_entry =
                      if existing_entry do
                        # Merge with existing entry by concatenating questions and responses
                        %{
                          "questions" => (existing_entry["questions"] || []) ++ questions,
                          "response" => (existing_entry["response"] || []) ++ response_content
                        }
                      else
                        # Create new entry
                        %{
                          "questions" => questions,
                          "response" => response_content
                        }
                      end

                    # Add or update category entry to file accumulator
                    Map.put(file_acc, category, category_entry)
                  end
                end)

              # Add file results to overall results
              Map.put(results, filename, file_results)
            end
          end
        end)
      else
        # For full report processing
        prompt_initial = PromptCreation.extract_prompt_element("prompt_initial")
        full_report_content = Map.get(processed_files_map_initial, "full_report", "")

        # Create a prompt template
        template_result =
          try do
            PromptTemplate.from_template(
              PromptCreation.extract_prompt_element("prompt_full_report")
            )
          rescue
            e ->
              IO.puts("Error creating template: #{inspect(e)}")
              {:error, e}
          end

        # Process full report
        full_messages =
          case template_result do
            {:ok, template} ->
              # Format the template with the full report
              formatted_result =
                try do
                  PromptTemplate.format(template, %{full_report: full_report_content})
                rescue
                  e ->
                    IO.puts("Error formatting template: #{inspect(e)}")
                    {:error, e}
                end

              case formatted_result do
                {:ok, formatted_prompt} ->
                  [
                    %Message{role: "system", content: system_prompt},
                    %Message{role: "user", content: prompt_initial <> formatted_prompt}
                  ]

                {:error, _} ->
                  [
                    %Message{role: "system", content: system_prompt},
                    %Message{
                      role: "user",
                      content: prompt_initial <> "\n\n" <> full_report_content
                    }
                  ]
              end

            {:error, _} ->
              [
                %Message{role: "system", content: system_prompt},
                %Message{role: "user", content: prompt_initial <> "\n\n" <> full_report_content}
              ]
          end

        # Process with LLM
        llm_response = process_with_llm(client, full_messages)
        parsed_response = parse_response_to_list_items(llm_response)

        # Create a single entry for the full report
        %{
          "full_report" => %{
            "general" => %{
              "questions" => ["Gesamtüberblick"],
              "response" => parsed_response
            }
          }
        }
      end

    # Return the extraction results as a JSON string
    Jason.encode!(extraction_results, pretty: true)
  end

  # Parse LLM response into list items based on newline patterns
  defp parse_response_to_list_items(response) do
    # Split by newlines followed by an uppercase letter and dot or number and dot pattern
    items = Regex.split(~r/\n\s*(?=[A-Z]\.|[0-9]+\.)/, response, trim: true)

    # Clean up each item and ensure they begin with a letter/number and period if they matched the pattern
    items
    |> Enum.map(&String.trim/1)
    |> Enum.map(fn item ->
      cond do
        # If item already starts with letter/number and period, keep as is
        Regex.match?(~r/^[A-Z]\.|^[0-9]+\./, item) -> item
        # Otherwise, it's likely a continuation or separate point
        true -> item
      end
    end)
    |> Enum.reject(fn item -> item == "" end)
  end

  # Reset the LLM context by unloading and preloading the model
  defp reset_llm_context(client, model \\ nil) do
    model_name = model || @llm_parameters.model

    # First try to unload the model
    unload_result = Ollama.unload(client, model: model_name)

    case unload_result do
      {:ok, _response} ->
        IO.puts("Successfully unloaded model: #{model_name}")
        # Then preload it again
        case Ollama.preload(client, model: model_name) do
          {:ok, _} ->
            IO.puts("Successfully preloaded model: #{model_name}")
            :ok

          {:error, reason} ->
            IO.puts("Warning: Failed to preload LLM model: #{inspect(reason)}")
            :error
        end

      {:error, reason} ->
        IO.puts("Warning: Failed to unload LLM model: #{inspect(reason)}")
        # Try to preload anyway
        case Ollama.preload(client, model: model_name) do
          {:ok, _} ->
            IO.puts("Successfully preloaded model: #{model_name}")
            :ok

          {:error, preload_reason} ->
            IO.puts("Warning: Failed to preload LLM model: #{inspect(preload_reason)}")
            :error
        end
    end
  end

  # Process a single message set with the LLM
  defp process_with_llm(client, message_set, reset_context \\ false) do
    # Reset context if requested
    if reset_context do
      reset_llm_context(client)
    end

    # Convert LangChain Message format to Ollama format
    ollama_messages =
      Enum.map(message_set, fn msg ->
        %{
          role: msg.role,
          content: msg.content
        }
      end)

    # Create options map dynamically from LLM parameters
    # Start with an empty map
    options = %{}

    # Define a mapping from @llm_parameters keys to Ollama option keys
    param_mapping = %{
      temperature: :temperature,
      top_p: :top_p,
      top_k: :top_k,
      max_tokens: :num_predict,
      repeat_penalty: :repeat_penalty,
      repeat_last_n: :repeat_last_n
    }

    # Dynamically add only the parameters that exist in @llm_parameters
    options =
      Enum.reduce(param_mapping, options, fn {param_key, option_key}, acc ->
        if Map.has_key?(@llm_parameters, param_key) do
          Map.put(acc, option_key, Map.get(@llm_parameters, param_key))
        else
          acc
        end
      end)

    # Try up to 3 times with timeout
    process_with_retries(client, ollama_messages, @llm_parameters.model, options, 3)
  end

  # Process with retries and timeout
  defp process_with_retries(client, ollama_messages, model, options, attempts_left)
       when attempts_left > 0 do
    # Using a direct call instead of Task.async to avoid potential concurrency issues
    try do
      # Set a timeout with a catch-all rescue block
      result =
        Ollama.chat(client,
          model: model,
          messages: ollama_messages,
          options: options
          # Timeout is now handled by the client initialization
        )

      case result do
        {:ok, response} ->
          # Successful response
          response["message"]["content"]

        {:error, reason} ->
          # Error occurred
          IO.puts("LLM processing error: #{inspect(reason)}. Attempts left: #{attempts_left - 1}")
          process_with_retries(client, ollama_messages, model, options, attempts_left - 1)
      end
    rescue
      e ->
        # Handle exceptions (timeouts, connection errors, etc.)
        IO.puts(
          "Exception during LLM processing: #{inspect(e)}. Attempts left: #{attempts_left - 1}"
        )

        process_with_retries(client, ollama_messages, model, options, attempts_left - 1)
    catch
      :exit, reason ->
        # Handle exit signals
        IO.puts(
          "Process exit during LLM processing: #{inspect(reason)}. Attempts left: #{attempts_left - 1}"
        )

        process_with_retries(client, ollama_messages, model, options, attempts_left - 1)
    end
  end

  defp process_with_retries(_client, _ollama_messages, _model, _options, 0) do
    IO.puts("Maximum LLM processing attempts reached. Returning error placeholder.")
    "ERROR IN PROCESSING"
  end

  # Format map or list as pretty JSON string or key-value pairs
  defp format_json_data(data) when is_map(data) do
    # Try to encode as pretty JSON
    case Jason.encode(data, pretty: true) do
      {:ok, json_string} ->
        json_string

      {:error, _} ->
        # If encoding fails, format as key-value pairs
        data
        |> Enum.map(fn {key, value} -> "#{key}: #{inspect(value)}" end)
        |> Enum.join("\n")
    end
  end

  defp format_json_data(data) when is_list(data) do
    # Handle list of items (could be objects or simple values)
    case Jason.encode(data, pretty: true) do
      {:ok, json_string} ->
        json_string

      {:error, _} ->
        Enum.map_join(data, "\n", &inspect/1)
    end
  end

  # Main function to format any input data (works for both report and meta_data)
  defp format_input(nil), do: ""

  defp format_input(data) when is_binary(data) do
    # Check if the string is JSON
    case Jason.decode(data) do
      {:ok, decoded_data} ->
        # If it's valid JSON, prettify it
        format_json_data(decoded_data)

      {:error, _} ->
        # If not JSON, return the data as is
        data
    end
  end

  defp format_input(data) when is_map(data) or is_list(data) do
    format_json_data(data)
  end

  defp format_input(data) do
    # For any other data type, convert to string representation
    inspect(data)
  end

  def construct_fallback_prompt(content) do
    # Fallback to a basic prompt when the template processing fails
    [
      %{
        messages: [
          %Message{
            role: "user",
            content: """
            Analysiere das folgende Kapitel eines Gutachtens:

            #{content}

            Extrahiere die wichtigsten Informationen in Stichpunkten.
            """
          }
        ],
        category: "ERROR_in_prompt_creation",
        questions: ["Wichtigste Informationen"]
      }
    ]
  end
end
