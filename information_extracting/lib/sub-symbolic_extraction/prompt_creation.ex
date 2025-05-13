defmodule SSE.PromptCreation do
  @moduledoc """
  This module handles the creation of prompts for the SubSymbolicExtractor.
  """

  alias LangChain.Message
  alias LangChain.PromptTemplate

  @filename_categories Path.join([
                         Path.dirname(__ENV__.file),
                         "cluster_filename_categories.json"
                       ])

  @prompt_elements Path.join([
                     Path.dirname(__ENV__.file),
                     "prompt_elements.json"
                   ])

  @doc """
  Constructs prompts based on filename classification and chapter content.
  Returns a list of maps, each containing messages, category, and questions.

  ## Parameters
  - filename: The filename to classify to determine prompt templates
  - chapter_content: The content to analyze and include in the prompts

  ## Returns
  - List of maps with keys: messages, category, questions
  """
  def construct_prompt(filename, chapter_content, variant \\ "b", num_questions \\ 2) do
    # Get categories from the filename
    categories = classify_filename(filename)

    # Get the chapter elements templates
    chapter_elements = extract_prompt_element("prompt_chapter_elements") || %{}

    IO.puts("\n")
    IO.inspect(filename, label: "Filename")
    IO.inspect(categories, label: "Categories")

    # Check for Technische_Daten category
    if "Technische_Daten" in categories do
      # Special handling for Technische_Daten
      base_template = extract_prompt_element("prompt_compare_text_json_initial")
      tech_data_template = extract_prompt_element("examples_compare_text_json")
      tech_data_element = Map.get(chapter_elements, "Technische_Daten", %{})
      tech_data_examples = Map.get(tech_data_element, "examples", [])

      # Generate examples section using the specialized structure
      examples_section = generate_tech_data_examples_section(tech_data_examples)

      {json_data, content_data} = chapter_content

      # Create the prompt text
      prompt_text =
        base_template
        |> String.replace("<%= @examples_section %>", examples_section)
        |> String.replace("<%= @chapter %>", content_data)
        |> String.replace("<%= @json %>", json_data)
        |> String.replace("\\n", "\n")

      # Create message
      message = [
        %Message{role: "user", content: prompt_text}
      ]

      # Return a single result for Technische_Daten
      [
        %{
          messages: message,
          category: "Technische_Daten",
          questions: ["Fehlende Meta-Daten"]
        }
      ]
    else
      # Standard processing for other categories
      # Get base template for chapter prompt
      base_template =
        case variant do
          "a" ->
            # Use the first variant of the prompt template
            extract_prompt_element("prompt_chapter_initial_a")

          "b" ->
            # Use the second variant of the prompt template
            extract_prompt_element("prompt_chapter_initial_b")
        end

      # If we found categories, construct specific prompts for each category
      if categories != [] do
        # For each category, try to find matching prompt elements and construct a prompt
        Enum.flat_map(categories, fn category ->
          if Map.has_key?(chapter_elements, category) do
            category_element = Map.get(chapter_elements, category)

            # Extract all questions and examples
            all_questions = Map.get(category_element, "questions", [])
            all_examples = Map.get(category_element, "examples", [])

            # Chunk questions and examples based on num_questions parameter
            create_prompt_chunks(
              category,
              all_questions,
              all_examples,
              chapter_content,
              base_template,
              variant,
              num_questions
            )
          else
            []
          end
        end)
      else
        []
      end
    end
  end

  def construct_summary_prompt(filename, chapter_content, use_specific_category \\ true) do
    categories = classify_filename(filename)
    IO.puts("\n\n\n")
    IO.inspect(categories, label: "Categories")
    IO.puts("\n")

    first_category =
      if use_specific_category do
        "Schadenursache_Angaben"
      else
        List.first(categories) || "Umfang"
      end

    # Get base summary template
    summary_template = extract_prompt_element("prompt_summary")

    # Get the chapter elements templates
    chapter_elements = extract_prompt_element("prompt_chapter_elements") || %{}

    # Get category-specific summary examples
    category_element = Map.get(chapter_elements, first_category, %{})
    summary_examples = Map.get(category_element, "summary", [])

    # Generate examples section with the summary examples
    examples_section = generate_summary_examples_section(summary_examples)

    # Create the prompt text
    prompt_text =
      summary_template
      |> String.replace("<%= @examples_section %>", examples_section)
      |> String.replace("<%= @chapter %>", chapter_content)
      |> String.replace("\\n", "\n")

    # Create message
    message = [
      %Message{role: "user", content: prompt_text}
    ]

    # Return the summary prompt information
    %{
      messages: message,
      category: "Summary",
      questions: ["Zusammenfassung des Kapitels"]
    }
  end

  @doc """
  Creates multiple prompt chunks for a category based on the max number of questions per prompt.
  """
  def create_prompt_chunks(
        category,
        all_questions,
        all_examples,
        chapter_content,
        base_template,
        variant,
        num_questions
      ) do
    # Calculate how many chunks we need
    total_questions = length(all_questions)
    chunks_needed = max(1, ceil(total_questions / num_questions))
    IO.inspect(chunks_needed, label: "Chunks Needed")

    # Create a prompt for each chunk
    0..(chunks_needed - 1)
    |> Enum.map(fn chunk_index ->
      # Calculate start and end indices for this chunk
      start_idx = chunk_index * num_questions
      end_idx = min(start_idx + num_questions - 1, total_questions - 1)

      # Extract the questions for this chunk
      chunk_questions = Enum.slice(all_questions, start_idx, num_questions)

      # Take the same subset from examples that we took from questions
      # This assumes the examples are in the same order as questions
      chunk_examples =
        if length(all_examples) > 0 do
          # For each example in all_examples
          Enum.map(all_examples, fn example ->
            # Find the keys that contain example_text and example_answers
            example_keys = Map.keys(example)
            text_key = Enum.find(example_keys, fn k -> String.contains?(k, "example_text") end)

            answers_key =
              Enum.find(example_keys, fn k -> String.contains?(k, "example_answers") end)

            # Get the example text (keep all text for context)
            example_text = Map.get(example, text_key, "")

            # Get all answers and slice the same portion as we did with questions
            all_answers = Map.get(example, answers_key, [])

            sliced_answers =
              if is_list(all_answers) do
                Enum.slice(all_answers, start_idx, num_questions)
              else
                all_answers
              end

            # Return a new example with full text but sliced answers
            Map.merge(%{}, %{
              text_key => example_text,
              answers_key => sliced_answers
            })
          end)
        else
          []
        end

      # Format the questions as a string
      questions_text = Enum.join(chunk_questions, "\n")

      # Generate examples section with the sliced examples
      examples_section =
        generate_examples_section(
          %{"examples" => chunk_examples},
          questions_text,
          variant
        )

      # Manually interpolate the template variables
      prompt_text =
        base_template
        |> String.replace("<%= @questions %>", questions_text)
        |> String.replace("<%= @examples_section %>", examples_section)
        |> String.replace("<%= @chapter %>", chapter_content)
        # Convert escaped newlines to actual newlines
        |> String.replace("\\n", "\n")

      # Create LangChain messages
      messages = [
        %Message{role: "user", content: prompt_text}
      ]

      # Return a map with messages, category, and questions
      %{
        messages: messages,
        category: category,
        questions: chunk_questions
      }
    end)
  end

  @doc """
  Generates the examples section of the prompt with multiple examples.
  """
  def generate_examples_section(category_element, questions, variant \\ "b") do
    examples = Map.get(category_element, "examples", [])

    example_template =
      case variant do
        "a" -> extract_prompt_element("examples_section_a")
        "b" -> extract_prompt_element("examples_section_b")
      end

    if Enum.empty?(examples) do
      # No examples available
      ""
    else
      # Generate example sections for each example in the array
      examples_text =
        examples
        # Add an index starting from 1
        |> Enum.with_index(1)
        |> Enum.map(fn {example, index} ->
          # Find the keys that contain example_text and example_answers
          example_keys = Map.keys(example)
          text_key = Enum.find(example_keys, fn k -> String.contains?(k, "example_text") end)

          answers_key =
            Enum.find(example_keys, fn k -> String.contains?(k, "example_answers") end)

          example_text = Map.get(example, text_key, "")

          example_answers =
            if is_list(Map.get(example, answers_key, [])) do
              Map.get(example, answers_key, []) |> Enum.join("\n")
            else
              Map.get(example, answers_key, "")
            end

          # Use the template from prompt_elements.json and replace example_num with the index
          example_template
          |> String.replace("<%= @example_num %>", to_string(index))
          |> String.replace("<%= @example_text %>", example_text)
          |> String.replace("<%= @questions %>", questions)
          |> String.replace("<%= @example_answers %>", example_answers)
        end)

      Enum.join(examples_text, "\n\n")
    end
  end

  @doc """
  Generates examples section for summary prompts with their specific structure.
  """
  def generate_summary_examples_section(examples) do
    if Enum.empty?(examples) do
      ""
    else
      # Get the template from prompt_elements.json
      example_template = extract_prompt_element("examples_summary")

      examples_text =
        examples
        |> Enum.with_index(1)
        |> Enum.map(fn {example, index} ->
          # Find the keys for text and summary in the example
          example_keys = Map.keys(example)
          text_key = Enum.find(example_keys, fn k -> String.contains?(k, "example_text") end)

          summary_key =
            Enum.find(example_keys, fn k -> String.contains?(k, "example_summary") end)

          # Extract values
          example_text = Map.get(example, text_key, "")

          # Format summary points
          example_summary =
            if is_list(Map.get(example, summary_key, [])) do
              Map.get(example, summary_key, []) |> Enum.join("\n")
            else
              Map.get(example, summary_key, "")
            end

          # Use the template from prompt_elements.json
          example_template
          |> String.replace("<%= @example_num %>", to_string(index))
          |> String.replace("<%= @example_text %>", example_text)
          |> String.replace("<%= @example_summary %>", example_summary)
        end)

      Enum.join(examples_text, "\n\n")
    end
  end

  @doc """
  Generates examples section for Technische_Daten category with its specific structure.
  """
  def generate_tech_data_examples_section(examples) do
    if Enum.empty?(examples) do
      ""
    else
      # Get the template from prompt_elements.json
      example_template = extract_prompt_element("examples_compare_text_json")

      examples_text =
        examples
        |> Enum.with_index(1)
        |> Enum.map(fn {example, index} ->
          # Find the keys for text, json, and answers in the example
          example_keys = Map.keys(example)
          text_key = Enum.find(example_keys, fn k -> String.contains?(k, "example_text") end)
          json_key = Enum.find(example_keys, fn k -> String.contains?(k, "example_json") end)

          missing_key =
            Enum.find(example_keys, fn k -> String.contains?(k, "example_missing") end)

          # Extract values
          example_text = Map.get(example, text_key, "")
          example_json = Map.get(example, json_key, "")

          # Format JSON properly
          formatted_json =
            case example_json do
              json when is_map(json) or is_list(json) ->
                Jason.encode!(json, pretty: true)

              json when is_binary(json) ->
                json

              _ ->
                ""
            end

          # Format missing data
          example_missing =
            if is_list(Map.get(example, missing_key, [])) do
              Map.get(example, missing_key, []) |> Enum.join("\n")
            else
              Map.get(example, missing_key, "")
            end

          # Use the template from prompt_elements.json
          example_template
          |> String.replace("<%= @example_num %>", to_string(index))
          |> String.replace("<%= @example_text %>", example_text)
          |> String.replace("<%= @example_json %>", formatted_json)
          |> String.replace("<%= @example_missing %>", example_missing)
        end)

      Enum.join(examples_text, "\n\n")
    end
  end

  @doc """
  Counts words in a text message and estimates the token count.
  """
  def count_words_and_tokens(text) when is_binary(text) do
    # Split by whitespace to count words
    words = text |> String.split(~r/\s+/, trim: true)
    word_count = length(words)

    # For German text, implement a more accurate token estimation
    # Counts special characters, punctuation, numbers, and splitting compound words

    # Count characters that likely become separate tokens
    special_chars =
      text
      |> String.graphemes()
      |> Enum.count(&(&1 =~ ~r/[.,;:!?()[\]{}""„"—–\-\/@#$%^&*=+]|[0-9]/))

    # Estimate tokens from words (using an average factor of 1.35 for German)
    # German compound words often get split into subword tokens
    base_token_estimate =
      words
      |> Enum.map(fn word ->
        cond do
          # Single character words are typically 1 token
          String.length(word) <= 1 -> 1
          # Short words usually map to 1 token
          String.length(word) <= 4 -> 1
          # Medium words may be 1-2 tokens
          String.length(word) <= 8 -> 1.3
          # Long words generally get split into multiple tokens (common in German)
          String.length(word) > 8 -> String.length(word) / 5.0
        end
      end)
      |> Enum.sum()
      |> round()

    # Add special character tokens to the base estimate
    estimated_tokens = base_token_estimate + special_chars

    %{
      word_count: word_count,
      estimated_tokens: estimated_tokens
    }
  end

  # Handle Message structs list
  def count_words_and_tokens(messages) when is_list(messages) do
    # Extract content from each message and concatenate
    combined_text =
      messages
      |> Enum.map(fn
        %{content: content} when is_binary(content) -> content
        _ -> ""
      end)
      |> Enum.join(" ")

    # Process the combined text
    count_words_and_tokens(combined_text)
  end

  # Handle non-binary input
  def count_words_and_tokens(nil), do: %{word_count: 0, estimated_tokens: 0}
  def count_words_and_tokens(_), do: %{word_count: 0, estimated_tokens: 0}

  @doc """
  Sanitizes a filename by removing chapter/subchapter numbers.
  """
  def sanitize_filename(filename) do
    # This pattern matches chapter numbering patterns like:
    # "3.1 ", "1.2.3 ", "1. ", etc. at the beginning of the string
    # It handles formats with or without spaces after periods
    Regex.replace(~r/^(\d+\.\s*)*\d+\.?\s*/, filename, "")
  end

  @doc """
  Classifies a filename to determine which categories it belongs to.
  Returns a list of category names that match the filename exactly.
  """
  def classify_filename(filename) do
    # Sanitize the filename by removing chapter numbers at the beginning
    sanitized_filename = sanitize_filename(filename)

    # Read the categories JSON file with proper error handling
    try do
      case File.read(@filename_categories) do
        {:ok, categories_json} ->
          case Jason.decode(categories_json) do
            {:ok, categories} ->
              # Look for exact matches of the filename in each category and collect all matches
              Enum.reduce(categories, [], fn {category, filenames}, acc ->
                # Check if the sanitized filename exactly matches any filename in the list
                if sanitized_filename in filenames do
                  [category | acc]
                else
                  acc
                end
              end)

            {:error, reason} ->
              IO.puts("Error decoding JSON: #{inspect(reason)}")
              []
          end

        {:error, reason} ->
          IO.puts("Error reading categories file at #{@filename_categories}: #{inspect(reason)}")
          IO.puts("Current directory: #{File.cwd!()}")
          []
      end
    rescue
      e ->
        IO.puts("Exception while classifying filename: #{inspect(e)}")
        []
    end
  end

  @doc """
  Extracts content from the prompt elements JSON file based on the provided key.
  Returns the content if found, or nil if not found or if an error occurs.
  """
  def extract_prompt_element(key) do
    try do
      case File.read(@prompt_elements) do
        {:ok, elements_json} ->
          case Jason.decode(elements_json) do
            {:ok, elements} ->
              Map.get(elements, key)

            {:error, reason} ->
              IO.puts("Error decoding prompt elements JSON: #{inspect(reason)}")
              nil
          end

        {:error, reason} ->
          IO.puts("Error reading prompt elements file at #{@prompt_elements}: #{inspect(reason)}")
          nil
      end
    rescue
      e ->
        IO.puts("Exception while extracting prompt element: #{inspect(e)}")
        nil
    end
  end
end
