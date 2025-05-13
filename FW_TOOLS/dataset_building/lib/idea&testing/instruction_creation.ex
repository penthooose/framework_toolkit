defmodule InstructionCreation do
  @moduledoc """
  Module to create the individual instructions based on the extracted summaries and report contents.
  """
  @meta_data_for_categories Path.join([
                              Path.dirname(__ENV__.file),
                              "meta_data_for_categories.json"
                            ])

  @dataset_template """
  {
    "input": "<%= @input %>",
    "output": "<%= @output %>"
  }
  """

  @base_prompt """
    [INST]
    <%= @task %>:

    Kapitel:
    "<%= @chapter_name %>"

    Kategorien:
    "<%= @categories %>"

    <%= @meta_data %>

    Zusammenfassung des Kapitels:
    "<%= @summary %>"

    <%= @category_QA %>

    <%= @previous_content %>
    [/INST]
  """

  @base_prompt_technical """
    [INST]
    <%= @task %>:

    Kapitel:
    "<%= @chapter_name %>"

    Kategorie:
    "<%= @categories %>"

    <%= @meta_data_all %>


    <%= @meta_data_missing %>

    [/INST]
  """

  @meta_data_all_prompt """
    Auflisting aller Spezifikationen:
    <%= @meta_data_objects %>
  """

  @meta_data_missing_prompt """
    Weitere Daten:
    <%= @missing_objects %>
  """

  @meta_data_prompt """
    Metadaten:
    <%= @meta_data_objects %>
  """

  @category_QA_prompt """
    Wichtige Fragen und Antworten:
    <%= @qa_input %>

  """

  @previous_content_prompt """
    Zusammenfassung vorausgehender Kapitel (für Kontext):
    <%= @previous_content %>
  """

  @tasks [
    {:create_chapter_b1,
     "Erstelle einen Kapitelinhalt für ein Gutachten über ein Medizinprodukt mit den folgenden Informationen"},
    {:create_chapter_b2,
     "Verfasse den Inhalt eines Gutachtenkapitels über ein Medizingerät basierend auf den gegebenen Daten"},
    {:create_chapter_b3,
     "Formuliere ein Kapitel für ein technisches Gutachten über ein medizintechnisches Gerät anhand dieser Informationen"},
    {:create_chapter_b4,
     "Schreibe den folgenden Gutachtenabschnitt über ein medizintechnisches Gerät mit den bereitgestellten Daten"},
    {:create_chapter_b5,
     "Erzeuge einen fachlich korrekten Gutachtentext über ein Medizinprodukt für dieses Kapitel"},
    {:create_chapter_b6,
     "Verfasse den Inhalt eines Gutachtenkapitels über ein Medizinprodukt auf der Grundlage der vorgegebenen Daten"},
    {:create_chapter_b7,
     "Verfasse den Inhalt eines Gutachtenkapitels über ein Medizinprodukt anhand der vorgegebenen Informationen"},
    {:create_technical_data_b1,
     "Erstelle für ein Gutachtenkapitel eine strukturierte Auflistung der technischen Gerätedaten zu einem medizinischen Gerät mit folgenden Spezifikationen"},
    {:create_technical_data_b2,
     "Formatiere die im Folgenden enthaltenen technischen Daten des begutachteten Medizinprodukts in übersichtlicher Auflistung"},
    {:create_technical_data_b3,
     "Fasse die technischen Spezifikationen des zu bewertenden Medizingeräts in einer klar strukturierten Auflistung der Gerätedaten zusammen"}
  ]

  # Function to create all instruction prompts for one category
  def create_instruction(
        task,
        chapter_name,
        meta_data,
        categories,
        summary,
        qa,
        content,
        previous_content \\ nil
      ) do
    # if chapter is technical data, create a technical data instruction
    if List.first(categories) == "Technische_Daten" do
      task_string = Atom.to_string(task)
      task_last_char = String.at(task_string, -1)
      task_integer = String.to_integer(task_last_char)
      # Apply modulo 3 and add 1 (for range 1-3)
      modulo_task = rem(task_integer, 3) + 1
      technical_task = String.to_atom("create_technical_data_b#{modulo_task}")

      create_technical_data_instruction(
        technical_task,
        chapter_name,
        meta_data,
        List.first(categories),
        summary,
        qa,
        content
      )
    else
      # if chapter is normal chapter, create a normal instruction
      task_tuple = Enum.find(@tasks, fn {t, _} -> t == task end)
      task_description = if task_tuple, do: elem(task_tuple, 1), else: "No description found"
      sorted_categories = Enum.sort(categories)
      merged_categories = Enum.join(sorted_categories, "+")
      categories_string = Enum.join(sorted_categories, ", ")

      # Get metadata values for all categories and merge them to avoid duplicates
      relevant_meta_data_values = get_meta_data_values(categories, meta_data)

      # Process QA data with our helper function
      qa_string = process_qa_pairs(qa, categories)

      category_QA_string =
        if qa_string != "",
          do: EEx.eval_string(@category_QA_prompt, assigns: %{qa_input: qa_string}),
          else: ""

      previous_content_string =
        if previous_content != nil,
          do:
            EEx.eval_string(@previous_content_prompt,
              assigns: %{previous_content: previous_content}
            ),
          else: ""

      meta_data_string =
        if relevant_meta_data_values != "",
          do:
            EEx.eval_string(@meta_data_prompt,
              assigns: %{meta_data_objects: relevant_meta_data_values}
            ),
          else: ""

      # Create the instruction string using the template
      input_string =
        EEx.eval_string(@base_prompt,
          assigns: %{
            task: task_description,
            chapter_name: chapter_name,
            categories: categories_string,
            meta_data: meta_data_string,
            summary: summary,
            category_QA: category_QA_string,
            previous_content: previous_content_string
          }
        )

      # Now use the template as a string, not as a map
      instruction_string =
        EEx.eval_string(@dataset_template, assigns: %{input: input_string, output: content})
        |> format_instruction_string()

      {instruction_string, merged_categories}
    end
  end

  def create_technical_data_instruction(
        task,
        chapter_name,
        meta_data,
        category,
        summary,
        qa,
        content
      ) do
    task_tuple = Enum.find(@tasks, fn {t, _} -> t == task end)
    task_description = if task_tuple, do: elem(task_tuple, 1), else: "No description found"

    formatted_meta_data = format_meta_data(meta_data)

    # Process QA data for technical data
    missing_objects = process_qa_for_technical_data(qa, category)

    meta_data_all_string =
      if formatted_meta_data != "",
        do:
          EEx.eval_string(@meta_data_all_prompt,
            assigns: %{meta_data_objects: formatted_meta_data}
          ),
        else: ""

    meta_data_missing_string =
      if missing_objects != "",
        do:
          EEx.eval_string(@meta_data_missing_prompt,
            assigns: %{missing_objects: missing_objects}
          ),
        else: ""

    # Create the instruction string using the technical template
    input_string =
      EEx.eval_string(@base_prompt_technical,
        assigns: %{
          task: task_description,
          chapter_name: chapter_name,
          categories: category,
          meta_data_all: meta_data_all_string,
          meta_data_missing: meta_data_missing_string
        }
      )

    # Create the final instruction using the dataset template
    instruction_string =
      EEx.eval_string(@dataset_template, assigns: %{input: input_string, output: content})
      |> format_instruction_string()

    # IO.inspect(instruction_string, label: "Instruction String")

    {instruction_string, category}
  end

  # Helper function to format the instruction string into a clean, well-formatted JSON
  defp format_instruction_string(instruction_string) do
    # Instead of trying to parse as JSON, work with raw string manipulation
    try do
      # Extract input section - look for content between "input": " and ", "output"
      input_pattern = ~r/"input":\s*"(.*?)(?=",\s*"output")/s
      output_pattern = ~r/"output":\s*"(.*?)(?="(?:\s*\}))/s

      input_raw =
        case Regex.run(input_pattern, instruction_string) do
          [_, captured] -> captured
          _ -> "[INST]Error extracting input[/INST]"
        end

      output_raw =
        case Regex.run(output_pattern, instruction_string) do
          [_, captured] -> captured
          _ -> "Error extracting output"
        end

      # Clean up the extracted content
      input_cleaned =
        input_raw
        |> unescape_special_chars()
        |> normalize_line_endings()
        |> trim_whitespace()

      output_cleaned =
        output_raw
        |> unescape_special_chars()
        |> normalize_line_endings()
        |> trim_whitespace()

      # Create a map that can be properly encoded as JSON
      %{
        "input" => input_cleaned,
        "output" => output_cleaned
      }
      |> Jason.encode!()
    rescue
      e ->
        IO.puts("Error in format_instruction_string: #{inspect(e)}")

        # Return a JSON string directly
        Jason.encode!(%{
          "input" => "[INST]Error processing input[/INST]",
          "output" => "Error processing output"
        })
    end
  end

  # Helper functions for string cleaning
  defp unescape_special_chars(text) do
    text
    # Unescape quotes
    |> String.replace("\\\"", "\"")
    # Convert escaped \r to actual carriage return
    |> String.replace("\\r", "\r")
    # Convert escaped \n to actual newline
    |> String.replace("\\n", "\n")
    # Convert escaped backslash to actual backslash
    |> String.replace("\\\\", "\\")
  end

  defp normalize_line_endings(text) do
    text
    # Normalize to \n line endings
    |> String.replace(~r/\r\n|\r/, "\n")
  end

  defp trim_whitespace(text) do
    text
    # Remove excessive whitespace between lines
    |> String.replace(~r/\n\s+\n/, "\n\n")
    # Remove leading whitespace
    |> String.replace(~r/^\s+/, "")
    # Remove trailing whitespace
    |> String.replace(~r/\s+$/, "")
    # Remove double spaces at start of lines
    |> String.replace(~r/\n  /, "\n")
  end

  # Function to format meta data into a pretty JSON string
  defp format_meta_data(meta_data) do
    case meta_data do
      nil ->
        ""

      map when is_map(map) and map_size(map) > 0 ->
        Jason.encode!(map, pretty: true)

      _ ->
        ""
    end
  end

  def get_meta_data_values(categories, meta_data) do
    # Check if the JSON file exists
    if File.exists?(@meta_data_for_categories) do
      # Read and parse the JSON file
      categories_mapping =
        @meta_data_for_categories
        |> File.read!()
        |> Jason.decode!()

      # Get all relevant metadata keys across all categories
      all_relevant_keys =
        categories
        |> Enum.flat_map(fn category ->
          Map.get(categories_mapping, category, [])
        end)
        |> Enum.uniq()

      # Filter meta_data to only include relevant keys
      relevant_meta_data =
        meta_data
        |> Map.take(all_relevant_keys)

      # If we have relevant metadata, format it as a JSON string
      if map_size(relevant_meta_data) > 0 do
        Jason.encode!(relevant_meta_data, pretty: true)
      else
        ""
      end
    else
      # Return empty string if file doesn't exist
      ""
    end
  end

  defp process_qa_for_technical_data(qa, category) do
    case qa do
      nil ->
        ""

      qa_map when is_map(qa_map) ->
        # Get the response for the specified category
        case Map.get(qa_map, category) do
          %{"response" => [response | _]} when is_binary(response) ->
            # Check if response is different from "N.A."
            if response != "N.A." do
              response
            else
              ""
            end

          _ ->
            ""
        end

      _ ->
        ""
    end
  end

  # Helper function to process QA pairs
  defp process_qa_pairs(qa, categories) do
    case qa do
      nil ->
        ""

      qa_map when is_map(qa_map) ->
        # Process all categories that are present in the QA map
        categories
        |> Enum.map(fn category ->
          case Map.get(qa_map, category) do
            %{"questions" => questions, "response" => responses} ->
              # Zip questions and responses
              pairs = Enum.zip(questions, responses)

              # Filter out pairs where response is only "N.A." and clean responses with "N.A." followed by text
              filtered_pairs =
                pairs
                |> Enum.filter(fn {_q, r} ->
                  # Filter out pairs where the response is exactly "Letter. N.A."
                  !Regex.match?(~r/^[A-Z]\.\s+N\.A\.\s*$/, r)
                end)
                |> Enum.map(fn {q, r} ->
                  # For responses containing "N.A." followed by text, clean it up by removing "N.A."
                  cleaned_r = Regex.replace(~r/^([A-Z]\.\s+)N\.A\.\s*/, r, "\\1")
                  {q, cleaned_r}
                end)

              # Format the questions and answers into a single string
              formatted_qa_string =
                Enum.with_index(filtered_pairs, 1)
                |> Enum.map(fn {{q, r}, index} ->
                  question = Regex.replace(~r/^[A-Z]\.\s+/, q, "Frage #{index}: ")
                  answer = Regex.replace(~r/^[A-Z]\.\s+/, r, "Antwort #{index}: ")
                  "#{question}\n#{answer}"
                end)
                |> Enum.join("\n\n")

              {category, formatted_qa_string}

            _ ->
              {category, ""}
          end
        end)
        |> Enum.filter(fn {_, qa_string} -> String.trim(qa_string) != "" end)
        |> case do
          [] -> ""
          # Just take the first category's data for now
          [{_cat, qa_string} | _] -> qa_string
        end

      _ ->
        ""
    end
  end
end
