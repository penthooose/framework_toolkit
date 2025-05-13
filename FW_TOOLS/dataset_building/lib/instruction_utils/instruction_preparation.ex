defmodule InstructionPreparation do
  @dataset_dir "N:/Thesis/data_prepare/datasets_ready/mixed_coherence/combined_datasets"
  @path_statistics "N:/Thesis/data_prepare/statistics"

  def filter_long_instructions(
        max_tokens \\ 3200,
        dataset_dir \\ @dataset_dir,
        instruction_stats \\ "instruction_token_lengths"
      ) do
    # Define paths
    path_statistics = @path_statistics
    dataset_dir = @dataset_dir

    # Read the instruction stats file
    stats_path = Path.join(path_statistics, "#{instruction_stats}.json")
    stats = File.read!(stats_path) |> Jason.decode!()

    # Sets to process
    sets = ["training_set", "validation_set", "test_set"]

    # Process each set
    results =
      Enum.reduce(sets, %{}, fn set, acc ->
        # Path to the JSONL file
        jsonl_path = Path.join(dataset_dir, "#{set}.jsonl")

        # Read the file content
        content = File.read!(jsonl_path)
        lines = String.split(content, "\n", trim: true)

        # Track lines to keep
        set_stats = stats[set]
        lines_to_remove = []

        # Process each line and check against stats
        {filtered_lines, removed_count} =
          Enum.reduce(Enum.with_index(lines, 1), {[], 0}, fn {line, idx}, {kept_lines, count} ->
            line_key = "Line #{idx}"

            # Check if this line should be removed based on token count
            if Map.has_key?(set_stats, line_key) do
              entry = set_stats[line_key]

              # Determine if supervised or unsupervised format
              token_count =
                cond do
                  Map.has_key?(entry, "total") ->
                    # Supervised format
                    entry["total"]["estimated_tokens"]

                  Map.has_key?(entry, "text") ->
                    # Unsupervised format
                    entry["text"]["estimated_tokens"]

                  true ->
                    0
                end

              # If token count exceeds max, don't keep this line
              if token_count > max_tokens do
                {kept_lines, count + 1}
              else
                {[line | kept_lines], count}
              end
            else
              # If entry not in stats, keep the line
              {[line | kept_lines], count}
            end
          end)

        # Write the filtered content back to file (reverse to maintain original order)
        filtered_content = Enum.reverse(filtered_lines) |> Enum.join("\n")
        File.write!(jsonl_path, filtered_content)

        # Add statistics to result
        Map.put(acc, set, %{
          total_lines: length(lines),
          removed_lines: removed_count,
          remaining_lines: length(lines) - removed_count
        })
      end)

    extract_token_length_of_instructions(dataset_dir, 3.8)

    # Return the results
    %{
      max_tokens: max_tokens,
      results: results
    }
  end

  def estimate_token_length(content, char_per_token \\ 3.6) do
    # Estimate token length based on character count (1 token ≈ 3.5 characters)
    char_count = String.length(content)
    token_estimate = Float.ceil(char_count / char_per_token, 1)
  end

  def extract_token_length_of_instructions(
        dataset_path \\ @dataset_dir,
        chars_per_token \\ 3.7
      ) do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all JSONL files in the datasets folder
    jsonl_files =
      File.ls!(dataset_path)
      |> Enum.filter(fn file -> String.ends_with?(file, ".jsonl") end)
      |> Enum.map(fn file -> Path.join(dataset_path, file) end)

    # Process each JSONL file and collect results
    results =
      jsonl_files
      |> Enum.reduce(%{}, fn file_path, acc ->
        # Extract category name from file name (remove .jsonl extension)
        category = file_path |> Path.basename() |> String.replace_suffix(".jsonl", "")

        # Detect dataset format by checking the first line
        dataset_format = detect_dataset_format(file_path)
        IO.puts("Detected format for #{category}: #{dataset_format}")

        # Read the file line by line and analyze each line
        lines =
          file_path
          |> File.stream!()
          |> Stream.with_index(1)
          |> Enum.map(fn {line, line_number} ->
            case dataset_format do
              :supervised ->
                # Handle supervised dataset format with input/output fields
                process_supervised_line(line, line_number, chars_per_token)

              :unsupervised ->
                # Handle unsupervised dataset format with just text field
                process_unsupervised_line(line, line_number, chars_per_token)

              :unknown ->
                # Try to parse line-by-line and determine format
                cond do
                  is_supervised_line?(line) ->
                    process_supervised_line(line, line_number, chars_per_token)

                  is_unsupervised_line?(line) ->
                    process_unsupervised_line(line, line_number, chars_per_token)

                  true ->
                    # Fallback to treating as raw text
                    process_raw_line(line, line_number, chars_per_token)
                end
            end
          end)
          # Sort by estimated tokens (highest first)
          |> sort_lines_by_tokens(dataset_format)
          # Format line numbers for output
          |> Enum.map(fn {line_number, stats} -> {"Line #{line_number}", stats} end)

        # Add this category to the results
        Map.put(acc, category, lines)
      end)

    # Sort categories and create the JSON manually to preserve order
    json_content =
      results
      |> Enum.sort_by(fn {category, _} -> category end)
      |> Enum.reduce("{\n", fn {category, lines}, acc ->
        # Add category opening
        category_json = acc <> "  " <> Jason.encode!(category) <> ": {\n"

        # Add each line with proper indentation
        lines_json =
          lines
          |> Enum.reduce(category_json, fn {line_name, stats}, lines_acc ->
            # Start building the line entry
            line_entry = lines_acc <> "    " <> Jason.encode!(line_name) <> ": {\n"

            # Format the stats sections based on what's available
            formatted_stats =
              if Map.has_key?(stats, "text") do
                # Unsupervised format with text field
                line_entry <>
                  "      \"text\": {\n" <>
                  "        \"chars\": " <>
                  Integer.to_string(stats["text"]["chars"]) <>
                  ",\n" <>
                  "        \"estimated_tokens\": " <>
                  Float.to_string(stats["text"]["estimated_tokens"]) <>
                  "\n" <>
                  "      }\n"
              else
                # Supervised format with input/output fields
                Enum.reduce(stats, line_entry, fn {section_name, section_stats}, section_acc ->
                  section_acc <>
                    "      " <>
                    Jason.encode!(section_name) <>
                    ": {\n" <>
                    "        \"chars\": " <>
                    Integer.to_string(section_stats["chars"]) <>
                    ",\n" <>
                    "        \"estimated_tokens\": " <>
                    Float.to_string(section_stats["estimated_tokens"]) <>
                    "\n" <>
                    "      },\n"
                end)
                |> String.replace_trailing(",\n", "\n")
              end

            # Close the line entry
            formatted_stats <> "    },\n"
          end)
          # Remove trailing comma
          |> String.replace_trailing(",\n", "\n")

        # Close the category
        lines_json <> "  },\n"
      end)
      # Remove trailing comma
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    # Write the result to the output file
    output_file = Path.join(@path_statistics, "instruction_token_lengths.json")
    File.write!(output_file, json_content)

    IO.puts("Instruction token lengths have been analyzed and saved to #{output_file}")
  end

  # Helper to detect dataset format (supervised or unsupervised)
  defp detect_dataset_format(file_path) do
    try do
      # Read the first line of the file
      first_line =
        file_path
        |> File.stream!()
        |> Enum.take(1)
        |> List.first()

      cond do
        is_supervised_line?(first_line) -> :supervised
        is_unsupervised_line?(first_line) -> :unsupervised
        true -> :unknown
      end
    rescue
      _ -> :unknown
    end
  end

  # Check if a line is in supervised format (has "input" and "output" fields)
  defp is_supervised_line?(line) do
    case Jason.decode(line) do
      {:ok, json_data} when is_map(json_data) ->
        Map.has_key?(json_data, "input") && Map.has_key?(json_data, "output")

      _ ->
        false
    end
  end

  # Check if a line is in unsupervised format (has "text" field)
  defp is_unsupervised_line?(line) do
    case Jason.decode(line) do
      {:ok, json_data} when is_map(json_data) ->
        Map.has_key?(json_data, "text")

      _ ->
        false
    end
  end

  # Process a supervised format line
  defp process_supervised_line(line, line_number, chars_per_token) do
    case Jason.decode(line) do
      {:ok, json_data} when is_map(json_data) ->
        input_text = Map.get(json_data, "input", "")
        output_text = Map.get(json_data, "output", "")

        # Calculate stats for input
        input_chars = String.length(input_text)
        input_tokens = Float.ceil(input_chars / chars_per_token, 1)

        # Calculate stats for output
        output_chars = String.length(output_text)
        output_tokens = Float.ceil(output_chars / chars_per_token, 1)

        # Calculate total
        total_chars = input_chars + output_chars
        total_tokens = input_tokens + output_tokens

        {line_number,
         %{
           "total" => %{
             "chars" => total_chars,
             "estimated_tokens" => total_tokens
           },
           "input" => %{
             "chars" => input_chars,
             "estimated_tokens" => input_tokens
           },
           "output" => %{
             "chars" => output_chars,
             "estimated_tokens" => output_tokens
           }
         }}

      _ ->
        # Fallback for parsing errors
        process_raw_line(line, line_number, chars_per_token)
    end
  end

  # Process an unsupervised format line
  defp process_unsupervised_line(line, line_number, chars_per_token) do
    case Jason.decode(line) do
      {:ok, json_data} when is_map(json_data) ->
        text = Map.get(json_data, "text", "")

        # Calculate stats
        chars = String.length(text)
        tokens = Float.ceil(chars / chars_per_token, 1)

        {line_number,
         %{
           "text" => %{
             "chars" => chars,
             "estimated_tokens" => tokens
           }
         }}

      _ ->
        # Fallback for parsing errors
        process_raw_line(line, line_number, chars_per_token)
    end
  end

  # Process a line as raw text (fallback)
  defp process_raw_line(line, line_number, chars_per_token) do
    chars = String.length(line)
    tokens = Float.ceil(chars / chars_per_token, 1)

    {line_number,
     %{
       "text" => %{
         "chars" => chars,
         "estimated_tokens" => tokens
       }
     }}
  end

  # Sort lines by token count depending on the format
  defp sort_lines_by_tokens(lines, :supervised) do
    Enum.sort_by(lines, fn {_, stats} ->
      -stats["total"]["estimated_tokens"]
    end)
  end

  defp sort_lines_by_tokens(lines, :unsupervised) do
    Enum.sort_by(lines, fn {_, stats} ->
      -stats["text"]["estimated_tokens"]
    end)
  end

  defp sort_lines_by_tokens(lines, _) do
    # For unknown format, check each line individually
    Enum.sort_by(lines, fn {_, stats} ->
      cond do
        Map.has_key?(stats, "total") -> -stats["total"]["estimated_tokens"]
        Map.has_key?(stats, "text") -> -stats["text"]["estimated_tokens"]
        true -> 0
      end
    end)
  end
end
