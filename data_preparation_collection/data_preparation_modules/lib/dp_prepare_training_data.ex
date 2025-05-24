defmodule DP.PrepareTrainingData do
  @filtered_md_files "N:/Thesis/data_prepare/filtered_md_files"
  @first_stage_files "N:/Thesis/data_prepare/first_stage_files"
  @second_stage_files "N:/Thesis/data_prepare/second_stage_files"
  @prompt_file_path "N:/Thesis/data_prepare/prompt_formatting"
  @temp_path "N:/Thesis/temp"

  def prepare_format_for_prompts do
    # Find all markdown files in the prompt_file_path directory
    file_path = Path.join(@prompt_file_path, "*.md")
    files = Path.wildcard(file_path)

    case files do
      [] ->
        IO.puts("No markdown files found in #{@prompt_file_path}")
        []

      files ->
        # Process all markdown files found
        processed_contents =
          Enum.map(files, fn file ->
            filename = Path.basename(file)
            IO.puts("\nProcessing file: #{filename}")

            content = File.read!(file)

            # Apply transformations
            processed_content =
              content
              |> replace_chapter_headings()
              |> remove_stars()
              |> process_paragraphs()
              |> replace_german_quotes()
              |> remove_escaped_chars()

            # Print content with escaped newlines shown as literal "\n"
            escaped_for_display = String.replace(processed_content, "\n", "\\n")
            IO.puts("Formatted content of #{filename}:")
            IO.puts(escaped_for_display)
            IO.puts("\n" <> String.duplicate("-", 80) <> "\n")

            # Return the processed content
            processed_content
          end)

        # Return the list of all processed contents
        processed_contents
    end
  end

  def prepare_json_for_prompts do
    file_path = Path.join(@prompt_file_path, "*.json")
    files = Path.wildcard(file_path)

    case files do
      [] ->
        IO.puts("No JSON files found in #{@prompt_file_path}")
        []

      files ->
        # Process all JSON files found
        processed_contents =
          Enum.map(files, fn file ->
            filename = Path.basename(file)
            IO.puts("\nProcessing JSON file: #{filename}")

            content = File.read!(file)

            # Parse and format JSON
            processed_content =
              case Jason.decode(content) do
                {:ok, json_data} ->
                  format_json_with_newlines(json_data)

                {:error, reason} ->
                  IO.puts("Error parsing JSON in #{filename}: #{inspect(reason)}")
                  # Return original content if parsing fails
                  content
              end

            # Print content with escaped newlines shown as literal "\n"
            escaped_for_display = String.replace(processed_content, "\n", "\\n")
            IO.puts("Formatted JSON content of #{filename}:")
            IO.puts(escaped_for_display)
            IO.puts("\n" <> String.duplicate("-", 80) <> "\n")

            # Return the processed content
            processed_content
          end)

        # Return the list of all processed contents
        processed_contents
    end
  end

  # Format JSON with proper newlines and escaping
  defp format_json_with_newlines(json_data) do
    # Use Jason for proper JSON encoding with escaping
    {:ok, json_string} = Jason.encode(json_data, pretty: true)

    # Ensure we have newlines after each line and proper escaping
    json_string
    # Normalize newlines
    |> String.replace("\r\n", "\n")
    # Remove any leading/trailing whitespace
    |> String.trim()
  end

  defp replace_chapter_headings(text) do
    # Replace chapter headings and ensure a newline follows
    # Add a special marker after heading that we can detect later
    Regex.replace(~r/# (\d+)\.\s*(.*?)(\n+)/s, text, "Kapitel \\1: \\2##HEADING_MARKER##\n")
  end

  defp remove_stars(text) do
    Regex.replace(~r/\*\*(.*?)\*\*/, text, "\\1")
  end

  defp replace_german_quotes(text) do
    text
    |> String.replace("„", "'")
    |> String.replace("\"", "'")
  end

  defp process_paragraphs(text) do
    # Ensure headings are followed by paragraph breaks by replacing our special marker
    text = String.replace(text, "##HEADING_MARKER##", "\n")

    # Split text into blocks using double newlines as separators, but preserve one newline
    # by temporarily replacing double newlines with a special marker
    text = String.replace(text, "\n\n", "##PARAGRAPH_BREAK##")

    # Now remove all single newlines with appropriate spacing
    text =
      Regex.replace(~r/\n(\s*)/, text, fn _, spaces ->
        if String.length(spaces) > 1 do
          # Replace newline followed by multiple spaces with a single space
          " "
        else
          # Replace newline with a single space
          " "
        end
      end)

    # Restore paragraph breaks with a single newline
    text = String.replace(text, "##PARAGRAPH_BREAK##", "\n")

    # Fix any extra spaces that may have been created
    Regex.replace(~r/\s{2,}/, text, " ")
  end

  defp process_list_items(text) do
    # Improved list item processing to handle multi-line items
    Regex.replace(~r/- (.*?)(\n\s+)(.*?)(?=\n-|\n\n|\z)/s, text, fn _,
                                                                    item_start,
                                                                    _,
                                                                    continuation ->
      # Join the list item start with its continuation, preserving the list marker
      "- #{item_start} #{String.trim(continuation)}"
    end)
  end

  defp remove_escaped_chars(text) do
    # Remove backslashes from escaped characters like "\--"
    Regex.replace(~r/\\([-\w])/, text, "\\1")
  end
end
