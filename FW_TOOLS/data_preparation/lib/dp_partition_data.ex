defmodule DP.PartitionData do
  @filtered_md_files "N:/Thesis/data_prepare/filtered_md_files"
  @partitioned_md_files "N:/Thesis/data_prepare/partitioned_md_files"
  @temp_path "N:/Thesis/temp"

  def partition_md_files(include_subchapters \\ false) do
    partition_md_files(@filtered_md_files, @partitioned_md_files, include_subchapters)
  end

  def partition_md_files(input, output, include_subchapters \\ true) do
    # Create the output directory if it doesn't exist
    File.mkdir_p!(output)

    # Check if input is a directory or a single file
    md_files =
      if File.dir?(input) do
        # If it's a directory, get all markdown files from it
        Path.wildcard(Path.join(input, "*.md"))
      else
        # If it's a single file with .md extension, use it
        if String.ends_with?(input, ".md") and File.exists?(input) do
          [input]
        else
          # If it's neither a directory nor a valid markdown file, return empty list
          []
        end
      end

    if md_files == [] do
      IO.puts("No markdown files found in #{input}")
      []
    else
      IO.puts("Processing #{length(md_files)} markdown file(s)")
    end

    # Process each file
    Enum.each(md_files, fn file_path ->
      # Get the file name without extension
      file_name = Path.basename(file_path, ".md")

      # Create a directory for this file
      file_dir = Path.join(output, file_name)
      File.mkdir_p!(file_dir)

      # Read the content of the file
      content = File.read!(file_path)

      # Remove ">" from the beginning of each line in the full content
      clean_content =
        content
        |> String.split(~r/\r?\n/)
        |> Enum.map(fn line -> Regex.replace(~r/^>\s*/, line, "") end)
        |> Enum.join("\n")

      # Split the content into lines (use the already cleaned content)
      lines = String.split(clean_content, ~r/\r?\n/)

      # Pre-process lines to fix specific issues before we do any chapter detection
      processed_lines = handle_special_cases(lines)

      # Extract meta info (lines until first chapter heading)
      {meta_info, remaining_lines} = extract_meta_info(processed_lines)
      meta_info_path = Path.join(file_dir, "meta_info.md")
      File.write!(meta_info_path, Enum.join(meta_info, "\n"))

      # Extract chapters and references
      {chapters, references} = extract_chapters_and_references(remaining_lines)

      # Write references to a separate file if they exist
      if references && references != "" do
        references_path = Path.join(file_dir, "references.md")
        File.write!(references_path, references)
        IO.puts("Created references file")
      end

      # Fix chapter numbering using the sequential approach
      fixed_chapters = fix_chapter_numbering(chapters)

      # Group subchapters if requested
      chapters_to_write =
        if include_subchapters do
          group_subchapters(fixed_chapters)
        else
          fixed_chapters
        end

      chapters_dir = Path.join(file_dir, "single_chapters")

      # Only create the single_chapters directory if there are chapters to write
      if chapters_to_write != [] do
        File.mkdir_p!(chapters_dir)
      end

      # Write each chapter to a separate file in the single_chapters subdirectory
      chapter_contents =
        Enum.reduce(chapters_to_write, [], fn {title, chapter_content}, acc ->
          # Create filename directly from the title
          IO.puts("Chapter title: #{title}")

          filename = "#{sanitize_filename(title)}.md"

          # Remove first line(s) from content if it contains the title to avoid duplication
          content_lines = String.split(chapter_content, ~r/\r?\n/)

          # Clean up chapter content by removing title lines
          clean_content_lines = clean_up_chapter_content(content_lines, title)

          # Join the cleaned content lines back together
          clean_chapter_content = Enum.join(clean_content_lines, "\n")

          post_processed_chapter_content = post_process_content(clean_chapter_content)
          post_processed_title = post_process_content(title)

          chapter_path = Path.join(chapters_dir, filename)

          File.write!(
            chapter_path,
            "#{post_processed_title}\n\n#{post_processed_chapter_content}"
          )

          # Add this chapter to our accumulator for the full report
          acc ++ [%{title: post_processed_title, content: post_processed_chapter_content}]
        end)

      # Now create the full_report.md with all content properly assembled
      # Start with meta info
      # Add each chapter with proper heading
      # Add references at the end if they exist
      full_report_content =
        Enum.join(meta_info, "\n") <>
          "\n\n" <>
          (chapter_contents
           |> Enum.map(fn %{title: title, content: content} ->
             "#{title}\n\n#{content}"
           end)
           |> Enum.join("\n\n")) <>
          if references && references != "" do
            "\n\n## References\n\n#{references}"
          else
            ""
          end

      # Write the complete full report
      full_report_path = Path.join(file_dir, "full_report.md")
      File.write!(full_report_path, full_report_content)

      IO.puts("Processed file: #{file_name}")
    end)

    IO.puts("Partitioning complete!")
  end

  # Handle special cases in the text content before we even attempt chapter detection
  def handle_special_cases(lines) do
    checked_single_lines =
      Enum.map(lines, fn line ->
        # Fix specific problematic text patterns
        line
        # Fix missing spaces between words
        |> fix_missing_spaces_between_words()
        # Fix other issues as needed
        |> fix_other_text_issues()
        # Remove backslash escapes from special characters
        |> remove_escaping()
        |> remove_brackets()
      end)

    checked_single_lines
    |> fix_misplaced_heading_contents()
    |> join_paragraph_lines()
  end

  def post_process_content(content) do
    content
    |> remove_surrounding_stars()
    |> remove_hashtags()
  end

  defp remove_hashtags(text) do
    # Process line by line to ensure proper handling of multi-line content
    text
    |> String.split(~r/\r?\n/)
    |> Enum.map(fn line ->
      line
      # Remove heading markers of any length (1-6 hashtags) from the beginning of lines
      |> String.replace(~r/^(\#{1,6})\s+/, "")
      # Also handle more complex markdown heading patterns with numbers and formatting
      |> String.replace(~r/^(\#{1,6})\s+(\d+(?:\.\d+)*\.?\s+)?/, "\\2")
    end)
    |> Enum.join("\n")
  end

  defp remove_brackets(line) do
    line
    # Handle specific patterns like "[Grundsätzliches:]"
    |> String.replace(~r/\[([a-zA-ZäöüÄÖÜß]*):\]/, "\\1:")
    # Replace brackets with their contents when they contain word characters
    |> String.replace(~r/\[([a-zA-ZäöüÄÖÜß][a-zA-Z0-9äöüÄÖÜß\s]*[a-zA-Z0-9äöüÄÖÜß])\]/, "\\1")
    # Also handle single alphanumeric character in brackets
    |> String.replace(~r/\[([a-zA-Z0-9äöüÄÖÜß])\]/, "\\1")

    # Keep specific patterns like "[^1]", "[1^]", etc. intact
  end

  # Remove backslash escaping from characters that shouldn't be escaped in markdown
  defp remove_escaping(line) do
    line
    # Remove escaping from hyphens in list items
    |> String.replace("\\-", "-")
    # Remove escaping from asterisks in list items
    |> String.replace("\\*", "*")
    # Remove escaping from dots in numbered lists
    |> String.replace("\\.", ".")
    # Remove escaping from parentheses
    |> String.replace("\\(", "(")
    |> String.replace("\\)", ")")
    # Remove escaping from brackets
    |> String.replace("\\[", "[")
    |> String.replace("\\]", "]")
    # Remove escaping from underscores
    |> String.replace("\\_", "_")
    # Remove escaping from hash symbols (headers)
    |> String.replace("\\#", "#")
    # Remove escaping from plus symbols (sometimes used in lists)
    |> String.replace("\\+", "+")
    # Remove escaping from greater than signs (blockquotes)
    |> String.replace("\\>", ">")
  end

  # Helper function to remove surrounding stars ("**") from text
  defp remove_surrounding_stars(text) do
    # First, handle inline bold patterns
    text
    # Handle inline bold patterns
    |> String.replace(~r/\*\*(.*?)\*\*/, "\\1")
    # Handle patterns like "(gemäß Schriftsatz vom 31.01.2019)" surrounded by ** on separate lines
    |> String.replace(~r/\n\*\*\(([^)]+)\)\*\*\n/, "\n(\\1)\n")
    # Handle patterns like "**Frage X.**" that are on their own line
    |> String.replace(~r/\n\*\*Frage (\d+)\.\*\*/, "\nFrage \\1.")
    # Handle cases like "**Frage X.**\nContent..."
    |> String.replace(~r/\*\*Frage (\d+)\.\*\*\n/, "Frage \\1.\n")
    # Handle any remaining patterns like "**Frage X.**"
    |> String.replace(~r/\*\*Frage (\d+)\.\*\*/, "Frage \\1.")
    # Handle cases with newlines like "\n**Text**\n"
    |> String.replace(~r/\n\*\*([^*\n]+)\*\*\n/, "\n\\1\n")
    # Handle cases like "**Text.**"
    |> String.replace(~r/\*\*([^*]+)\.\*\*/, "\\1.")
    |> String.replace(~r/\*\*/, "")
  end

  # Check if a line looks like part of an ASCII table or diagram
  defp is_table_line?(line) do
    trimmed = String.trim(line)

    # Check for common ASCII table patterns
    cond do
      # Lines with multiple "+" characters and "-" or "=" characters between them
      Regex.match?(~r/\+[-=]+\+/, trimmed) ->
        true

      # Lines with multiple "|" characters that look like table rows
      Regex.match?(~r/\|[^|]*\|[^|]*\|/, trimmed) ->
        true

      # Lines that are predominantly made up of formatting characters
      String.length(trimmed) > 3 &&
          Regex.match?(~r/^[+|=\-:]+$/, trimmed) ->
        true

      # Lines with a high ratio of formatting characters to total length
      String.length(trimmed) > 0 &&
          String.graphemes(trimmed)
          |> Enum.count(fn c -> c in ["+", "|", "-", "=", ":", "~"] end)
          |> Kernel./(String.length(trimmed)) > 0.4 ->
        true

      true ->
        false
    end
  end

  # Join lines that should be part of the same paragraph or fix bulleted list formatting
  defp join_paragraph_lines(lines) do
    # Process lines in groups
    {result, _} =
      Enum.reduce(lines, {[], :unknown}, fn line, {acc, context} ->
        trimmed = String.trim(line)

        cond do
          # Empty line - always keep as-is and reset context
          trimmed == "" ->
            {acc ++ [line], :unknown}

          # ASCII Table/Diagram - preserve as-is
          is_table_line?(line) ->
            {acc ++ [line], :table}

          # Line after a table - don't join with previous line
          context == :table ->
            {acc ++ [line], determine_line_context(line)}

          # Heading lines - keep as-is and set context
          is_chapter_heading(trimmed) ->
            {acc ++ [line], :heading}

          # List item start - keep as-is and set list context
          Regex.match?(~r/^\s*[-•*]\s+\S/, trimmed) ->
            {acc ++ [line], :list_item}

          # Numbered list item - keep as-is and set list context
          is_numbered_list_item?(trimmed) ->
            {acc ++ [line], :list_item}

          # Continuation of a list item - check if it's indented or aligned with list marker
          context == :list_item ->
            # Get the last line and ensure proper joining
            last_line = List.last(acc)
            last_line_trimmed = String.trim(last_line)

            # Check if last line ends with period or other punctuation
            needs_space = !Regex.match?(~r/[.,:;]\s*$/, last_line_trimmed)

            # Create properly joined line with appropriate spacing
            joined_line =
              if needs_space do
                String.trim_trailing(last_line) <> " " <> trimmed
              else
                # If last line ends with punctuation, just add a space before joining
                String.trim_trailing(last_line) <> " " <> trimmed
              end

            # Replace the last line with our newly joined line
            {List.delete_at(acc, -1) ++ [joined_line], :list_item}

          # Start of a paragraph or continuing a paragraph
          true ->
            # If we're continuing a paragraph (not after a heading, list, or empty line)
            if context == :paragraph do
              # Join with the previous line
              last_line = List.last(acc)
              last_line_with_space = String.trim_trailing(last_line) <> " "
              new_acc = List.delete_at(acc, -1) ++ [last_line_with_space <> trimmed]
              {new_acc, :paragraph}
            else
              # New paragraph starts
              {acc ++ [line], :paragraph}
            end
        end
      end)

    result
  end

  defp is_numbered_list_item?(line) do
    trimmed = String.trim(line)
    # First check if it's a date pattern - if so, it's not a list item
    if is_date_pattern(trimmed) do
      false
    else
      # If not a date, check if it matches the numbered list pattern
      Regex.match?(~r/^\s*\d+\.\s+\S/, trimmed)
    end
  end

  # Helper to determine the context of a line for paragraph joining
  defp determine_line_context(line) do
    trimmed = String.trim(line)

    cond do
      trimmed == "" -> :unknown
      is_chapter_heading(trimmed) -> :heading
      is_table_line?(line) -> :table
      is_date_pattern(trimmed) -> :paragraph
      Regex.match?(~r/^\s*[-•*]\s+\S/, trimmed) -> :list_item
      Regex.match?(~r/^\s*\d+\.\s+\S/, trimmed) -> :list_item
      true -> :paragraph
    end
  end

  # Fix missing spaces between words in common patterns
  defp fix_missing_spaces_between_words(line) do
    line
    |> String.replace("VorwortundAufgabenstellung", "Vorwort und Aufgabenstellung")
    |> String.replace("VorwortundVorgang", "Vorwort und Vorgang")
    |> String.replace(
      "Anzahl der Schadenereignisse und Obliegenheitsverletzung",
      "Anzahl der Schadenereignisse und Obliegenheitsverletzungen"
    )
    |> String.replace(
      "SchadenereignisseundObliegenheitsverletzung",
      "Schadenereignisse und Obliegenheitsverletzungen"
    )
    |> String.replace("26. und 27.04.2016", "Der 26. und 27.04.2016")
    |> String.replace("SchadenhergangundAngaben", "Schadenhergang und Angaben")
    |> String.replace(
      "BewertungsgrundlageundWertermittlung",
      "Bewertungsgrundlage und Wertermittlung"
    )
    |> String.replace("4.0 Ergebnis der Untersuchungen", "4. Ergebnis der Untersuchungen")
    |> String.replace(
      "Feststellung zur Schadenursache bzw. zur Schaden-auswirkung",
      "Feststellungen zur Schadenursache bzw. Schadenauswirkung"
    )
    |> String.replace(
      "Feststellung zur Schadenursache bzw. Schadenauswirkung",
      "Feststellungen zur Schadenursache bzw. Schadenauswirkung"
    )
    |> String.replace("Reparaturmöglichkeit und Kosten", "Reparaturmöglichkeit und -kosten")
    |> String.replace("Sachverständigen Untersuchungen", "Sachverständige Untersuchungen")
    |> String.replace(
      "Beschädigte Teile und Reparaturmöglichkeiten und Kosten",
      "Beschädigte Teile und Reparaturmöglichkeiten und -kosten"
    )
    |> String.replace(
      "Instandsetzungsmöglichkeiten und Kosten",
      "Instandsetzungsmöglichkeiten und -kosten"
    )
    |> String.replace("Reparaturmöglichkeit und Kosten", "Reparaturmöglichkeit und -kosten")
    |> String.replace("Reparaturmöglichkeit- und Kosten", "Reparaturmöglichkeit und -kosten")
    |> String.replace(
      "verschleißbedingtenReparaturaufwand",
      "verschleißbedingten Reparaturaufwand"
    )
    |> String.replace(
      "Aufteilung in schaden- und verschleißbedingten",
      "Aufteilung in schadenbedingten und verschleißbedingten"
    )
    |> String.replace(
      "Aufteilung in schaden- bzw. verschleißbedingten",
      "Aufteilung in schadenbedingten und verschleißbedingten"
    )

    # |> String.replace("", "")
    # |> String.replace("", "")
  end

  # Fix other text issues that might affect chapter detection or content quality
  defp fix_other_text_issues(line) do
    line
    # Add more specific replacements as needed
    |> String.replace("Eidesstattliche Erklaerung", "Eidesstattliche Erklärung")
    # Fix any other common OCR or formatting errors
    |> String.replace("l/ersicherung", "Versicherung")
    # fix escaping issues
    |> String.replace(~r/(\d+)\\\./, "\\1.")
    |> String.replace(~r/(\d+)\\\)/, "\\1)")
    |> String.replace(~r/([a-zA-Z])\\\)/, "\\1)")
    |> String.replace(~r/([a-zA-Z])\\\./, "\\1.")
    # remove {.underline}
    |> String.replace(~r/\{\..*?\}/, "")
    # fix dot being too close to the word (add space after period when followed by a word)
    |> String.replace(~r/([a-zA-Z])\.([a-zA-Z])/, "\\1. \\2")

    # fix
  end

  # Fix misplaced heading contents by looking for specific patterns across multiple lines
  defp fix_misplaced_heading_contents(lines) do
    # Process the lines to identify and fix misplaced heading contents
    {fixed_lines, _} =
      Enum.reduce(Enum.with_index(lines), {[], nil}, fn {line, index}, {acc_lines, skip_index} ->
        # If this line index should be skipped (already processed as part of a fix)
        if skip_index != nil && index <= skip_index do
          # Skip this line completely (don't add to accumulator)
          {acc_lines, skip_index}
        else
          # Check if the current line matches any of our patterns for split headings
          if line =~
               ~r/\*\*.*(Sachverständige Untersuchungen \/|Aufteilung in erstattungs- und nicht erstattungsfähige|Aufteilung in schadenbedingte und verschleißbedingte|Aufteilung in schadenbedingten und verschleißbedingten|Aufteilung in schadenbedingten und verschleißbedingten Reparaturaufwand\.? Erstattungsanspruch bei)\*\*\s*$/ do
            # Look ahead for continuation lines
            next_lines = Enum.slice(lines, index + 1, 3)

            # Check for empty line(s) followed by continuation text
            continuation_pattern = fn lines ->
              case lines do
                # Pattern: empty line, then continuation
                ["", continuation | _] when is_binary(continuation) ->
                  if continuation =~ ~r/^\*\*([^*]+)\*\*/ do
                    {1,
                     Regex.run(~r/^\*\*([^*]+)\*\*/, continuation, capture: :all_but_first)
                     |> hd()}
                  else
                    nil
                  end

                # Pattern: two empty lines, then continuation
                ["", "", continuation | _] when is_binary(continuation) ->
                  if continuation =~ ~r/^\*\*([^*]+)\*\*/ do
                    {2,
                     Regex.run(~r/^\*\*([^*]+)\*\*/, continuation, capture: :all_but_first)
                     |> hd()}
                  else
                    nil
                  end

                # No pattern match
                _ ->
                  nil
              end
            end

            case continuation_pattern.(next_lines) do
              {skip_count, continuation_text} ->
                # Extract the current prefix (text before the closing **)
                current_prefix =
                  Regex.run(~r/^(.*)\*\*(.*)\*\*\s*$/, line, capture: :all_but_first)

                if current_prefix do
                  # Create the fixed line by combining the prefix with continuation text
                  fixed_line =
                    "#{Enum.at(current_prefix, 0)}**#{Enum.at(current_prefix, 1)} #{String.trim(continuation_text)}**"

                  # Calculate the index to skip to (index of the continuation line)
                  continuation_index = index + skip_count + 1

                  # Add the fixed line to our accumulator, and set index to skip up through continuation line
                  {acc_lines ++ [fixed_line], continuation_index}
                else
                  {acc_lines ++ [line], nil}
                end

              nil ->
                {acc_lines ++ [line], nil}
            end
          else
            {acc_lines ++ [line], nil}
          end
        end
      end)

    fixed_lines
  end

  # Fix chapter numbering based on extracted titles
  defp fix_chapter_numbering(chapters) do
    # Process chapters in order, keeping track of the current main chapter number
    {fixed_chapters, _} =
      Enum.map_reduce(chapters, %{current_main: nil, last_num: 0}, fn {title, content}, acc ->
        # Extract chapter number from title
        chapter_number = extract_chapter_number(title)

        if chapter_number do
          # Clean up the chapter number to work with
          clean_number = String.replace(chapter_number, ~r/\.$/, "")

          # Check if it's already a subchapter (contains dots)
          if String.contains?(clean_number, ".") do
            # For existing subchapters, keep as is
            {{title, content}, acc}
          else
            # For main chapters, check if it's in sequence
            {num, _} = Integer.parse(clean_number)

            if acc.current_main != nil && num < acc.last_num do
              # Out of sequence - convert to subchapter of current main
              new_number = "#{acc.current_main}.#{num}"
              title_without_number = String.trim(String.replace(title, chapter_number, ""))
              new_title = "#{new_number} #{title_without_number}"

              IO.puts("Fixing: #{title} -> #{new_title}")
              {{new_title, content}, acc}
            else
              # In sequence - update current main chapter
              {{title, content}, %{current_main: num, last_num: num}}
            end
          end
        else
          # No chapter number, keep as is
          {{title, content}, acc}
        end
      end)

    fixed_chapters
  end

  # Process chapters with subchapter detection
  defp process_chapters_with_subchapter_detection(chapter_ranges, lines) do
    # Extract all chapters first
    all_chapters =
      Enum.map(chapter_ranges, fn {start_idx, end_idx} ->
        # Get the chapter's content lines
        chapter_lines = Enum.slice(lines, start_idx, end_idx - start_idx)

        # Extract the title (may span multiple lines)
        title = extract_multiline_title(chapter_lines)

        # Content is all lines including the first one (we'll handle duplicate title later)
        content = Enum.join(chapter_lines, "\n")

        {title, content}
      end)

    # Extract the last chapter separately to handle references
    {regular_chapters, last_chapter} =
      case all_chapters do
        [] -> {[], nil}
        chapters -> {Enum.slice(chapters, 0, length(chapters) - 1), List.last(chapters)}
      end

    # Process the last chapter to extract references
    if last_chapter do
      {title, content} = last_chapter
      content_lines = String.split(content, ~r/\r?\n/)

      # Look for references section in the last chapter
      {references_index, references_content} = extract_references_section(content_lines)

      if references_index && references_content && references_content != "" do
        # Keep the last chapter content up to the references section
        chapter_content =
          content_lines
          |> Enum.take(references_index)
          |> Enum.join("\n")

        # Return all chapters with the modified last chapter, plus references
        {regular_chapters ++ [{title, chapter_content}], references_content}
      else
        # No references section found
        {all_chapters, nil}
      end
    else
      {all_chapters, nil}
    end
  end

  # Extract meta info (content before the first chapter heading)
  defp extract_meta_info(lines) do
    # Look for the first line that might be a chapter heading
    chapter_start_index =
      Enum.find_index(lines, fn line ->
        is_chapter_heading(line)
      end)

    case chapter_start_index do
      nil ->
        # If no chapter heading found, return all content as meta info
        {lines, []}

      index ->
        # Split at the chapter heading
        {Enum.take(lines, index), Enum.drop(lines, index)}
    end
  end

  # Extract chapters from the content and separate references section
  defp extract_chapters_and_references(lines) do
    # Find all chapter heading indices, including two-line headings
    chapter_indices =
      lines
      |> Enum.with_index()
      |> Enum.filter(fn {line, index} ->
        # Check if this line by itself is a chapter heading
        is_single_line = is_chapter_heading(line)

        # Or check if this line + next line form a chapter heading
        is_multi_line =
          if index < length(lines) - 1 do
            next_line = Enum.at(lines, index + 1)
            is_two_line_heading(line, next_line)
          else
            false
          end

        is_single_line || is_multi_line
      end)
      |> Enum.map(fn {_, index} -> index end)

    # If no chapters found, return empty list and check for references
    if chapter_indices == [] do
      # Look for references section in entire document if no chapters found
      {references_index, references_content} = extract_references_section(lines)

      if references_index do
        {[], references_content}
      else
        {[], nil}
      end
    else
      # Pair each chapter start with the next chapter start (or end of document)
      chapter_ranges =
        chapter_indices
        |> Enum.zip(chapter_indices |> Enum.drop(1) |> Kernel.++([:end]))
        |> Enum.map(fn
          {start, :end} -> {start, length(lines)}
          {start, finish} -> {start, finish}
        end)

      # Process chapters with subchapter detection
      {chapters_with_content, references} =
        process_chapters_with_subchapter_detection(chapter_ranges, lines)

      {chapters_with_content, references}
    end
  end

  # Process chapters and extract references
  defp process_chapters_and_extract_references(chapter_ranges, lines) do
    # Get all chapters except the last one
    non_last_chapters = Enum.slice(chapter_ranges, 0, max(0, length(chapter_ranges) - 1))

    # Extract regular chapters
    regular_chapters =
      Enum.map(non_last_chapters, fn {start_idx, end_idx} ->
        # Get the chapter's content lines
        chapter_lines = Enum.slice(lines, start_idx, end_idx - start_idx)

        # Extract the title (may span multiple lines)
        title = extract_multiline_title(chapter_lines)

        # Content is all lines including the first one (we'll handle duplicate title later)
        content = Enum.join(chapter_lines, "\n")

        {title, content}
      end)

    # Process the last chapter separately to extract references
    if length(chapter_ranges) > 0 do
      {start_idx, end_idx} = List.last(chapter_ranges)
      last_chapter_lines = Enum.slice(lines, start_idx, end_idx - start_idx)

      # Look for references section in the last chapter
      {references_index, references_content} = extract_references_section(last_chapter_lines)

      if references_index && references_content && references_content != "" do
        # Extract title from chapter lines
        title = extract_multiline_title(last_chapter_lines)

        # Keep the last chapter content up to
        chapter_content =
          last_chapter_lines
          |> Enum.take(references_index)
          |> Enum.join("\n")

        # Return all chapters with the modified last chapter, plus references
        {regular_chapters ++ [{title, chapter_content}], references_content}
      else
        # No references section found, process last chapter normally
        title = extract_multiline_title(last_chapter_lines)
        content = Enum.join(last_chapter_lines, "\n")

        {regular_chapters ++ [{title, content}], nil}
      end
    else
      {regular_chapters, nil}
    end
  end

  # Extract the references section starting with "Kirchheim,"
  defp extract_references_section(lines) do
    # Find the line that starts with "Kirchheim,"
    kirchheim_index =
      Enum.find_index(lines, fn line ->
        String.trim(line) |> String.starts_with?("Kirchheim")
      end)

    if kirchheim_index do
      # Extract up to 10 lines after "Kirchheim," to search for Walter Sigloch
      scan_range = min(10, length(lines) - kirchheim_index - 1)
      next_lines = Enum.slice(lines, kirchheim_index + 1, scan_range)

      # Find all lines containing "Walter Sigloch" in the scan range
      sigloch_indices =
        next_lines
        |> Enum.with_index()
        |> Enum.filter(fn {line, _} ->
          String.contains?(line, "Walter Sigloch")
        end)
        |> Enum.map(fn {_, idx} -> idx end)

      case sigloch_indices do
        [] ->
          # No Walter Sigloch found, use Kirchheim line as reference marker
          references_content =
            Enum.drop(lines, kirchheim_index + 1)
            |> Enum.join("\n")

          # Return the index after Kirchheim and the references content
          {kirchheim_index + 1, references_content}

        indices when is_list(indices) ->
          # Get the index of the last occurrence of Walter Sigloch
          last_sigloch_relative_index = List.last(indices)

          # Calculate the absolute index of the last line containing Walter Sigloch
          last_sigloch_absolute_index = kirchheim_index + 1 + last_sigloch_relative_index

          # Check if there's content after the last line with Walter Sigloch
          if last_sigloch_absolute_index < length(lines) - 1 do
            # Extract content AFTER the last line containing Walter Sigloch as references
            references_content =
              Enum.drop(lines, last_sigloch_absolute_index + 1)
              |> Enum.join("\n")

            # Return the index AFTER the last line containing Walter Sigloch and the references content
            {last_sigloch_absolute_index + 1, references_content}
          else
            # No content after the last line containing Walter Sigloch
            {nil, nil}
          end
      end
    else
      {nil, nil}
    end
  end

  # Extract multiline title from chapter heading lines - improved to handle whitespace better
  defp extract_multiline_title(chapter_lines) do
    first_line = List.first(chapter_lines, "")
    second_line = if length(chapter_lines) > 1, do: Enum.at(chapter_lines, 1), else: ""

    # Check if we have a two-line heading where the title spans across both lines
    if is_two_line_heading(first_line, second_line) do
      # Join the lines to form a complete heading, with proper spacing
      first_line_cleaned = String.trim_trailing(first_line)
      second_line_cleaned = String.trim(second_line)
      combined_heading = "#{first_line_cleaned} #{second_line_cleaned}"
      extract_chapter_title(combined_heading)
    else
      # Otherwise handle as before
      if String.contains?(first_line, "**") &&
           has_closing_stars(first_line) &&
           !String.ends_with?(String.trim(first_line), "**") do
        # First line has complete title
        extract_chapter_title(first_line)
      else
        # If second line exists and contains the closing **, use first + second line
        if second_line != "" && String.contains?(second_line, "**") do
          # Combine first and second line to form the title
          combined_title = "#{first_line} #{second_line}"
          extract_chapter_title(combined_title)
        else
          # Otherwise, just use the first line
          extract_chapter_title(first_line)
        end
      end
    end
  end

  # Helper to extract the full title from a chapter heading line
  defp extract_chapter_title(line) do
    line = String.trim(line)

    # Extract number and title text
    {number, title_text} = extract_chapter_info(line)

    # Clean the title
    clean_text = clean_title(title_text)

    # Combine number and title if number exists
    if number do
      "#{number} #{clean_text}"
    else
      clean_text
    end
  end

  # Extract chapter number and title from a heading line or multi-line heading
  defp extract_chapter_info(line) do
    # Clean up the line
    clean_line =
      line
      # Remove leading ">"
      |> String.replace(~r/^>\s*/, "")
      # Replace newlines+> with space
      |> String.replace(~r/\n>\s*/, " ")

    # Try to extract number and title using different patterns
    {number, title} =
      cond do
        # Handle "1. **Title spanning multiple lines**" format
        Regex.match?(~r/^(\d+(?:\.\d+)*\.?)\s+\*\*(.+?\*\*)/, clean_line) ->
          captures =
            Regex.named_captures(
              ~r/^(?<num>\d+(?:\.\d+)*\.?)\s+\*\*(?<title>.+?\*\*)/,
              clean_line
            )

          {captures["num"], captures["title"]}

        # Handle "**1. Title spanning multiple lines**" format
        Regex.match?(~r/^\*\*\s*(\d+(?:\.\d+)*\.?)\s+(.+?\*\*)/, clean_line) ->
          captures =
            Regex.named_captures(
              ~r/^\*\*\s*(?<num>\d+(?:\.\d+)*\.?)\s+(?<title>.+?\*\*)/,
              clean_line
            )

          {captures["num"], captures["title"]}

        # Handle "**Title**" format (no number)
        Regex.match?(~r/^\*\*([^0-9].*?)\*\*/, clean_line) ->
          captures =
            Regex.named_captures(
              ~r/^\*\*(?<title>.*?)\*\*/,
              clean_line
            )

          {nil, captures["title"]}

        # Other patterns as before...
        true ->
          {nil, clean_line}
      end

    {number, title}
  end

  # Check if a line or two consecutive lines make up a chapter heading
  defp is_chapter_heading(line) do
    line = String.trim(line)

    # First check exclusions - patterns that should NOT be considered chapter headings
    cond do
      # Empty lines are not headings
      line == "" ->
        false

      # Date patterns should not be considered headings
      is_date_pattern(line) ->
        false

      # Exclude numbers without dots followed by text (like "8600 Dübendorf und Schweiz")
      Regex.match?(~r/^\d{3,}\s+\w+/, line) ->
        false

      # Exclude lines that are likely page numbers or dates rather than chapter numbers
      Regex.match?(~r/^[1-9]\d{1,3}$/, line) ->
        false

      # Exclude lines that look like addresses or locations
      Regex.match?(~r/^\d+\s+[A-Za-z]+\s+und\s+[A-Za-z]+/, line) ||
        String.contains?(line, "Straße") ||
          String.contains?(line, "Str.") ->
        false

      # Now check for valid chapter heading patterns
      # Format: "> 1. **Title**" or variations with leading >
      Regex.match?(~r/^>\s*\d+(?:\.\d+)*\.?\s+\*\*.+/, line) ->
        true

      # Format: "1. **Title**" or variations
      Regex.match?(~r/^\d+(?:\.\d+)*\.?\s+\*\*.+/, line) ->
        true

      # Format: "> **1. Title**" or variations with leading >
      Regex.match?(~r/^>\s*\*\*\s*\d+(?:\.\d+)*\.?\s+.+/, line) ->
        true

      # Format: "**1. Title**" or variations
      Regex.match?(~r/^\*\*\s*\d+(?:\.\d+)*\.?\s+.+/, line) ->
        true

      # Otherwise it's not a chapter heading
      true ->
        false
    end
  end

  # Helper function to check for date patterns that should not be considered chapter headings
  defp is_date_pattern(line) do
    # Remove any bold markers or leading special characters for checking
    clean_line =
      line
      |> String.replace(~r/^\*\*|\*\*$/, "")
      |> String.replace(~r/^>\s*/, "")
      |> String.trim()

    # German month names for the pattern matching
    month_names =
      "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"

    # Check for numeric date format (DD.MM.YYYY)
    numeric_date = Regex.match?(~r/^\d{1,2}\.\d{1,2}\.\d{2,4}/, clean_line)

    # Check for written date format (DD. Month YYYY)
    written_date = Regex.match?(~r/^\d{1,2}\.\s+(#{month_names})\s+\d{2,4}/, clean_line)

    numeric_date || written_date
  end

  # For checking if a two-line sequence forms a chapter heading
  defp is_two_line_heading(line1, line2) do
    line1 = String.trim(line1)
    line2 = String.trim(line2)

    # Check exclusions first - if line1 matches an exclusion pattern, it's not a heading
    if line1 == "" ||
         is_date_pattern(line1) ||
         Regex.match?(~r/^\d{3,}\s+\w+/, line1) ||
         Regex.match?(~r/^[1-9]\d{1,3}$/, line1) ||
         Regex.match?(~r/^\d+\s+[A-Za-z]+\s+und\s+[A-Za-z]+/, line1) ||
         String.contains?(line1, "Straße") ||
         String.contains?(line1, "Str.") do
      false
    else
      # Now do the regular two-line heading check
      starts_heading =
        Regex.match?(~r/^(?:>)?\s*(?:\*\*)?\s*\d+(?:\.\d+)*\.?\s+\*\*.*/, line1) &&
          !has_closing_stars(line1)

      completes_heading = String.contains?(line2, "**")

      starts_heading && completes_heading
    end
  end

  # Helper to check if a line has closing ** after the opening **
  defp has_closing_stars(line) do
    case String.split(line, "**", parts: 3) do
      # Has both opening and closing **
      [_before, _between, _after] -> true
      # Doesn't have both opening and closing **
      _ -> false
    end
  end

  # Clean up a title by removing formatting characters and metadata
  defp clean_title(title) do
    title
    # Remove bold markers
    |> String.replace(~r/\*\*/, "")
    # Remove brackets
    |> String.replace(~r/\[|\]/, "")
    # Remove metadata like {.underline}
    |> String.replace(~r/\{[^}]*\}/, "")
    # Remove < > symbols
    |> String.replace(~r/<|>/, "")
    # Remove trailing colon
    |> String.replace(~r/:$/, "")
    # Replace multiple consecutive whitespace with a single space
    |> String.replace(~r/\s+/, " ")
    |> String.trim()
  end

  # Format the content of a chapter
  defp format_chapter_content(chapter_lines) do
    chapter_lines
    # Since we built the list in reverse
    |> Enum.reverse()
    |> Enum.join("\n")
  end

  # Sanitize a string to be used as a filename
  defp sanitize_filename(name) do
    name
    # Replace " / " with " und "
    |> String.replace(" / ", " und ")
    # Replace tight slashes with "und"
    |> String.replace("/", " und ")
    # Replace invalid chars with underscore, but keep umlauts and other special characters
    |> String.replace(~r/[^a-zA-Z0-9_\-\. äöüÄÖÜß]/, "")
    # Replace multiple underscores with single one
    |> String.replace(~r/_+/, "_")
    # Limit length to avoid very long filenames
    |> String.slice(0, 150)
    # Trim leading/trailing underscores
    |> String.trim("_")
    # Remove multiple spaces with a single one
    |> String.replace(~r/\s+/, " ")
    # Remove leading and trailing whitespace
    |> String.trim()
  end

  # Clean up chapter content by removing title lines (both for single and multi-line titles)
  defp clean_up_chapter_content(content_lines, title) do
    # Get first and second lines for evaluation
    first_line = if length(content_lines) > 0, do: hd(content_lines), else: ""
    second_line = if length(content_lines) > 1, do: Enum.at(content_lines, 1), else: ""

    cond do
      # Case 1: First line contains chapter heading
      is_chapter_heading(first_line) ->
        # If the second line looks like a continuation of a title (contains closing ** or underline tags)
        if second_line != "" &&
             (String.contains?(second_line, "**") ||
                String.contains?(second_line, "].underline") ||
                String.contains?(second_line, "]:")) do
          # Remove both the first and second lines
          Enum.drop(content_lines, 2)
        else
          # Only remove the first line
          Enum.drop(content_lines, 1)
        end

      # Case 2: First line starts title but doesn't finish it (no closing **)
      String.contains?(first_line, "**") && !has_closing_stars(first_line) &&
          String.contains?(second_line, "**") ->
        # Remove both the first and second lines
        Enum.drop(content_lines, 2)

      # Case 3: Second line contains formatting remnants like underline tags or closing **
      second_line != "" &&
          (String.contains?(second_line, "].underline") ||
             String.contains?(second_line, "**") ||
             Regex.match?(~r/^\s*[\]\}]/, second_line)) ->
        # Remove both the first and second lines
        Enum.drop(content_lines, 2)

      # Default case: content seems fine, no removal needed
      true ->
        content_lines
    end
  end

  # Extract chapter number from a title
  defp extract_chapter_number(title) do
    # Try to find number patterns like "5." or "5.1." at the beginning of the title
    case Regex.run(~r/^\s*(\d+(?:\.\d+)*\.?)/, String.trim(title)) do
      [_, number] -> String.trim(number)
      _ -> nil
    end
  end

  # Extract the main chapter number from a subchapter number (e.g. "5.1.2" -> "5")
  defp extract_main_chapter_from_subchapter(chapter_number) do
    if String.contains?(chapter_number, ".") do
      chapter_number |> String.split(".", parts: 2) |> hd
    else
      chapter_number
    end
  end

  # Check if a chapter number is a valid continuation in the hierarchy
  defp valid_hierarchical_number?(chapter_number, state) do
    # If we don't have a previous number to compare with, consider it valid
    if state.last_number == nil do
      true
    else
      # Get the main chapter part (before first dot)
      main_prefix = extract_main_chapter_from_subchapter(chapter_number)

      # It's valid if it has the same prefix as the current main chapter
      main_prefix == state.current_main
    end
  end

  # Check if a chapter number appears to be a main chapter (no dots)
  defp is_main_chapter?(chapter_number) do
    # In addition to checking for dots, also verify it doesn't look like a page number or other non-chapter number
    # Avoid mistaking short numeric prefixes as chapter numbers if they lack proper formatting
    !String.contains?(chapter_number, ".") &&
      Regex.match?(~r/^\d+$/, String.trim(chapter_number)) &&
      (String.length(String.trim(chapter_number)) > 2 ||
         is_chapter_numbered_format?(chapter_number))
  end

  # Check if a number is properly formatted as a chapter number (e.g., "5." or "5 " followed by title)
  defp is_chapter_numbered_format?(chapter_number) do
    # Usually chapter numbers are followed by a dot or space before the title
    Regex.match?(~r/^\d+\.?\s*$/, String.trim(chapter_number))
  end

  # Helper to extract the last numeric part from a chapter number
  defp extract_last_number(number) do
    if String.contains?(number, ".") do
      # For numbers like "5.2", get the "2" part
      number
      |> String.split(".")
      |> List.last()
      |> Integer.parse()
    else
      # For simple numbers, parse directly
      Integer.parse(number)
    end
  end

  # Check if a subchapter has a valid format relative to current state
  defp valid_subchapter?(chapter_number, state) do
    prefix = extract_main_chapter_from_subchapter(chapter_number)
    result = prefix == state.current_main_num

    IO.puts("valid_subchapter?(#{chapter_number}, #{state.current_main_num}) -> #{result}")
    result
  end

  # Parse a chapter number into main and sub components
  defp parse_chapter_number(chapter_number) do
    if String.contains?(chapter_number, ".") do
      [main | rest] = String.split(chapter_number, ".")
      {main, rest}
    else
      {chapter_number, []}
    end
  end

  # Group subchapters with their main chapters based on the first integer in the title
  defp group_subchapters(chapters) do
    # First pass: group chapters by their main chapter number
    chapters_by_main =
      Enum.reduce(chapters, %{}, fn {title, content}, acc ->
        chapter_number = extract_chapter_number(title)

        if chapter_number do
          # Extract the main chapter number (before any dots)
          main_num =
            chapter_number
            # Remove trailing dot if present
            |> String.replace(~r/\.$/, "")
            # Split on first dot
            |> String.split(".", parts: 2)
            # Take the first part
            |> hd

          # Group chapters by main number
          chapter_info = %{
            title: title,
            content: content,
            number: chapter_number,
            is_main: !String.contains?(chapter_number, ".")
          }

          # Add this chapter to its group
          Map.update(acc, main_num, [chapter_info], fn existing -> existing ++ [chapter_info] end)
        else
          # Chapters without numbers are treated as standalone
          Map.put(acc, "no_number_#{:erlang.unique_integer([:positive])}", [
            %{
              title: title,
              content: content,
              number: nil,
              is_main: true
            }
          ])
        end
      end)

    # Second pass: for each main chapter number, combine content
    chapters_combined =
      Enum.map(chapters_by_main, fn {main_num, chapter_list} ->
        # Sort so main chapter comes first, then subchapters by number
        sorted_chapters =
          Enum.sort_by(chapter_list, fn chap ->
            if chap.is_main, do: "0", else: chap.number
          end)

        case sorted_chapters do
          [] ->
            # Should never happen
            nil

          [main_chapter] ->
            # Only one chapter in this group
            {main_chapter.title, main_chapter.content}

          [main_chapter | subchapters] ->
            # Combine main chapter with all subchapters
            combined_content = main_chapter.content

            # Add all subchapters content with proper headings
            combined_content =
              Enum.reduce(subchapters, combined_content, fn subchap, content_so_far ->
                # Extract the subchapter title without the number
                title_without_number =
                  String.trim(String.replace(subchap.title, subchap.number, ""))

                # Clean up subchapter content to avoid duplicate titles
                content_lines = String.split(subchap.content, ~r/\r?\n/)
                clean_content_lines = clean_up_chapter_content(content_lines, subchap.title)
                clean_subchapter_content = Enum.join(clean_content_lines, "\n")

                # Create the subheading with appropriate level but without markdown ## markers
                # We'll add the actual number and title without any ## markers
                subheading = "#{subchap.number} #{title_without_number}"

                # Combine with appropriate heading level
                content_so_far <>
                  "\n\n#{subheading}\n\n#{clean_subchapter_content}"
              end)

            # Return the combined chapter
            {main_chapter.title, combined_content}
        end
      end)

    # Filter out any nil values and convert to list
    chapters_combined
    |> Enum.filter(&(&1 != nil))
    |> Enum.sort_by(fn {title, _} ->
      num = extract_chapter_number(title)
      if num, do: String.to_integer(String.replace(num, ~r/\..*$/, "")), else: 999
    end)
  end
end
