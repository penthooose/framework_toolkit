defmodule InformationExtractor do
  @moduledoc """
  Hybrid AI system with components to extract key information from textual data for fine-tuning purposes.
  This module serves as a main controller and API for the information extraction process.
  """

  alias SE.SymbolicExtractor
  alias SSE.SubSymbolicExtractor

  # @dir_of_reports "N:/Thesis/data_prepare/partitioned_md_files"  information_extraction
  # @input_subsymbolic "N:/Thesis/data_prepare/information_extraction/sub_symbolic"
  @input_subsymbolic "N:/Thesis/data_prepare/information_extraction/sub_symbolic_revise"
  @input_symbolic "N:/Thesis/data_prepare/information_extraction/symbolic_checked_meta_data"
  @log_dir "N:/Thesis/data_prepare/information_extraction"

  @temp "N:/Thesis/temp"

  # specify files where information should be extracted or not extracted from (whitelist / blacklist)
  # use "~" to include / exclude files that have the specified text in the file name
  # use "#" to include / exclude values of a specified key, in a json file
  @include_categories "N:/Thesis/data_prepare/statistics/cluster_filename_categories.json"
  @files_for_symbolic_extraction [
    "meta_info.md",
    "#Technische_Daten",
    "~Gerätedaten"
  ]

  @files_for_subsymbolic_extraction [
    # "full_report.md",
    "extracted_meta_info.json"
  ]
  @exclude_categories "N:/Thesis/data_prepare/statistics/cluster_filename_categories.json"
  @excluded_files_for_subsymbolic_extraction [
    # "#Umfang",
    # "#Aufgaben_und_Fragestellungen",
    # "#Schadenursache_Angaben",
    # "#Technische_Daten"
    # "#Systembeschreibung",
    # "#Begutachtung",
    # "#Schäden_und_Schadenereignisse",
    # "#Wiederinstandsetzungsmöglichkeiten",
    # "#Finanzielle_Bewertung",
    # "#Zusammenfassung_und_Bewertung",
    # "#Beweisschluss",
    # "#Referenzen"
  ]

  def symbolic_extraction do
    # Ensure log directory exists
    File.mkdir_p!(@log_dir)

    # Initialize the log structure
    log_file_path = Path.join(@log_dir, "extraction_log.json")

    log_data =
      if File.exists?(log_file_path) do
        existing_log_data =
          log_file_path
          |> File.read!()
          |> Jason.decode!()

        # Keep all keys except "uncaptured_content_of_symbolic_extraction"
        # We'll add a fresh, empty version of this key
        existing_log_data
        |> Map.drop(["uncaptured_content_of_symbolic_extraction"])
        |> Map.put("uncaptured_content_of_symbolic_extraction", %{})
      else
        %{"uncaptured_content_of_symbolic_extraction" => %{}}
      end

    # Get all subfolders in the directory
    subfolders =
      @input_symbolic
      |> File.ls!()
      |> Enum.filter(fn path ->
        File.dir?(Path.join(@input_symbolic, path))
      end)

    # Process each subfolder and accumulate log data
    updated_log_data =
      Enum.reduce(subfolders, log_data, fn subfolder, acc_log ->
        try do
          # Construct full path for the subfolder
          subfolder_path = Path.join(@input_symbolic, subfolder)

          # Find all files in the subfolder recursively and filter by inclusion criteria
          all_files_in_subfolder =
            find_all_files_recursively(subfolder_path)
            |> Enum.filter(fn file_path ->
              file_name = Path.basename(file_path)
              should_include_file?(file_name)
            end)

          # Also include files matched by patterns in @files_for_symbolic_extraction
          pattern_matched_files =
            Enum.flat_map(@files_for_symbolic_extraction, fn file_pattern ->
              find_matching_files(subfolder_path, file_pattern)
            end)

          # Combine and remove duplicates
          all_files_to_process =
            (all_files_in_subfolder ++ pattern_matched_files)
            |> Enum.uniq()

          IO.inspect(all_files_to_process, label: "All files to process")

          if Enum.empty?(all_files_to_process) do
            IO.puts("Warning: No matching files found in #{subfolder_path}")
            acc_log
          else
            # Process all matching files and collect their contents and extracted info
            {all_contents, extracted_info_list} =
              Enum.map(all_files_to_process, &process_file/1)
              |> Enum.unzip()

            # Filter out nil contents and merge extracted info
            combined_content = all_contents |> Enum.reject(&is_nil/1) |> Enum.join("\n\n")
            merged_extracted_info = merge_extracted_info(extracted_info_list)

            if combined_content != "" do
              # Create output filename and path
              output_filename = "extracted_meta_info.json"
              output_path = Path.join(subfolder_path, output_filename)

              # Save the extracted information to a file
              File.write!(output_path, Jason.encode!(merged_extracted_info))

              # Detect uncaptured content using the combined content and merged info
              uncaptured_content =
                SymbolicExtractor.detect_uncaptured_content(
                  combined_content,
                  merged_extracted_info
                )

              # Add to log if uncaptured content was found
              updated_acc_log =
                if uncaptured_content do
                  # Get the existing uncaptured content map
                  uncaptured_map = acc_log["uncaptured_content_of_symbolic_extraction"]

                  # Update the map with the new subfolder entry
                  updated_map =
                    Map.put(uncaptured_map, subfolder, %{
                      "content" => uncaptured_content
                    })

                  # Update the log data
                  Map.put(acc_log, "uncaptured_content_of_symbolic_extraction", updated_map)
                else
                  acc_log
                end

              IO.puts("Processed #{subfolder} successfully")
              updated_acc_log
            else
              acc_log
            end
          end
        rescue
          e ->
            IO.puts("Error processing subfolder #{subfolder}: #{inspect(e)}")
            acc_log
        end
      end)

    # Create a sorted version of uncaptured content by extracting all entries,
    # sorting them by key, and then reconstructing the map in that order
    sorted_uncaptured_entries =
      updated_log_data["uncaptured_content_of_symbolic_extraction"]
      |> Enum.sort_by(fn {key, _value} -> key end)

    # Create a new map with updated_log_data but with the sorted uncaptured content
    final_log_data =
      Map.put(
        updated_log_data,
        "uncaptured_content_of_symbolic_extraction",
        Enum.into(sorted_uncaptured_entries, %{})
      )

    # Write the updated log data back to file
    File.write!(log_file_path, Jason.encode!(final_log_data, pretty: true))

    IO.puts("Symbolic extraction complete for all subfolders")
    IO.puts("Uncaptured content logged to #{log_file_path}")
  end

  def sub_symbolic_extraction(
        single_chapters \\ true,
        only_summary \\ true,
        write_md_file \\ false
      ) do
    # Ensure log directory exists
    File.mkdir_p!(@log_dir)

    # Initialize or check for the log file
    log_file_path = Path.join(@log_dir, "extraction_log.json")

    log_data =
      if File.exists?(log_file_path) do
        log_file_path
        |> File.read!()
        |> Jason.decode!()
      else
        %{"uncaptured_content_of_subsymbolic_extraction" => %{}}
      end

    # Get all subfolders in the directory
    subfolders =
      @input_subsymbolic
      |> File.ls!()
      |> Enum.filter(fn path ->
        File.dir?(Path.join(@input_subsymbolic, path))
      end)

    # Get total number of folders to process
    total_folders = length(subfolders)

    # Start timer for total processing
    total_start_time = :os.system_time(:millisecond)

    # Process each subfolder with progress tracking
    subfolders
    |> Enum.with_index(1)
    |> Enum.each(fn {subfolder, current_index} ->
      try do
        # Start timer for this subfolder
        subfolder_start_time = :os.system_time(:millisecond)

        # Construct full path for the subfolder
        subfolder_path = Path.join(@input_subsymbolic, subfolder)
        subfolder_base_path = Path.basename(subfolder_path)

        IO.puts(
          "\n\nProcessing subfolder: #{subfolder_base_path} (#{current_index}/#{total_folders})"
        )

        # Create files_map by combining chapter files and other files
        chapter_files_map =
          if single_chapters do
            chapters_folder_path = Path.join(subfolder_path, "single_chapters")

            if File.dir?(chapters_folder_path) do
              # Get all MD files from the single_chapters folder
              chapters_folder_path
              |> File.ls!()
              |> Enum.filter(fn file -> String.ends_with?(file, ".md") end)
              |> Enum.sort()
              |> Enum.filter(fn file -> not should_exclude_file?(file) end)
              |> Enum.reduce(%{}, fn file, acc ->
                file_path = Path.join(chapters_folder_path, file)
                content = File.read!(file_path)
                # Apply formatting to the content
                formatted_content =
                  content
                  |> format_headings()

                # |> format_markdown_text(true)

                # IO.inspect(formatted_content, label: "Formatted Content for #{file}")

                # Use the full filename as the key
                Map.put(acc, file, formatted_content)
              end)
            else
              %{}
            end
          else
            %{}
          end

        # Add files from files_for_subsymbolic_extraction
        subsymbolic_files_map =
          Enum.reduce(@files_for_subsymbolic_extraction, %{}, fn file_pattern, acc ->
            found_files = find_matching_files(subfolder_path, file_pattern)

            if !Enum.empty?(found_files) do
              # Get the first matching file
              file_path = List.first(found_files)
              file_content = File.read!(file_path)

              # Format content if it's markdown, keep as is for JSON
              formatted_content =
                if Path.extname(file_path) == ".json" do
                  file_content
                else
                  file_content
                  # |> format_markdown_text(true)
                  # |> remove_surrounding_stars()
                end

              # Use the basename without extension as the key
              file_key =
                file_path
                |> Path.basename()
                |> Path.rootname()

              Map.put(acc, file_key, formatted_content)
            else
              acc
            end
          end)

        # Merge both maps
        files_map = Map.merge(chapter_files_map, subsymbolic_files_map)

        if map_size(files_map) > 0 do
          # Call the subsymbolic extractor with all found files
          # IO.inspect(files_map, label: "Files Map for #{subfolder}")

          extraction_result =
            SubSymbolicExtractor.extract(files_map, single_chapters, only_summary)

          # Save the extraction result to extracted_summary.json
          json_output_filename = "extracted_summary.json"
          json_output_path = Path.join(subfolder_path, json_output_filename)
          File.write!(json_output_path, extraction_result)

          if write_md_file do
            # Also save a markdown version for human readability
            md_output_filename = "extracted_summary.md"
            md_output_path = Path.join(subfolder_path, md_output_filename)

            # Parse JSON back to create markdown
            case Jason.decode(extraction_result) do
              {:ok, parsed_json} ->
                markdown_content = json_to_markdown(parsed_json)
                File.write!(md_output_path, markdown_content)

              {:error, _} ->
                # If JSON parsing fails, write the raw text
                File.write!(md_output_path, extraction_result)
            end
          end

          IO.puts("Processed #{subfolder} successfully!\n")

          # Calculate and output the time taken for this subfolder with progress indicator
          subfolder_end_time = :os.system_time(:millisecond)
          subfolder_time_seconds = (subfolder_end_time - subfolder_start_time) / 1000.0

          IO.puts(
            "Time taken for #{subfolder} (#{current_index}/#{total_folders}): #{subfolder_time_seconds} seconds.\n\n\n\n\n\n"
          )
        else
          IO.puts(
            "Warning: No matching files found in #{subfolder_path} (#{current_index}/#{total_folders})"
          )
        end
      rescue
        e ->
          IO.puts(
            "Error processing subfolder #{subfolder} (#{current_index}/#{total_folders}): #{inspect(e)}"
          )
      end
    end)

    # Calculate and output the total time taken
    total_end_time = :os.system_time(:millisecond)
    total_time_seconds = (total_end_time - total_start_time) / 1000.0

    IO.puts("Sub-symbolic extraction complete for all #{total_folders} subfolders!\n")
    IO.puts("Total processing time: #{total_time_seconds} seconds")
  end

  # Helper function to check if a file should be excluded based on the exclusion patterns
  defp should_exclude_file?(filename) do
    # Load the exclude categories JSON file once
    exclude_categories_data =
      if File.exists?(@exclude_categories) do
        @exclude_categories
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    Enum.any?(@excluded_files_for_subsymbolic_extraction, fn pattern ->
      cond do
        String.starts_with?(pattern, "~") ->
          # Extract the pattern without the "~" prefix
          match_pattern = String.replace_prefix(pattern, "~", "")
          String.contains?(filename, match_pattern)

        String.starts_with?(pattern, "#") ->
          # Extract the category key without the "#" prefix
          category_key = String.replace_prefix(pattern, "#", "")

          # Get the exclusion values for this category
          exclusion_values = Map.get(exclude_categories_data, category_key, [])

          # Check if filename contains any of the exclusion values
          Enum.any?(exclusion_values, fn value ->
            String.contains?(filename, value)
          end)

        true ->
          # Exact match
          filename == pattern
      end
    end)
  end

  # Helper function to check if a file should be included based on the inclusion patterns
  defp should_include_file?(filename) do
    # Load the include categories JSON file once
    include_categories_data =
      if File.exists?(@include_categories) do
        @include_categories
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    # If no specific inclusion patterns, include no files
    if Enum.empty?(@files_for_symbolic_extraction) do
      false
    else
      Enum.any?(@files_for_symbolic_extraction, fn pattern ->
        cond do
          String.starts_with?(pattern, "~") ->
            # Extract the pattern without the "~" prefix
            match_pattern = String.replace_prefix(pattern, "~", "")
            String.contains?(filename, match_pattern)

          String.starts_with?(pattern, "#") ->
            # Extract the category key without the "#" prefix
            category_key = String.replace_prefix(pattern, "#", "")

            # Get the inclusion values for this category
            inclusion_values = Map.get(include_categories_data, category_key, [])

            # Check if filename contains any of the inclusion values
            Enum.any?(inclusion_values, fn value ->
              String.contains?(filename, value)
            end)

          true ->
            # Exact match
            filename == pattern
        end
      end)
    end
  end

  # Helper function to recursively find all files in a directory
  defp find_all_files_recursively(dir) do
    case File.ls(dir) do
      {:ok, files_and_dirs} ->
        Enum.flat_map(files_and_dirs, fn item ->
          item_path = Path.join(dir, item)

          cond do
            # If directory, recurse into it
            File.dir?(item_path) ->
              find_all_files_recursively(item_path)

            # If regular file, add to list
            File.regular?(item_path) ->
              [item_path]

            # Otherwise (e.g., symlink), skip
            true ->
              []
          end
        end)

      _ ->
        []
    end
  end

  # Helper function to convert extracted JSON to markdown for human readability
  defp json_to_markdown(json_data) when is_map(json_data) do
    json_data
    |> Enum.map(fn {filename, categories} ->
      """
      # #{filename}

      #{categories |> Enum.map(fn {category, data} -> """
        ## #{category}

        ### Questions:
        #{Enum.map_join(data["questions"] || [], "\n", fn q -> "- #{q}" end)}

        ### Responses:
        #{Enum.map_join(data["response"] || [], "\n", fn item -> "- #{item}" end)}
        """ end) |> Enum.join("\n\n")}
      """
    end)
    |> Enum.join("\n\n---\n\n")
  end

  defp json_to_markdown(json_data), do: inspect(json_data)

  # Helper function to find files matching the pattern in a directory (including nested directories)
  defp find_matching_files(dir, file_pattern) do
    cond do
      # Exact filename match
      not String.starts_with?(file_pattern, "~") and not String.starts_with?(file_pattern, "#") ->
        file_path = Path.join(dir, file_pattern)
        if File.exists?(file_path), do: [file_path], else: []

      # Category match with "#"
      String.starts_with?(file_pattern, "#") ->
        # Extract the category key without the "#" prefix
        category_key = String.replace_prefix(file_pattern, "#", "")

        # Load the include categories JSON file
        include_categories_data =
          if File.exists?(@include_categories) do
            @include_categories
            |> File.read!()
            |> Jason.decode!()
          else
            %{}
          end

        # Get the inclusion values for this category
        inclusion_values = Map.get(include_categories_data, category_key, [])

        # Find all files in the directory
        case File.ls(dir) do
          {:ok, files_and_dirs} ->
            Enum.flat_map(files_and_dirs, fn item ->
              item_path = Path.join(dir, item)

              cond do
                # If directory, skip (we don't recurse for category matches)
                File.dir?(item_path) ->
                  []

                # If file and contains any inclusion value
                File.regular?(item_path) &&
                    Enum.any?(inclusion_values, fn value -> String.contains?(item, value) end) ->
                  [item_path]

                # Otherwise skip
                true ->
                  []
              end
            end)

          _ ->
            []
        end

      # Pattern match with "~"
      true ->
        # Extract the pattern without the "~" prefix
        pattern = String.replace_prefix(file_pattern, "~", "")

        # Walk through directory recursively
        case File.ls(dir) do
          {:ok, files_and_dirs} ->
            Enum.flat_map(files_and_dirs, fn item ->
              item_path = Path.join(dir, item)

              cond do
                # If directory, recurse into it
                File.dir?(item_path) ->
                  find_matching_files(item_path, file_pattern)

                # If file and matches pattern
                File.regular?(item_path) && String.contains?(item, pattern) ->
                  [item_path]

                # Otherwise skip
                true ->
                  []
              end
            end)

          _ ->
            []
        end
    end
  end

  # Helper function to process a single file and extract information
  defp process_file(file_path) do
    if File.exists?(file_path) do
      content = File.read!(file_path)

      # Handle JSON files appropriately
      if Path.extname(file_path) == ".json" do
        case Jason.decode(content) do
          {:ok, parsed_json} ->
            # Pass the parsed JSON directly to the extractor
            {content, SymbolicExtractor.extract(parsed_json)}

          {:error, _} ->
            # If JSON parsing fails, treat as regular content
            formatted_content = format_markdown_text(content)
            {formatted_content, SymbolicExtractor.extract(formatted_content)}
        end
      else
        # For non-JSON files, apply formatting before extraction
        formatted_content = format_markdown_text(content)
        {formatted_content, SymbolicExtractor.extract(formatted_content)}
      end
    else
      {nil, %{}}
    end
  end

  # Helper function to merge multiple extracted information maps
  defp merge_extracted_info(extracted_info_list) do
    Enum.reduce(extracted_info_list, %{}, fn extracted_info, acc ->
      Map.merge(acc, extracted_info)
    end)
  end

  # Helper function to format markdown text by removing random line breaks within sentences
  # Added remove_double_newlines parameter to control whether to remove double newlines
  defp format_markdown_text(text, remove_double_newlines \\ false) do
    # Split the text by double newlines (paragraph breaks)
    paragraphs = String.split(text, ~r/\n\s*\n+/, trim: true)

    # Process each paragraph
    formatted_paragraphs =
      Enum.map(paragraphs, fn paragraph ->
        # Preserve headings, lists, and code blocks
        cond do
          # If it's a heading (starts with #)
          String.match?(paragraph, ~r/^\s*#/) ->
            paragraph

          # If it's a list item (starts with -, *, or number.)
          String.match?(paragraph, ~r/^\s*(-|\*|\d+\.)/) ->
            paragraph

          # If it's a code block (starts with ```)
          String.match?(paragraph, ~r/^\s*```/) ->
            paragraph

          # Otherwise, it's regular text - remove line breaks within it
          true ->
            paragraph
            |> String.replace(~r/\n(?!\n)/, " ")
            |> String.replace(~r/\s{2,}/, " ")
        end
      end)

    # Join the formatted paragraphs with appropriate separator
    # If remove_double_newlines is true, join with single newline, otherwise double newlines
    separator = if remove_double_newlines, do: "\n", else: "\n\n"
    Enum.join(formatted_paragraphs, separator)
  end

  # Helper function to format headings, replacing "# X.Y. Title" with "Kapitel X.Y: Title"
  defp format_headings(text) do
    String.replace(text, ~r/^\s*#\s*(\d+(?:\.\d+)*)\.\s*(.+)$/m, "Kapitel \\1: \\2")
  end
end
