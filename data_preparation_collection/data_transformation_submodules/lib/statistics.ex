defmodule Statistics do
  @moduledoc """
  Module for statistical analysis of produced or extracted data for fine-tuning.
  """

  @partitioned_md_files "N:/Thesis/data_prepare/partitioned_md_files"
  @processed_files "N:/Thesis/data_prepare/processed_files"
  @information_extraction "N:/Thesis/data_prepare/information_extraction/symbolic_checked_meta_data"
  @information_extraction_revised "N:/Thesis/data_prepare/information_extraction/sub_symbolic_revise"
  @building_supervised "N:/Thesis/data_prepare/datasets_building/supervised/multiple_chapters_format4"
  @building_supervised_qa "N:/Thesis/data_prepare/datasets_building/supervised/questions_and_answers"
  @datasets_for_mdb "N:/Thesis/data_prepare/datasets_ready/unsupervised/mdb_datasets/datasets_by_category"
  @dataset_supervised "N:/Thesis/data_prepare/datasets_ready/supervised/multiple_chapters_format4/combined_datasets"
  @dataset_supervised_qa "N:/Thesis/data_prepare/datasets_ready/supervised/questions_and_answers/combined_datasets"
  @dataset_unsupervised "N:/Thesis/data_prepare/datasets_ready/unsupervised/mixed_chapters/combined_datasets"
  @dataset_coherence_us "N:/Thesis/data_prepare/datasets_ready/unsupervised_coherence/combined_datasets"
  @dataset_coherence_sp "N:/Thesis/data_prepare/datasets_ready/supervised_coherence/combined_datasets"
  @path_statistics "N:/Thesis/data_prepare/statistics"
  @temp "N:/Thesis/data_prepare/temp"

  @filename_categories "N:/Thesis/data_prepare/statistics/cluster_filename_categories.json"

  def extract_token_length_of_qa_files(reports_dir \\ @building_supervised_qa) do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Output file path
    output_file = Path.join(@path_statistics, "qa_token_lengths.json")

    # Get all directories in the reports_dir
    parent_dirs =
      File.ls!(reports_dir)
      |> Enum.map(fn dir -> Path.join(reports_dir, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each parent directory
    results =
      Enum.reduce(parent_dirs, %{}, fn parent_dir, acc ->
        parent_name = Path.basename(parent_dir)
        json_path = Path.join(parent_dir, "finetuning_layout.json")

        if File.exists?(json_path) do
          # Read and parse the JSON file
          try do
            json_content = File.read!(json_path)

            case Jason.decode(json_content) do
              {:ok, data} ->
                # Extract filenames and their combined token values
                files_with_tokens =
                  Enum.map(data, fn {filename, file_data} ->
                    token_value = Map.get(file_data, "combined_token_value", 0)
                    # Ensure token_value is a number
                    token_value =
                      case token_value do
                        value when is_number(value) -> value
                        _ -> 0
                      end

                    {filename, token_value}
                  end)
                  # Sort by token value (descending)
                  |> Enum.sort_by(fn {_, token_value} -> -token_value end)

                # Add this parent's data to the results
                Map.put(acc, parent_name, files_with_tokens)

              _ ->
                # If JSON parsing fails, skip this directory
                IO.puts("Warning: Failed to parse JSON in #{json_path}")
                acc
            end
          rescue
            e ->
              # If file reading fails, skip this directory
              IO.puts("Warning: Failed to read file #{json_path}: #{inspect(e)}")
              acc
          end
        else
          # Skip if no finetuning_layout.json exists
          acc
        end
      end)

    # Find maximum token value for each parent directory for sorting
    parent_max_tokens =
      results
      |> Enum.map(fn {parent, files_with_tokens} ->
        # Calculate the maximum token value in this parent directory
        max_token =
          files_with_tokens
          |> Enum.map(fn {_, token_value} -> token_value end)
          # Default to 0 if the list is empty
          |> Enum.max(fn -> 0 end)

        {parent, max_token}
      end)
      |> Map.new()

    # Sort parents by their highest token value (descending)
    sorted_parents =
      results
      |> Enum.sort_by(fn {parent, _} ->
        # Sort by negative max token value to get descending order
        -(parent_max_tokens[parent] || 0)
      end)

    # Manually create JSON to preserve order
    json_content =
      sorted_parents
      |> Enum.reduce("{\n", fn {parent_name, files_with_tokens}, acc ->
        # Add parent directory opening
        parent_json = acc <> "  " <> Jason.encode!(parent_name) <> ": {\n"

        # Add each file with its token value
        files_json =
          files_with_tokens
          |> Enum.reduce(parent_json, fn {filename, token_value}, files_acc ->
            files_acc <>
              "    " <> Jason.encode!(filename) <> ": " <> Float.to_string(token_value) <> ",\n"
          end)
          # Remove trailing comma
          |> String.replace_trailing(",\n", "\n")

        # Close the parent section
        files_json <> "  },\n"
      end)
      # Remove trailing comma
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    # Write results to file
    File.write!(output_file, json_content)

    IO.puts(
      "QA token lengths have been analyzed and sorted by highest token value and saved to #{output_file}"
    )
  end

  def extract_heading_only_statistics(reports_dir \\ @building_supervised) do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all parent directories in reports_dir
    parent_dirs =
      File.ls!(reports_dir)
      |> Enum.map(fn dir -> Path.join(reports_dir, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each parent directory
    results =
      Enum.reduce(parent_dirs, %{}, fn parent_dir, acc ->
        parent_name = Path.basename(parent_dir)
        statistics_file_path = Path.join(parent_dir, "extracted_statistics.json")

        if File.exists?(statistics_file_path) do
          # Read and parse the statistics file
          {:ok, statistics_data} = File.read!(statistics_file_path) |> Jason.decode()

          # Find files with type "heading_only"
          heading_only_files =
            Enum.reduce(statistics_data, [], fn {_key, file_data}, files_acc ->
              if Map.get(file_data, "type") == "heading_only" do
                # Extract relevant information
                file_info = %{
                  "filename" => Map.get(file_data, "filename"),
                  "sanitized_filename" => Map.get(file_data, "sanitized_filename"),
                  "num_token_chapter" => Map.get(file_data, "num_token_chapter")
                }

                [file_info | files_acc]
              else
                files_acc
              end
            end)

          # Only add to results if heading_only files were found
          if heading_only_files != [] do
            Map.put(acc, parent_name, heading_only_files)
          else
            acc
          end
        else
          # Skip directories without statistics file
          acc
        end
      end)

    # Sort the results map alphabetically by directory names (keys)
    sorted_results =
      results
      |> Enum.sort_by(fn {key, _value} -> key end)
      |> Map.new()

    # Write results to a JSON file
    output_file = Path.join(@path_statistics, "heading_only_statistics.json")
    File.write!(output_file, Jason.encode!(sorted_results, pretty: true))

    # Print summary
    file_count = sorted_results |> Enum.flat_map(fn {_, files} -> files end) |> length()
    dir_count = map_size(sorted_results)

    IO.puts("Found #{file_count} heading-only files across #{dir_count} directories.")
    IO.puts("Results saved to #{output_file}.")
  end

  def get_reports_with_fragmented_QA(report_dir \\ @building_supervised_qa) do
  end

  def get_reports_with_fragmented_summary do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Markers to search for in summaries
    markers = ["### BEISPIEL", "### ZUSAMMENFASSUNG", "### ANWEISUNG", "### FALL BEGINN"]

    # Get all parent directories in building_supervised
    parent_dirs =
      File.ls!(@building_supervised)
      |> Enum.map(fn dir -> Path.join(@building_supervised, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each parent directory and collect results
    results =
      Enum.reduce(parent_dirs, %{}, fn parent_dir, acc ->
        parent_name = Path.basename(parent_dir)
        summary_file_path = Path.join(parent_dir, "extracted_summary.json")
        source_chapters_path = Path.join(parent_dir, "single_chapters")

        # Skip if summary file doesn't exist or single_chapters directory doesn't exist
        if File.exists?(summary_file_path) and File.dir?(source_chapters_path) do
          # Read and parse summary file
          {:ok, summary_data} = File.read!(summary_file_path) |> Jason.decode()

          # Find files with markers in their summaries
          files_with_markers =
            Enum.reduce(summary_data, [], fn {filename, file_data}, matched_files ->
              if Map.has_key?(file_data, "summary") do
                summary_content = file_data["summary"]

                # Check if summary contains any of our markers
                if Enum.any?(markers, &String.contains?(summary_content, &1)) do
                  # Check if the file actually exists in single_chapters
                  source_file_path = Path.join(source_chapters_path, filename)

                  if File.exists?(source_file_path) do
                    # Read file content and check word count
                    file_content = File.read!(source_file_path)
                    word_count = file_content |> String.split(~r/\s+/, trim: true) |> length()

                    # Only include files with at least 13 words
                    if word_count >= 13 do
                      [filename | matched_files]
                    else
                      matched_files
                    end
                  else
                    matched_files
                  end
                else
                  matched_files
                end
              else
                matched_files
              end
            end)

          # If we found files with markers, create temp directory and copy files
          if files_with_markers != [] do
            # Create destination directory structure
            dest_dir = Path.join(@information_extraction_revised, parent_name)
            dest_chapters_dir = Path.join(dest_dir, "single_chapters")
            File.mkdir_p!(dest_chapters_dir)

            # Copy each file with markers
            Enum.each(files_with_markers, fn filename ->
              source_file = Path.join(source_chapters_path, filename)
              dest_file = Path.join(dest_chapters_dir, filename)
              File.copy!(source_file, dest_file)
            end)

            # Add to results
            Map.put(acc, parent_name, files_with_markers)
          else
            acc
          end
        else
          acc
        end
      end)

    # Write results to JSON file
    output_file = Path.join(@path_statistics, "revise_summary_creation.json")
    File.write!(output_file, Jason.encode!(results, pretty: true))

    # Print summary
    file_count = results |> Enum.flat_map(fn {_, files} -> files end) |> length()
    dir_count = map_size(results)

    IO.puts(
      "Found #{file_count} files with fragmented summaries across #{dir_count} directories."
    )

    IO.puts("Results saved to #{output_file}.")
    IO.puts("Files copied to temporary directories under #{@information_extraction_revised}.")
  end

  def get_instructions_without_summary do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Path to the output file
    output_file = Path.join(@path_statistics, "supervised_finetuning_statistics.json")

    # Get all JSONL files in the dataset_supervised folder
    jsonl_files =
      File.ls!(@dataset_supervised)
      |> Enum.filter(fn file -> String.ends_with?(file, ".jsonl") end)
      |> Enum.map(fn file -> Path.join(@dataset_supervised, file) end)

    # Initialize counters
    count_contained_summaries = 0
    count_not_contained_summaries = 0

    # Process each JSONL file
    {count_contained_summaries, count_not_contained_summaries} =
      Enum.reduce(
        jsonl_files,
        {count_contained_summaries, count_not_contained_summaries},
        fn file_path, {contained, not_contained} ->
          # Read the file line by line
          file_path
          |> File.stream!()
          |> Enum.reduce({contained, not_contained}, fn line, {c, nc} ->
            # Parse the JSON line
            case Jason.decode(line) do
              {:ok, json_data} when is_map(json_data) ->
                # Check if input or output contains the target string
                input_text = Map.get(json_data, "input", "")
                output_text = Map.get(json_data, "output", "")

                if String.contains?(input_text, "[KAPITELZUSAMMENFASSUNG]") ||
                     String.contains?(output_text, "[KAPITELZUSAMMENFASSUNG]") do
                  {c + 1, nc}
                else
                  {c, nc + 1}
                end

              _ ->
                # If JSON parsing fails, count as not containing
                {c, nc + 1}
            end
          end)
        end
      )

    # Read existing JSON file if it exists, otherwise start with empty map
    existing_data =
      if File.exists?(output_file) do
        case File.read!(output_file) |> Jason.decode() do
          {:ok, data} -> data
          _ -> %{}
        end
      else
        %{}
      end

    # Update with new data
    updated_data =
      Map.merge(existing_data, %{
        "count_contained_summaries" => count_contained_summaries,
        "count_not_contained_summaries" => count_not_contained_summaries
      })

    # Write the updated data back to the file
    File.write!(output_file, Jason.encode!(updated_data, pretty: true))

    IO.puts("Summary statistics have been updated in #{output_file}")
    IO.puts("Lines with [KAPITELZUSAMMENFASSUNG]: #{count_contained_summaries}")
    IO.puts("Lines without [KAPITELZUSAMMENFASSUNG]: #{count_not_contained_summaries}")
  end

  def extract_supervised_finetuning_statistics(path_reports_dir \\ @building_supervised) do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all subdirectories in the path_reports_dir
    subdirectories =
      File.ls!(path_reports_dir)
      |> Enum.map(fn dir -> Path.join(path_reports_dir, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Initialize category counter
    category_counts = %{
      "overlong" => 0,
      "only_chapters" => 0,
      "only_summaries" => 0,
      "single" => 0
    }

    # Process each subdirectory
    category_counts =
      Enum.reduce(subdirectories, category_counts, fn dir, acc ->
        # Path to finetuning_layout.json in this directory
        finetuning_layout_path = Path.join(dir, "finetuning_layout.json")

        if File.exists?(finetuning_layout_path) do
          # Read and parse the finetuning_layout.json file
          {:ok, layout_data} = File.read!(finetuning_layout_path) |> Jason.decode()

          # Count each category occurrence in this file
          Enum.reduce(layout_data, acc, fn {_filename, chapter_data}, inner_acc ->
            if Map.has_key?(chapter_data, "category") do
              category = chapter_data["category"]
              # Update counter for this category if it's one we're tracking
              if Map.has_key?(inner_acc, category) do
                Map.update!(inner_acc, category, &(&1 + 1))
              else
                inner_acc
              end
            else
              inner_acc
            end
          end)
        else
          # If file doesn't exist, return accumulator unchanged
          acc
        end
      end)

    # Sort categories by count in descending order
    sorted_categories =
      category_counts
      |> Enum.sort_by(fn {_key, count} -> -count end)
      |> Map.new()

    # Create formatted JSON string
    json_content = Jason.encode!(sorted_categories, pretty: true)

    # Write results to output file
    output_file = Path.join(@path_statistics, "supervised_finetuning_statistics.json")
    File.write!(output_file, json_content)

    IO.puts("Supervised finetuning categories have been extracted to #{output_file}")
  end

  def extract_statistics_per_chapter(path_reports_dir \\ @information_extraction) do
    # Load categories from cluster_filename_categories.json
    categories_file_path = @filename_categories
    {:ok, categories_json} = File.read!(categories_file_path) |> Jason.decode()

    # Extract technical data filenames
    technical_data_filenames = Map.get(categories_json, "Technische_Daten", [])

    # Get all parent directories
    parent_dirs =
      File.ls!(path_reports_dir)
      |> Enum.map(fn dir -> Path.join(path_reports_dir, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each parent directory
    Enum.each(parent_dirs, fn parent_dir ->
      parent_name = Path.basename(parent_dir)
      IO.puts("Processing directory: #{parent_name}")

      # Path to single_chapters inside parent directory
      single_chapters_path = Path.join(parent_dir, "single_chapters")

      # Path to extracted_summary.json
      summary_file_path = Path.join(parent_dir, "extracted_summary.json")

      # Check if both required paths exist
      if File.dir?(single_chapters_path) and File.exists?(summary_file_path) do
        # Read and parse summary file
        {:ok, summary_content} = File.read(summary_file_path)
        summary_data = Jason.decode!(summary_content)

        # Get all MD files in single_chapters directory
        md_files =
          File.ls!(single_chapters_path)
          |> Enum.filter(fn file -> String.ends_with?(file, ".md") end)

        # Create a map of chapter numbers to filenames using the same logic as create_map_of_filenames
        files_with_chapter_nums =
          Enum.map(md_files, fn filename ->
            case extract_chapter_number(filename) do
              {:ok, chapter_num} -> {chapter_num, filename}
              # For files without proper chapter numbers
              :error -> {999, filename}
            end
          end)

        # Sort by the extracted chapter numbers
        sorted_files =
          files_with_chapter_nums
          |> Enum.sort_by(fn {chapter_num, _} -> chapter_num end)
          |> Enum.map(fn {_, filename} -> filename end)

        # Create map with sequential integer keys
        files_map =
          sorted_files
          |> Enum.with_index(1)
          |> Enum.map(fn {filename, index} -> {index, filename} end)
          |> Map.new()

        # Process each MD file in order and collect statistics
        chapter_stats =
          Enum.reduce(files_map, %{}, fn {index, md_file}, acc ->
            # Process the chapter content
            chapter_path = Path.join(single_chapters_path, md_file)
            chapter_content = File.read!(chapter_path)

            # Split content into lines and exclude the first 2 lines before estimating token length
            chapter_content_without_first_lines =
              chapter_content
              # Split into max 3 parts
              |> String.split("\n", parts: 3)
              |> case do
                # If there are at least 3 parts, take only the rest
                [_, _, rest] -> rest
                # If only 2 parts, take the 2nd part
                [_, rest] -> rest
                # If only 1 part, return empty string
                [single] -> ""
                # If empty, return empty string
                [] -> ""
              end

            chapter_tokens = estimate_token_length(chapter_content_without_first_lines)

            # Extract chapter number from filename if possible
            chapter_num =
              case extract_chapter_number(md_file) do
                {:ok, num} -> num
                :error -> nil
              end

            # Get sanitized filename (without chapter numbers and .md extension)
            sanitized_filename =
              md_file
              |> sanitize_filename()
              |> String.replace_suffix(".md", "")

            # Initialize the statistics for this file
            file_stats = %{
              "filename" => md_file,
              "sanitized_filename" => sanitized_filename,
              "num_token_chapter" => chapter_tokens,
              "num_token_summary" => 0,
              "num_token_meta_data" => 0,
              "sanitized_chapter_num" => chapter_num
            }

            # Check if this file exists in summary data
            file_stats =
              if Map.has_key?(summary_data, md_file) do
                md_summary_data = summary_data[md_file]

                # Process summary if it exists (check for "summary" key - case sensitive)
                summary_tokens =
                  if Map.has_key?(md_summary_data, "summary") do
                    estimate_token_length(md_summary_data["summary"])
                  else
                    0
                  end

                # Process metadata if it exists
                meta_data_tokens =
                  if Map.has_key?(md_summary_data, "included_meta_data") do
                    meta_data_json = Jason.encode!(md_summary_data["included_meta_data"])
                    estimate_token_length(meta_data_json)
                  else
                    0
                  end

                # Update the file statistics
                %{
                  file_stats
                  | "num_token_summary" => summary_tokens,
                    "num_token_meta_data" => meta_data_tokens
                }
              else
                file_stats
              end

            # Calculate the total token count
            total_tokens =
              file_stats["num_token_chapter"] +
                file_stats["num_token_summary"] +
                file_stats["num_token_meta_data"]

            # Add the total token count to file stats
            file_stats = Map.put(file_stats, "num_token_total", total_tokens)

            # Determine the chapter type
            file_stats =
              cond do
                # Check if the filename matches any entry in technical_data_filenames
                Enum.any?(technical_data_filenames, fn technical_filename ->
                  sanitized_md_file = sanitize_filename(md_file)
                  sanitized_technical = sanitize_filename(technical_filename)
                  String.contains?(sanitized_md_file, sanitized_technical)
                end) ->
                  Map.put(file_stats, "type", "technical")

                # Check if the chapter has fewer than 50 tokens
                file_stats["num_token_chapter"] < 10 ->
                  Map.put(file_stats, "type", "heading_only")

                # Default case
                true ->
                  Map.put(file_stats, "type", "regular_chapter")
              end

            # Add this file's statistics to the accumulator using the sequence number as key
            Map.put(acc, Integer.to_string(index), file_stats)
          end)

        # Sort the keys numerically before saving to JSON
        # First convert to list, sort by numeric keys, then create ordered JSON
        stats_json =
          chapter_stats
          |> Enum.sort_by(fn {key, value} -> String.to_integer(key) end)
          |> Enum.reduce(%{}, fn {key, value}, acc -> Map.put(acc, key, value) end)
          |> Jason.encode!(pretty: true)

        # Save the statistics to a JSON file in the parent directory
        stats_file_path = Path.join(parent_dir, "extracted_statistics.json")
        File.write!(stats_file_path, stats_json)

        IO.puts("Created statistics file: #{stats_file_path}")
      else
        IO.puts(
          "Skipping directory #{parent_name}: missing single_chapters directory or extracted_summary.json"
        )
      end
    end)

    IO.puts("Statistics extraction complete.")
  end

  # Helper function to extract chapter number from filename (supports multi-level numbering like "5.1.2" or "1.2.3.1")
  defp extract_chapter_number(filename) do
    # Match patterns like "1. Something", "4.2 Something", "5.1.2 Something" etc.
    case Regex.run(~r/^(\d+(?:\.\d+)*)/, filename) do
      [_, num_str] ->
        # Convert to numeric value for proper sorting
        parts = String.split(num_str, ".")

        # First handle the main chapter number
        main = String.to_integer(hd(parts))

        # Process the subchapter parts (if any)
        if length(parts) > 1 do
          # Convert each subchapter part to a fractional value based on position
          # e.g., "5.1.2" -> 5 + 0.1 + 0.002
          decimal_value =
            parts
            # Skip the main number
            |> tl()
            |> Enum.with_index()
            |> Enum.reduce(0, fn {sub_part, idx}, acc ->
              # Calculate decimal place (0.1, 0.01, 0.001, etc.)
              decimal_place = :math.pow(10, -(idx + 1))
              # Add current sub-part value
              acc + String.to_integer(sub_part) * decimal_place
            end)

          {:ok, main + decimal_value}
        else
          # For simple chapter numbers like "5."
          {:ok, main}
        end

      nil ->
        :error
    end
  end

  def estimate_token_length(content) do
    # Estimate token length based on character count (1 token ≈ 3.5 characters)
    char_count = String.length(content)
    token_estimate = Float.ceil(char_count / 3.5, 1)
  end

  def get_report_names_without_trained_full_report do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Define file paths
    input_file = Path.join(@path_statistics, "chapter_token_lengths.json")
    output_file = Path.join(@path_statistics, "report_names_without_trained_full_report.json")

    # Read and parse the chapter_token_lengths.json file
    {:ok, json_content} = File.read(input_file)
    chapter_data = Jason.decode!(json_content)

    # Filter out reports that contain full_report.md
    reports_without_full_report =
      chapter_data
      |> Enum.filter(fn {_report_name, chapters} ->
        not Map.has_key?(chapters, "full_report.md")
      end)
      |> Enum.into(%{})

    # Find the max token count for each report for sorting
    reports_with_max_tokens =
      reports_without_full_report
      |> Enum.map(fn {report_name, chapters} ->
        # Find the highest token count in this report's chapters
        max_token = chapters |> Map.values() |> Enum.max(fn -> 0 end)
        {report_name, max_token}
      end)
      |> Map.new()

    # Sort reports by their highest token count (descending)
    sorted_reports =
      reports_without_full_report
      |> Enum.sort_by(fn {report_name, _} ->
        -(reports_with_max_tokens[report_name] || 0)
      end)

    # Format the JSON content with sorted structure
    json_content =
      sorted_reports
      |> Enum.reduce("{\n", fn {report_name, chapters}, acc ->
        # Sort chapters by token count (descending)
        sorted_chapters =
          chapters
          |> Enum.sort_by(fn {_, token_count} -> -token_count end)

        # Format report entry with proper indentation
        report_json = acc <> "  " <> Jason.encode!(report_name) <> ": {\n"

        # Add each chapter with token count
        chapters_json =
          sorted_chapters
          |> Enum.reduce(report_json, fn {chapter_name, token_count}, chapters_acc ->
            chapters_acc <>
              "    " <>
              Jason.encode!(chapter_name) <>
              ": " <>
              Float.to_string(token_count) <>
              ",\n"
          end)
          # Remove trailing comma
          |> String.replace_trailing(",\n", "\n")

        # Close the report section
        chapters_json <> "  },\n"
      end)
      # Remove trailing comma
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    # Write the result to the output file
    File.write!(output_file, json_content)

    # Count the number of reports without full_report.md
    report_count = map_size(reports_without_full_report)

    IO.puts(
      "#{report_count} reports without trained full report have been saved to #{output_file}"
    )
  end

  def calculate_token_length_for_chapters do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all parent folders in partitioned_md_files
    parent_folders =
      File.ls!(@partitioned_md_files)
      |> Enum.map(fn dir -> Path.join(@partitioned_md_files, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each parent folder
    parent_data =
      Enum.reduce(parent_folders, %{}, fn parent_folder, acc ->
        parent_name = Path.basename(parent_folder)
        files_data = %{}

        # Process full_report.md file
        full_report_path = Path.join(parent_folder, "full_report.md")

        files_data =
          if File.exists?(full_report_path) do
            content = File.read!(full_report_path)
            char_count = String.length(content)
            token_estimate = Float.ceil(char_count / 3.5, 1)

            Map.put(files_data, "full_report.md", token_estimate)
          else
            files_data
          end

        # Process files in single_chapters folder
        single_chapters_path = Path.join(parent_folder, "single_chapters")

        files_data =
          if File.dir?(single_chapters_path) do
            File.ls!(single_chapters_path)
            |> Enum.reduce(files_data, fn filename, folder_acc ->
              file_path = Path.join(single_chapters_path, filename)
              content = File.read!(file_path)
              char_count = String.length(content)
              token_estimate = Float.ceil(char_count / 3.5, 1)

              Map.put(folder_acc, filename, token_estimate)
            end)
          else
            files_data
          end

        # Only add parent folder if it has any files
        if map_size(files_data) > 0 do
          Map.put(acc, parent_name, files_data)
        else
          acc
        end
      end)

    # Find max token count for each parent folder for sorting
    parent_max_tokens =
      parent_data
      |> Enum.map(fn {parent, files} ->
        max_token = files |> Map.values() |> Enum.max(fn -> 0 end)
        {parent, max_token}
      end)
      |> Map.new()

    # Sort parent folders by their highest token count (descending)
    sorted_parents =
      parent_data
      |> Enum.sort_by(fn {parent, _} ->
        -(parent_max_tokens[parent] || 0)
      end)

    # Create JSON content with parents sorted by max token count
    # and files within each parent sorted by token count (descending)
    json_content =
      sorted_parents
      |> Enum.reduce("{\n", fn {parent, files}, acc ->
        # Sort files by token count (descending)
        sorted_files =
          files
          |> Enum.sort_by(fn {_, token_count} -> -token_count end)

        # Format parent entry
        parent_json = acc <> "  " <> Jason.encode!(parent) <> ": {\n"

        # Add each file with token count
        files_json =
          sorted_files
          |> Enum.reduce(parent_json, fn {filename, token_count}, files_acc ->
            files_acc <>
              "    " <>
              Jason.encode!(filename) <>
              ": " <>
              Float.to_string(token_count) <>
              ",\n"
          end)
          # Remove trailing comma
          |> String.replace_trailing(",\n", "\n")

        # Close the parent section
        files_json <> "  },\n"
      end)
      # Remove trailing comma
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    # Write the result to the output file
    output_file = Path.join(@path_statistics, "chapter_token_lengths.json")
    File.write!(output_file, json_content)

    IO.puts("Chapter token lengths have been analyzed and saved to #{output_file}")
  end

  def extract_error_in_processing do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all subfolders in processed_files
    subfolders =
      File.ls!(@information_extraction)
      |> Enum.map(fn dir -> Path.join(@information_extraction, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each subfolder
    errors_by_subfolder =
      Enum.reduce(subfolders, %{}, fn subfolder, acc ->
        subfolder_basename = Path.basename(subfolder)
        summary_file = Path.join(subfolder, "extracted_summary.json")

        if File.exists?(summary_file) do
          # Read and parse the summary file
          {:ok, summary_data} = File.read!(summary_file) |> Jason.decode()

          # Find errors recursively in the entire JSON structure
          files_with_errors = find_errors_in_json(summary_data)

          # Only add to result if there are any errors
          if files_with_errors != [] do
            Map.put(acc, subfolder_basename, files_with_errors)
          else
            acc
          end
        else
          # If no summary file exists, skip this subfolder
          acc
        end
      end)

    # Sort the result map by keys and create a formatted JSON string
    json_content =
      errors_by_subfolder
      |> Enum.sort_by(fn {key, _} -> key end)
      |> Enum.reduce("{\n", fn {key, values}, acc ->
        sorted_values = Enum.sort(values)
        acc <> "  " <> Jason.encode!(key) <> ": " <> Jason.encode!(sorted_values) <> ",\n"
      end)
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    # Write the result to a JSON file
    output_file = Path.join(@path_statistics, "errors_during_subsymbolic_processing.json")
    File.write!(output_file, json_content)

    IO.puts("Errors during processing have been extracted to #{output_file}")
  end

  # Helper function to recursively find "ERROR IN PROCESSING" in JSON structure
  defp find_errors_in_json(json_data) do
    Enum.flat_map(json_data, fn {filename, file_content} ->
      error_paths = find_error_in_structure(file_content, [])

      if error_paths != [] do
        # Format the error details with the paths to error locations
        error_details =
          error_paths
          |> Enum.map(fn path -> "#{filename} (at #{Enum.join(path, ".")})" end)
          |> Enum.uniq()

        error_details
      else
        []
      end
    end)
  end

  # Recursively search for errors in nested structures
  defp find_error_in_structure(data, path) when is_map(data) do
    # Search through map entries
    Enum.flat_map(data, fn {key, value} ->
      current_path = path ++ [key]
      find_error_in_structure(value, current_path)
    end)
  end

  defp find_error_in_structure(data, path) when is_list(data) do
    # Search through list items with index
    data
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} ->
      current_path = path ++ ["[#{index}]"]
      find_error_in_structure(value, current_path)
    end)
  end

  defp find_error_in_structure(data, path) when is_binary(data) do
    # Check if current string contains the error message
    if String.contains?(data, "ERROR IN PROCESSING") do
      [path]
    else
      []
    end
  end

  defp find_error_in_structure(_data, _path) do
    # For other types (numbers, booleans, nil) - no errors
    []
  end

  def extract_token_length_of_instructions(
        dataset_path \\ @dataset_unsupervised,
        chars_per_token \\ 3.8
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

  def extract_santizied_chapter_filenames do
    # First run extract_single_chapter_filenames to ensure the file exists
    extract_single_chapter_filenames()

    # Read the JSON file
    input_file = Path.join(@path_statistics, "single_chapter_filenames.json")
    {:ok, json_data} = File.read!(input_file) |> Jason.decode()

    # Extract all keys and sanitize them by removing chapter numbers
    sanitized_filenames =
      json_data
      |> Map.keys()
      |> Enum.map(fn filename ->
        # Improved regex to properly remove all chapter numbering patterns
        # This will match patterns like "1. ", "5.2 ", "10. ", etc. at the beginning of the filename
        sanitized = Regex.replace(~r/^(\d+\.)+\s+|^\d+\.\s+/, filename, "")
        sanitized
      end)

    # Count occurrences of each sanitized filename
    filename_counts =
      sanitized_filenames
      |> Enum.reduce(%{}, fn sanitized, acc ->
        Map.update(acc, sanitized, 1, &(&1 + 1))
      end)

    # Sort by values (occurrence count) in descending order and create a JSON string manually
    json_content =
      filename_counts
      # Sort by count in descending order
      |> Enum.sort_by(fn {_key, count} -> -count end)
      |> Enum.reduce("{\n", fn {key, count}, acc ->
        acc <> "  " <> Jason.encode!(key) <> ": " <> Integer.to_string(count) <> ",\n"
      end)
      # Remove the trailing comma and close the JSON object
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    output_file = Path.join(@path_statistics, "sanitized_single_chapter_filenames.json")
    File.write!(output_file, json_content)

    IO.puts("Sanitized chapter filenames have been extracted to #{output_file}")
  end

  def extract_single_chapter_filenames do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all parent folders in partitioned_md_files
    parent_folders =
      File.ls!(@partitioned_md_files)
      |> Enum.map(fn dir -> Path.join(@partitioned_md_files, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Create a map of filename => list of parent directories
    filename_map =
      parent_folders
      |> Enum.reduce(%{}, fn parent_folder, acc ->
        single_chapters_path = Path.join(parent_folder, "single_chapters")

        if File.dir?(single_chapters_path) do
          parent_name = Path.basename(parent_folder)

          File.ls!(single_chapters_path)
          |> Enum.reduce(acc, fn filename, folder_acc ->
            # Update the map to add this parent folder to the list for this filename
            Map.update(folder_acc, filename, [parent_name], fn existing ->
              [parent_name | existing]
            end)
          end)
        else
          acc
        end
      end)

    # Sort the keys alphabetically and create a JSON string manually to ensure order is preserved
    # Also sort the values (lists of parent directories) for each key
    json_content =
      filename_map
      # Sort the values alphabetically
      |> Enum.map(fn {key, value} -> {key, Enum.sort(value)} end)
      # Sort by keys
      |> Enum.sort_by(fn {key, _value} -> key end)
      |> Enum.reduce("{\n", fn {key, value}, acc ->
        # Properly escape the key and encode the value
        json_value = Jason.encode!(value)
        acc <> "  " <> Jason.encode!(key) <> ": " <> json_value <> ",\n"
      end)
      # Remove the trailing comma and close the JSON object
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    output_file = Path.join(@path_statistics, "single_chapter_filenames.json")
    File.write!(output_file, json_content)

    IO.puts("Single chapter filenames have been extracted to #{output_file}")
  end

  def get_chapters_without_category do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Load the filename categories to get "Technische_Daten" exclusions
    {:ok, categories_json} = File.read!(@filename_categories) |> Jason.decode()

    # Extract filenames that should be excluded (under "Technische_Daten")
    excluded_filenames = Map.get(categories_json, "Technische_Daten", [])

    # Get all subfolders in processed_files
    subfolders =
      File.ls!(@processed_files)
      |> Enum.map(fn dir -> Path.join(@processed_files, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Process each subfolder
    result_map =
      Enum.reduce(subfolders, %{}, fn subfolder, acc ->
        subfolder_basename = Path.basename(subfolder)
        summary_file = Path.join(subfolder, "extracted_summary.json")

        if File.exists?(summary_file) do
          # Read and parse the summary file
          {:ok, summary_data} = File.read!(summary_file) |> Jason.decode()

          # Filter for keys that only have "Summary" as value and aren't in excluded_filenames
          keys_with_only_summary =
            summary_data
            |> Enum.filter(fn {key, value} ->
              # Check if the value is a map with only one key and that key is "Summary"
              is_map(value) &&
                map_size(value) == 1 &&
                Map.has_key?(value, "Summary") &&
                !Enum.member?(excluded_filenames, sanitize_filename(key))
            end)
            |> Enum.map(fn {key, _} -> key end)

          # Only add to result if there are any matching keys
          if length(keys_with_only_summary) > 0 do
            Map.put(acc, subfolder_basename, keys_with_only_summary)
          else
            acc
          end
        else
          # If no summary file exists, skip this subfolder
          acc
        end
      end)

    # Sort the result map by keys and create a formatted JSON string
    json_content =
      result_map
      |> Enum.sort_by(fn {key, _} -> key end)
      |> Enum.reduce("{\n", fn {key, values}, acc ->
        sorted_values = Enum.sort(values)
        acc <> "  " <> Jason.encode!(key) <> ": " <> Jason.encode!(sorted_values) <> ",\n"
      end)
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    # Write the result to a JSON file
    output_file = Path.join(@path_statistics, "reports_without_category.json")
    File.write!(output_file, json_content)

    IO.puts("Reports without category have been extracted to #{output_file}")
  end

  def sanitize_filename(filename) do
    # This pattern matches chapter numbering patterns like:
    # "3.1 ", "1.2.3 ", "1. ", etc. at the beginning of the string
    # It handles formats with or without spaces after periods
    Regex.replace(~r/^(\d+\.\s*)*\d+\.?\s*/, filename, "")
  end

  # check on cases where number of chapters is very low
  def check_low_chapter_count do
    # Ensure the statistics directory exists
    File.mkdir_p!(@path_statistics)

    # Get all parent folders in partitioned_md_files
    parent_folders =
      File.ls!(@partitioned_md_files)
      |> Enum.map(fn dir -> Path.join(@partitioned_md_files, dir) end)
      |> Enum.filter(&File.dir?/1)

    # Create a map of parent directory => file count in single_chapters
    parent_file_counts =
      Enum.reduce(parent_folders, %{}, fn parent_folder, acc ->
        parent_name = Path.basename(parent_folder)
        single_chapters_path = Path.join(parent_folder, "single_chapters")

        if File.dir?(single_chapters_path) do
          file_count =
            File.ls!(single_chapters_path)
            |> length()

          Map.put(acc, parent_name, file_count)
        else
          # If there's no single_chapters directory, count as 0
          Map.put(acc, parent_name, 0)
        end
      end)

    # Filter parents with less than 4 files and sort them alphabetically
    low_file_count_parents =
      parent_file_counts
      |> Enum.filter(fn {_parent, count} -> count < 4 end)
      |> Enum.sort_by(fn {parent, _count} -> parent end)
      |> Map.new()

    # Create a JSON string manually to ensure order is preserved
    json_content =
      low_file_count_parents
      |> Enum.reduce("{\n", fn {key, value}, acc ->
        # Properly escape the key and encode the value
        acc <> "  " <> Jason.encode!(key) <> ": " <> Integer.to_string(value) <> ",\n"
      end)
      # Remove the trailing comma and close the JSON object
      |> String.replace_trailing(",\n", "\n")
      |> Kernel.<>("}\n")

    output_file = Path.join(@path_statistics, "low_chapter_count.json")
    File.write!(output_file, json_content)

    IO.puts("Parent directories with low chapter count have been extracted to #{output_file}")
  end
end
