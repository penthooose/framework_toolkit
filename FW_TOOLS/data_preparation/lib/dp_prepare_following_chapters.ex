defmodule DP.PrepareFollowingChapters do
  alias Statistics

  @path_statistics "N:/Thesis/data_prepare/statistics"
  @building_dir_1 "N:/Thesis/data_prepare/datasets_building/unsupervised/multiple_chapters_unprocessed"
  @building_dir_2 "N:/Thesis/data_prepare/datasets_building/unsupervised/multiple_chapters_processed"

  @information_extractor_dir "N:/Thesis/data_prepare/information_extraction/symbolic_checked_meta_data"
  @building_dir_3 "N:/Thesis/data_prepare/datasets_building/supervised/multiple_chapters_format4"

  def prepare_following_chapters_supervised(building_dir \\ @building_dir_3) do
    # Statistics.extract_statistics_per_chapter(building_dir)
    define_structure_per_chapter(building_dir, 2200)
  end

  def prepare_following_chapters_unsupervised do
    # after execution of remove_short_full_reports
    report_names_map = create_map_of_report_names()
    following_chapters_map = create_following_chapters(report_names_map)
    write_map_to_json(following_chapters_map)
    combine_chapter_files(following_chapters_map)
  end

  def define_structure_per_chapter(building_dir \\ @building_dir_3, max_tokens \\ 3000) do
    # Get all directories in @building_dir_3
    {:ok, dirs} = File.ls(building_dir)

    # Process each directory
    Enum.each(dirs, fn dir ->
      dir_path = Path.join(building_dir, dir)
      stats_file_path = Path.join(dir_path, "extracted_statistics.json")

      # Check if the statistics file exists
      if File.exists?(stats_file_path) do
        # Read and parse the JSON file
        json_data = File.read!(stats_file_path)
        stats_data = Jason.decode!(json_data)

        # Create the finetuning layout structure
        finetuning_layout = process_chapter_statistics(stats_data, max_tokens)

        # Write the finetuning layout to a JSON file
        output_path = Path.join(dir_path, "finetuning_layout.json")
        File.write!(output_path, Jason.encode!(finetuning_layout, pretty: true))

        IO.puts("Created finetuning_layout.json for #{dir}")
      else
        IO.puts("Warning: No extracted_statistics.json found in #{dir}")
      end
    end)

    IO.puts("Completed processing all directories.")
  end

  # Process statistics data and create the finetuning layout structure
  defp process_chapter_statistics(stats_data, max_tokens) do
    # Get all keys and convert them to integers for proper sorting
    keys =
      stats_data
      |> Map.keys()
      |> Enum.map(&String.to_integer/1)
      |> Enum.sort()

    # Process each chapter entry
    Enum.reduce(keys, %{}, fn key, acc ->
      # Get the chapter data using the string key
      chapter_data = Map.get(stats_data, Integer.to_string(key))
      filename = Map.get(chapter_data, "filename")
      chapter_type = Map.get(chapter_data, "type")

      # Skip processing for current chapters of type "heading_only"
      if chapter_type == "heading_only" do
        acc
      else
        # Continue with regular processing for other chapter types
        num_token_total = Map.get(chapter_data, "num_token_total")

        # Create the initial chapter entry
        chapter_entry =
          if num_token_total > max_tokens do
            # Handle overlong chapters
            %{
              "contained_chapters" => [],
              "contained_summaries" => [],
              "type" => chapter_type,
              "category" => "overlong",
              "num_tokens_total" => num_token_total,
              "sanitized_chapter_num" => Map.get(chapter_data, "sanitized_chapter_num"),
              "sanitized_filename" => Map.get(chapter_data, "sanitized_filename")
            }
          else
            # Find previous chapters to include
            {contained_chapters, chapters_token_count} =
              collect_previous_chapters(keys, key, stats_data, num_token_total, max_tokens)

            # Check if we have enough previous chapters
            # If there's only one chapter and it's not the only previous chapter available,
            # then try using summaries instead
            available_prev_chapters = Enum.count(Enum.filter(keys, fn k -> k < key end))

            if length(contained_chapters) <= 30 and available_prev_chapters > 2 do
              # Only one chapter could be combined, try summaries instead
              {contained_summaries, summaries_token_count} =
                collect_previous_summaries(keys, key, stats_data, num_token_total, max_tokens)

              if Enum.empty?(contained_summaries) do
                # Cannot combine with either chapters or summaries effectively
                %{
                  "contained_chapters" => [],
                  "contained_summaries" => [],
                  "type" => chapter_type,
                  "category" => "single",
                  "num_tokens_total" => num_token_total,
                  "sanitized_chapter_num" => Map.get(chapter_data, "sanitized_chapter_num"),
                  "sanitized_filename" => Map.get(chapter_data, "sanitized_filename")
                }
              else
                # Can combine with summaries
                %{
                  "contained_chapters" => [],
                  "contained_summaries" => contained_summaries,
                  "type" => chapter_type,
                  "category" => "only_summaries",
                  "num_tokens_total" => summaries_token_count,
                  "sanitized_chapter_num" => Map.get(chapter_data, "sanitized_chapter_num"),
                  "sanitized_filename" => Map.get(chapter_data, "sanitized_filename")
                }
              end
            else
              # Either we have multiple chapters or this is the only previous chapter available
              if Enum.empty?(contained_chapters) do
                # Try to combine with previous summaries as no chapters fit
                {contained_summaries, summaries_token_count} =
                  collect_previous_summaries(keys, key, stats_data, num_token_total, max_tokens)

                if Enum.empty?(contained_summaries) do
                  # Cannot combine with either chapters or summaries
                  %{
                    "contained_chapters" => [],
                    "contained_summaries" => [],
                    "type" => chapter_type,
                    "category" => "single",
                    "num_tokens_total" => num_token_total,
                    "sanitized_chapter_num" => Map.get(chapter_data, "sanitized_chapter_num"),
                    "sanitized_filename" => Map.get(chapter_data, "sanitized_filename")
                  }
                else
                  # Can combine with summaries
                  %{
                    "contained_chapters" => [],
                    "contained_summaries" => contained_summaries,
                    "type" => chapter_type,
                    "category" => "only_summaries",
                    "num_tokens_total" => summaries_token_count,
                    "sanitized_chapter_num" => Map.get(chapter_data, "sanitized_chapter_num"),
                    "sanitized_filename" => Map.get(chapter_data, "sanitized_filename")
                  }
                end
              else
                # Can combine with chapters (either multiple or the only one available)
                %{
                  "contained_chapters" => contained_chapters,
                  "contained_summaries" => [],
                  "type" => chapter_type,
                  "category" => "only_chapters",
                  "num_tokens_total" => chapters_token_count,
                  "sanitized_chapter_num" => Map.get(chapter_data, "sanitized_chapter_num"),
                  "sanitized_filename" => Map.get(chapter_data, "sanitized_filename")
                }
              end
            end
          end

        # Add the entry to the accumulator
        Map.put(acc, filename, chapter_entry)
      end
    end)
  end

  # Collect previous chapters that can be combined with the current chapter
  defp collect_previous_chapters(
         all_keys,
         current_key,
         stats_data,
         initial_token_total,
         max_tokens
       ) do
    # Get all keys that come before the current key
    previous_keys = Enum.filter(all_keys, fn k -> k < current_key end) |> Enum.sort(:desc)

    # Start accumulating previous chapters
    Enum.reduce_while(previous_keys, {[], initial_token_total}, fn prev_key,
                                                                   {chapters_acc, token_count} ->
      prev_chapter_data = Map.get(stats_data, Integer.to_string(prev_key))
      prev_chapter_type = Map.get(prev_chapter_data, "type")
      prev_chapter_token = Map.get(prev_chapter_data, "num_token_chapter")

      # For technical or heading_only chapters, use only chapter tokens
      # For other types, use chapter tokens for now (will handle summaries later)
      new_token_count = token_count + prev_chapter_token

      if new_token_count <= max_tokens do
        # Can include this chapter
        prev_chapter_entry = %{
          "filename" => Map.get(prev_chapter_data, "filename"),
          "type" => prev_chapter_type,
          "position" => prev_key,
          "sanitized_chapter_num" => Map.get(prev_chapter_data, "sanitized_chapter_num"),
          "sanitized_filename" => Map.get(prev_chapter_data, "sanitized_filename")
        }

        # Continue accumulating
        {:cont, {[prev_chapter_entry | chapters_acc], new_token_count}}
      else
        # Would exceed max_tokens, stop here
        {:halt, {chapters_acc, token_count}}
      end
    end)
  end

  # Collect previous summaries that can be combined with the current chapter
  defp collect_previous_summaries(
         all_keys,
         current_key,
         stats_data,
         initial_token_total,
         max_tokens
       ) do
    # Get all keys that come before the current key
    previous_keys = Enum.filter(all_keys, fn k -> k < current_key end) |> Enum.sort(:desc)

    # Start accumulating previous summaries
    Enum.reduce_while(previous_keys, {[], initial_token_total}, fn prev_key,
                                                                   {summaries_acc, token_count} ->
      prev_chapter_data = Map.get(stats_data, Integer.to_string(prev_key))
      prev_chapter_type = Map.get(prev_chapter_data, "type")

      # For technical or heading_only chapters, use chapter tokens
      # For other types, use summary tokens
      prev_token_value =
        if prev_chapter_type in ["technical", "heading_only"] do
          Map.get(prev_chapter_data, "num_token_chapter", 0)
        else
          Map.get(prev_chapter_data, "num_token_summary", 0)
        end

      new_token_count = token_count + prev_token_value

      if new_token_count <= max_tokens do
        # Can include this summary
        prev_summary_entry = %{
          "filename" => Map.get(prev_chapter_data, "filename"),
          "type" => prev_chapter_type,
          "position" => prev_key,
          "sanitized_chapter_num" => Map.get(prev_chapter_data, "sanitized_chapter_num"),
          "sanitized_filename" => Map.get(prev_chapter_data, "sanitized_filename")
        }

        # Continue accumulating
        {:cont, {[prev_summary_entry | summaries_acc], new_token_count}}
      else
        # Would exceed max_tokens, stop here
        {:halt, {summaries_acc, token_count}}
      end
    end)
  end

  def prepare_structure do
  end

  def create_map_of_report_names do
    # Read the JSON file with report statistics
    json_data =
      File.read!(Path.join([@path_statistics, "report_names_without_trained_full_report.json"]))

    reports_data = Jason.decode!(json_data)

    # Get list of directories from @building_dir_1
    {:ok, dirs} = File.ls(@building_dir_1)

    # Process each directory
    Enum.reduce(dirs, %{}, fn dir, acc ->
      # Check if the directory exists in the JSON data
      if Map.has_key?(reports_data, dir) do
        files_data = Map.get(reports_data, dir)

        # Extract chapter numbers from filenames, sort by chapter number, then create a sequential map
        files_map = create_map_of_filenames(files_data)

        Map.put(acc, dir, files_map)
      else
        acc
      end
    end)
  end

  def create_map_of_filenames(files_data) do
    files_with_chapter_nums =
      Enum.map(files_data, fn {filename, token_length} ->
        case extract_chapter_number(filename) do
          {:ok, chapter_num} -> {chapter_num, {filename, token_length, false}}
          # For files without proper chapter numbers
          :error -> {999, {filename, token_length, false}}
        end
      end)

    # Sort by the extracted chapter numbers
    sorted_files =
      files_with_chapter_nums
      |> Enum.sort_by(fn {chapter_num, _} -> chapter_num end)
      |> Enum.map(fn {_, file_data} -> file_data end)

    # Create map with sequential integer keys
    files_map =
      sorted_files
      |> Enum.with_index(1)
      |> Enum.map(fn {file_data, index} -> {index, file_data} end)
      |> Map.new()
  end

  def create_following_chapters(report_names_map) do
    Enum.reduce(report_names_map, %{}, fn {report_name, files_map}, acc ->
      # Get the highest key integer in the map (the last file)
      max_key = files_map |> Map.keys() |> Enum.max()

      # Group filenames based on token values with limit of 3000
      file_groups = group_files_by_token_limit(files_map, max_key)

      # Skip reports with no file groups
      if Enum.empty?(file_groups) do
        acc
      else
        # Add the report with its file groups to the accumulator
        Map.put(acc, report_name, file_groups)
      end
    end)
  end

  def combine_chapter_files(following_chapters_map) do
    # Process each folder entry in the map
    Enum.each(following_chapters_map, fn {folder_name, file_groups} ->
      # Create the destination folder if it doesn't exist
      dest_folder_path = Path.join(@building_dir_2, folder_name)
      File.mkdir_p!(dest_folder_path)

      # Process each file group
      Enum.each(file_groups, fn {files_list, _total_tokens} ->
        # Sort files by key in ascending order
        sorted_files = Enum.sort_by(files_list, fn {key, _} -> key end)

        # Get the min and max keys for naming the combined file
        min_key = sorted_files |> List.first() |> elem(0)
        max_key = sorted_files |> List.last() |> elem(0)

        # Create the combined filename
        combined_filename = "#{min_key}-#{max_key}.md"
        combined_file_path = Path.join(dest_folder_path, combined_filename)

        # Combine the content of all files without adding headers
        combined_content =
          Enum.map(sorted_files, fn {_key, filename} ->
            # Source file path
            source_file_path = Path.join([@building_dir_1, folder_name, filename])

            # Read the content - just the raw content without any additions
            File.read!(source_file_path)
          end)
          |> Enum.join("\n\n")

        # Write the combined content to the destination file
        File.write!(combined_file_path, combined_content)
      end)

      IO.puts("Successfully created combined files for #{folder_name}")
    end)

    IO.puts("Successfully combined all chapter files.")
  end

  def write_map_to_json(map) do
    # Convert tuples to lists for JSON encoding
    converted_map = convert_tuples_to_lists(map)

    output_path = Path.join([@path_statistics, "maps_of_following_chapters.json"])
    json_data = Jason.encode!(converted_map, pretty: true)
    File.write!(output_path, json_data)
    IO.puts("Successfully wrote map to #{output_path}")
    map
  end

  # Helper function to convert tuples in the map to lists
  defp convert_tuples_to_lists(map) when is_map(map) do
    Map.new(map, fn {key, value} ->
      {key, convert_tuples_to_lists(value)}
    end)
  end

  defp convert_tuples_to_lists(list) when is_list(list) do
    Enum.map(list, &convert_tuples_to_lists/1)
  end

  # Handle tuples specifically - convert ANY tuple to a list
  defp convert_tuples_to_lists(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&convert_tuples_to_lists/1)
  end

  defp convert_tuples_to_lists(value), do: value

  # Helper function to group files based on token limit
  defp group_files_by_token_limit(files_map, start_key) do
    max_token_limit = 3000
    group_files(files_map, start_key, [], [], 0, max_token_limit)
  end

  defp group_files(files_map, current_key, groups, current_group, current_sum, max_token_limit)
       when current_key < 1 do
    # We've processed all files, add the current group if not empty
    if current_group == [] do
      groups
    else
      # Check if the current group has more than one file
      case length(current_group) do
        # Skip if only one file in the list
        1 -> groups
        # Valid group with multiple files, add it (regardless of whether it starts with chapter 1)
        _ -> [{Enum.reverse(current_group), current_sum} | groups]
      end
    end
  end

  defp group_files(files_map, current_key, groups, current_group, current_sum, max_token_limit) do
    # Get the file data for the current key
    {filename, token_value, _used} = Map.get(files_map, current_key)

    # Check if adding this file would exceed the token limit
    if current_sum + token_value <= max_token_limit do
      # We can add this file to the current group
      group_files(
        files_map,
        current_key - 1,
        groups,
        [{current_key, filename} | current_group],
        current_sum + token_value,
        max_token_limit
      )
    else
      # Check if the current group has more than one file
      current_group_to_add =
        case current_group do
          # Empty group, nothing to add
          [] -> []
          # Single file group, skip it
          [_] -> []
          # Valid group with multiple files, add it (regardless of whether it starts with chapter 1)
          _ -> [{Enum.reverse(current_group), current_sum}]
        end

      # Start a new group with the current file if it can fit alone
      if token_value <= max_token_limit do
        # Start a new group with current file
        group_files(
          files_map,
          current_key - 1,
          current_group_to_add ++ groups,
          [{current_key, filename}],
          token_value,
          max_token_limit
        )
      else
        # Skip this file as it exceeds the limit on its own
        group_files(
          files_map,
          current_key - 1,
          current_group_to_add ++ groups,
          [],
          0,
          max_token_limit
        )
      end
    end
  end

  # Helper function to extract chapter number from filename
  def extract_chapter_number(filename) do
    # Match patterns like "1. Something" or "4.2 Something"
    case Regex.run(~r/^(\d+(?:\.\d+)?)/, filename) do
      [_, num_str] ->
        # Convert to numeric value for proper sorting
        parts = String.split(num_str, ".")

        case parts do
          [main] ->
            {:ok, String.to_integer(main)}

          [main, sub] ->
            # For subchapters, we convert to a numeric value
            # e.g., "4.2" becomes 4 + 0.2 = 4.2
            {:ok, String.to_integer(main) + String.to_float("0." <> sub)}

          _ ->
            :error
        end

      nil ->
        :error
    end
  end
end
