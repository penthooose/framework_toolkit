defmodule DP.RemoveOverlengthFiles do
  @path_statistics "N:/Thesis/data_prepare/statistics/chapter_token_lengths.json"
  @path_statistics_2 "N:/Thesis/data_prepare/statistics/report_names_without_trained_full_report.json"
  @partitioned_md_files "N:/Thesis/data_prepare/partitioned_md_files"
  @building_dir "N:/Thesis/data_prepare/datasets_building/unsupervised/single_chapters"
  @building_dir_2 "N:/Thesis/data_prepare/datasets_building/unsupervised/multiple_chapters_unprocessed"

  def remove_short_full_reports do
    # Read the JSON file with statistics
    {:ok, json_content} = File.read(@path_statistics_2)
    {:ok, reports_data} = Jason.decode(json_content)

    # Get all directories in the parent directory
    parent_dirs = File.ls!(@partitioned_md_files)

    # Process each directory that appears in the JSON file
    parent_dirs
    |> Enum.filter(fn dir -> Map.has_key?(reports_data, dir) end)
    |> Enum.each(fn dir ->
      process_directory(dir, reports_data[dir])
    end)
  end

  def remove_overlength_files(token_limit \\ 3000, only_full_reports \\ false) do
    # Read and parse the statistics JSON file
    case File.read(@path_statistics) do
      {:ok, content} ->
        token_lengths = Jason.decode!(content)
        remove_files_exceeding_limit(token_lengths, token_limit, only_full_reports)

      {:error, reason} ->
        IO.puts("Error reading statistics file: #{reason}")
    end
  end

  def move_processed_files_to_building do
    # Read and parse the statistics JSON file
    case File.read(@path_statistics) do
      {:ok, content} ->
        token_lengths = Jason.decode!(content)
        move_specified_files(token_lengths)

      {:error, reason} ->
        IO.puts("Error reading statistics file: #{reason}")
    end
  end

  defp process_directory(dir_name, file_entries) do
    # Define the source directory containing the single_chapter files
    source_dir = Path.join([@partitioned_md_files, dir_name, "single_chapters"])

    # Create the target directory
    target_dir = Path.join(@building_dir_2, dir_name)
    File.mkdir_p!(target_dir)

    # Check if the source directory exists
    if File.dir?(source_dir) do
      # Get all files in the source directory
      files = File.ls!(source_dir)

      # Get the filenames that are in the JSON entries
      valid_files = Map.keys(file_entries)

      # Extract files that match the JSON entries
      files
      |> Enum.filter(fn file -> Enum.member?(valid_files, file) end)
      |> Enum.each(fn file ->
        source_path = Path.join(source_dir, file)
        target_path = Path.join(target_dir, file)

        # Copy the file to the target directory
        File.cp!(source_path, target_path)
      end)

      IO.puts("Processed directory: #{dir_name}, copied #{length(valid_files)} files")
    else
      IO.puts("Source directory not found: #{source_dir}")
    end
  end

  defp move_specified_files(token_lengths) do
    # Iterate through each directory in the token_lengths map
    Enum.each(token_lengths, fn {directory, files} ->
      source_dir = Path.join(@partitioned_md_files, directory)
      target_dir = Path.join(@building_dir, directory)

      if File.exists?(source_dir) do
        # Create the target directory if it doesn't exist
        File.mkdir_p!(target_dir)

        # Get all files in the source directory (recursively)
        all_source_files = find_files_recursively(source_dir)

        # For each file listed in the JSON
        Enum.each(files, fn {filename, _token_length} ->
          # Find the file in the source directory (case insensitive)
          source_file_path = find_file_by_name(all_source_files, filename)

          if source_file_path do
            target_file_path = Path.join(target_dir, Path.basename(source_file_path))

            # Copy the file to the target directory
            File.copy!(source_file_path, target_file_path)
            IO.puts("Copied #{source_file_path} to #{target_file_path}")
          else
            IO.puts("Warning: File #{filename} not found in #{source_dir}")
          end
        end)
      else
        IO.puts("Warning: Directory #{source_dir} not found")
      end
    end)
  end

  defp find_files_recursively(directory) do
    case File.ls(directory) do
      {:ok, files} ->
        Enum.flat_map(files, fn file ->
          path = Path.join(directory, file)

          if File.dir?(path) do
            find_files_recursively(path)
          else
            [path]
          end
        end)

      {:error, _reason} ->
        []
    end
  end

  defp find_file_by_name(file_paths, filename) do
    # Find a file with the matching name (case insensitive)
    Enum.find(file_paths, fn path ->
      String.downcase(Path.basename(path)) == String.downcase(filename)
    end)
  end

  defp remove_files_exceeding_limit(token_lengths, token_limit, only_full_reports) do
    # Iterate through each directory entry in the token_lengths map
    Enum.each(token_lengths, fn {directory, files} ->
      directory_path = Path.join(@partitioned_md_files, directory)

      if File.exists?(directory_path) do
        # Process each file in the token_lengths data
        Enum.each(files, fn {filename, token_count} ->
          # Skip files that aren't full_report.md if only_full_reports is true
          should_process = !only_full_reports || String.downcase(filename) == "full_report.md"

          if should_process && token_count > token_limit do
            # Find and remove the file
            find_and_remove_file(directory_path, filename, token_count, token_limit)
          end
        end)
      else
        IO.puts("Directory not found: #{directory_path}")
      end
    end)
  end

  defp find_and_remove_file(directory_path, filename, token_count, token_limit) do
    # Look for the file directly in the directory
    file_path = Path.join(directory_path, filename)

    if File.exists?(file_path) do
      remove_file(file_path, filename, token_count, token_limit)
    else
      # The file might be in a subdirectory, so we need to search
      case find_file_in_subdirectories(directory_path, filename) do
        nil ->
          IO.puts("File not found: #{filename} in #{directory_path}")

        found_path ->
          remove_file(found_path, filename, token_count, token_limit)
      end
    end
  end

  defp find_file_in_subdirectories(directory_path, filename) do
    directory_path
    |> File.ls!()
    |> Enum.filter(fn entry ->
      entry_path = Path.join(directory_path, entry)
      File.dir?(entry_path)
    end)
    |> Enum.find_value(fn subdir ->
      file_path = Path.join([directory_path, subdir, filename])
      if File.exists?(file_path), do: file_path, else: nil
    end)
  end

  defp remove_file(file_path, filename, token_count, token_limit) do
    IO.puts("Removing #{file_path} (#{token_count} tokens > #{token_limit} token limit)")

    case File.rm(file_path) do
      :ok ->
        IO.puts("Successfully removed #{filename}")

      {:error, reason} ->
        IO.puts("Error removing #{filename}: #{reason}")
    end
  end
end
