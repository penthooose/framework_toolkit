defmodule UnsupervisedDatasets.MultipleChapters do
  @building_dir "N:/Thesis/data_prepare/datasets_building/unsupervised/multiple_chapters_processed"
  @dataset_output_dir "N:/Thesis/data_prepare/datasets_ready/unsupervised/multiple_chapters"

  def create_unsupervised_dataset_multiple_chapters do
    # Create output directories if they don't exist
    File.mkdir_p!(Path.join(@dataset_output_dir, "datasets_by_category"))
    File.mkdir_p!(Path.join(@dataset_output_dir, "combined_datasets"))

    # Get all directories in the building directory
    folders =
      File.ls!(@building_dir)
      |> Enum.filter(fn dir ->
        File.dir?(Path.join(@building_dir, dir))
      end)

    # Process each folder to create category datasets
    all_entries =
      Enum.reduce(folders, [], fn folder, acc ->
        folder_path = Path.join(@building_dir, folder)
        entries = process_folder(folder_path, folder)

        # Create a JSONL file for this category
        if length(entries) > 0 do
          jsonl_path = Path.join([@dataset_output_dir, "datasets_by_category", "#{folder}.jsonl"])
          jsonl_content = entries |> Enum.map(&Jason.encode!/1) |> Enum.join("\n")
          File.write!(jsonl_path, jsonl_content)
        end

        # Add entries to the accumulator with their category
        acc ++ Enum.map(entries, fn entry -> {entry, folder} end)
      end)

    # Split and save datasets into training, validation, and test sets
    split_and_save_datasets(all_entries, @dataset_output_dir)

    IO.puts("Unsupervised dataset for multiple chapters created successfully.")
  end

  # Process a folder and extract text from markdown files
  defp process_folder(folder_path, category) do
    IO.puts("Processing folder: #{category}")

    # Get all markdown files in the folder
    files =
      File.ls!(folder_path)
      |> Enum.filter(fn file -> String.ends_with?(file, ".md") end)

    # Process each file
    Enum.reduce(files, [], fn file, acc ->
      file_path = Path.join(folder_path, file)
      content = File.read!(file_path)

      # Clean and format the content
      cleaned_content = clean_content(content)

      # Only add non-empty content that is more than just a title
      if is_valid_content?(cleaned_content) do
        acc ++ [%{"text" => cleaned_content}]
      else
        acc
      end
    end)
  end

  # Check if content is valid (more than just a title)
  defp is_valid_content?(content) do
    trimmed = String.trim(content)

    # If empty, not valid
    if trimmed == "" do
      false
    else
      # Count words in the trimmed content
      word_count = trimmed |> String.split(~r/\s+/) |> Enum.count()

      # Content is valid if it has at least 10 words
      word_count >= 10
    end
  end

  # Clean and format file content
  defp clean_content(content) do
    content
    |> String.trim()
    # Replace 2+ consecutive newlines with single newline
    |> String.replace(~r/\n{2,}/, "\n")
    # Normalize horizontal whitespace (spaces and tabs)
    |> String.replace(~r/[ \t]+/, " ")
    |> String.trim()
  end

  # Split and save datasets into training, validation, and test sets
  defp split_and_save_datasets(entries, output_path) do
    # Instead of splitting by category, we shuffle all entries and split by total count
    # Shuffle all entries together
    shuffled_entries = Enum.shuffle(entries)

    # Calculate split sizes based on total entries
    total_entries = length(shuffled_entries)
    training_size = floor(total_entries * 0.8)
    validation_size = floor(total_entries * 0.15)
    # Test set gets the remainder (approximately 5%)
    test_size = total_entries - training_size - validation_size

    # Split the entries
    {training_entries, rest} = Enum.split(shuffled_entries, training_size)
    {validation_entries, test_entries} = Enum.split(rest, validation_size)

    # Log split sizes
    IO.puts("\nSplit distribution:")
    IO.puts("Total entries: #{total_entries}")
    IO.puts("Training set (80%): #{length(training_entries)}")
    IO.puts("Validation set (15%): #{length(validation_entries)}")
    IO.puts("Test set (5%): #{length(test_entries)}")

    # Save each split to a JSONL file
    save_split_to_file(
      training_entries,
      Path.join([output_path, "combined_datasets", "training_set.jsonl"])
    )

    save_split_to_file(
      validation_entries,
      Path.join([output_path, "combined_datasets", "validation_set.jsonl"])
    )

    save_split_to_file(
      test_entries,
      Path.join([output_path, "combined_datasets", "test_set.jsonl"])
    )
  end

  # Helper function to distribute items based on count
  defp distribute_items_by_count(items, train, val, test, total) when total <= 0 do
    {train, val, test}
  end

  defp distribute_items_by_count(items, train, val, test, total) do
    # For larger sets, use a more balanced distribution
    cond do
      total == 1 ->
        {items, [], []}

      total == 2 ->
        [first, second] = items
        {[first], [second], []}

      total == 3 ->
        [first, second, third] = items
        {[first], [second], [third]}

      total > 3 ->
        target_train = max(1, floor(total * 0.75))
        target_val = max(1, floor(total * 0.20))
        target_test = total - target_train - target_val
        target_test = max(1, target_test)
        target_train = total - target_val - target_test

        {train_items, rest1} = Enum.split(items, target_train)
        {val_items, test_items} = Enum.split(rest1, target_val)

        {train_items, val_items, test_items}
    end
  end

  # Save a list of entries to a JSONL file
  defp save_split_to_file(entries, file_path) do
    jsonl_content =
      entries
      |> Enum.map(fn {entry, _} -> Jason.encode!(entry) end)
      |> Enum.join("\n")

    File.write!(file_path, jsonl_content)
  end
end
