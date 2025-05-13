defmodule UnsupervisedDatasets.SingleChapters do
  @building_dir "N:/Thesis/data_prepare/datasets_building/unsupervised/single_chapters"
  @dataset_output_dir "N:/Thesis/data_prepare/datasets_ready/unsupervised/single_chapters"

  def create_unsupervised_dataset_single_chapters do
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

    IO.puts("Unsupervised dataset for single chapters created successfully.")
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

      if String.contains?(Path.basename(file), "Sachverständige Ausführungen.md") and
           String.contains?(Path.basename(folder_path), "GA_12_32") do
        IO.puts("Found title: #{cleaned_content}")

        IO.puts(
          "does match? #{Regex.match?(~r/^\d+(\.\d+)*\.?\s+[\p{L}\s\-]+$/, cleaned_content)}"
        )

        trimmed = String.trim(cleaned_content)
        IO.puts("Trimmed: #{trimmed}")
        require IEx
        # IEx.pry()
      end

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
    # Group entries by category
    grouped = Enum.group_by(entries, fn {_entry, category} -> category end)

    # Initialize empty lists for each split
    training_set = []
    validation_set = []
    test_set = []

    # For each category, apply round-robin split
    {final_training, final_validation, final_test} =
      Enum.reduce(grouped, {training_set, validation_set, test_set}, fn {category, cat_entries},
                                                                        {train_acc, val_acc,
                                                                         test_acc} ->
        # Calculate split sizes
        total = length(cat_entries)

        # Shuffle the category entries
        shuffled = Enum.shuffle(cat_entries)

        # Apply distribution
        {train, val, test} = distribute_items_by_count(shuffled, [], [], [], total)

        # Log the category split for debugging
        IO.puts("Category: #{category}")
        IO.puts("  Total: #{total}")
        IO.puts("  Training: #{length(train)}")
        IO.puts("  Validation: #{length(val)}")
        IO.puts("  Test: #{length(test)}")

        # Add to accumulators
        {train_acc ++ train, val_acc ++ val, test_acc ++ test}
      end)

    # Log overall split sizes
    IO.puts("\nOverall split:")
    IO.puts("Total entries: #{length(entries)}")
    IO.puts("Training set size: #{length(final_training)}")
    IO.puts("Validation set size: #{length(final_validation)}")
    IO.puts("Test set size: #{length(final_test)}")

    # Save each split to a JSONL file
    save_split_to_file(
      final_training,
      Path.join([output_path, "combined_datasets", "training_set.jsonl"])
    )

    save_split_to_file(
      final_validation,
      Path.join([output_path, "combined_datasets", "validation_set.jsonl"])
    )

    save_split_to_file(
      final_test,
      Path.join([output_path, "combined_datasets", "test_set.jsonl"])
    )
  end

  # Helper function to distribute items based on count
  defp distribute_items_by_count(items, train, val, test, total) when total <= 0 do
    {train, val, test}
  end

  defp distribute_items_by_count(items, train, val, test, total) do
    cond do
      # For 1 item, put it in training
      total == 1 ->
        {items, [], []}

      # For 2 items, split between training and validation
      total == 2 ->
        [first, second] = items
        {[first], [second], []}

      # For 3 items, allocate one to each set
      total == 3 ->
        [first, second, third] = items
        {[first], [second], [third]}

      # For larger sets, use a more balanced distribution
      total > 3 ->
        # Calculate target proportions
        target_train = max(1, floor(total * 0.75))
        target_val = max(1, floor(total * 0.20))
        target_test = total - target_train - target_val

        # Ensure test set gets at least one item
        target_test = max(1, target_test)

        # If needed, adjust training to ensure all sets get some items
        target_train = total - target_val - target_test

        # Split the items according to targets
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
