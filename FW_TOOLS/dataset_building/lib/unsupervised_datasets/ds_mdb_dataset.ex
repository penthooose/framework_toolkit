defmodule DatasetBuilder.Unsupervised.MDBDataset do
  @mdb_records_dir "N:/Thesis/data_prepare/mdb/mdb_data"
  @dataset_output_dir "N:/Thesis/data_prepare/datasets_ready/unsupervised/mdb_datasets"

  @tables [
    "Autraggeber",
    "Geräteart",
    "Gerätetyp",
    "Hersteller",
    "Makler",
    "Schaden",
    "Versicherungsnehmer"
  ]

  @entries_without_zero_values [
    "Sonstiges",
    "Durchwahl",
    "Durchwahl 1",
    "Durchwahl 2",
    "Durchwahl 3",
    "Durchwahl 4",
    "Durchwahl 5",
    "Durchwahl 6",
    "Postleitzahl"
  ]

  def create_unsupervised_dataset_mdb do
    # Create output directories if they don't exist
    File.mkdir_p!(Path.join(@dataset_output_dir, "datasets_by_category"))
    File.mkdir_p!(Path.join(@dataset_output_dir, "combined_datasets"))

    # Get all jsonl files from the MDB records directory
    jsonl_files =
      File.ls!(@mdb_records_dir)
      |> Enum.filter(fn file -> String.ends_with?(file, ".jsonl") end)

    # Process each file and collect all formatted records with their category
    all_records =
      Enum.flat_map(jsonl_files, fn file ->
        category = Path.basename(file, ".jsonl")

        # Read and process records from the file
        file_path = Path.join(@mdb_records_dir, file)
        records = process_jsonl_file(file_path, category)

        # Save records by category
        save_category_records(records, category)

        # Return records with category for combined datasets
        records |> Enum.map(fn record -> {record, category} end)
      end)

    # Split into training, validation, and test sets and save
    split_and_save_datasets(all_records, @dataset_output_dir)

    # Return count of processed records
    length(all_records)
  end

  @doc """
  Processes a single JSONL file and formats each record.

  Reads each JSON record from the file, formats it with the category name,
  and filters out empty values. Skips the first line as it contains column names.
  """
  def process_jsonl_file(file_path, category) do
    File.read!(file_path)
    |> String.split("\n", trim: true)
    # Drop the first line containing column names
    |> Enum.drop(1)
    |> Enum.map(fn line ->
      case sanitize_and_decode_json(line) do
        {:ok, record} -> format_record(record, category)
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  Sanitizes JSON strings that might contain escaped quotes or other problematic characters,
  then attempts to decode the sanitized string.
  """
  def sanitize_and_decode_json(json_string) do
    # First try sanitizing and then decoding
    sanitized_string = sanitize_json_string(json_string)

    case Jason.decode(sanitized_string) do
      {:ok, decoded} ->
        # Clean up any escaped quotes in the decoded values too
        cleaned_decoded = clean_escaped_quotes_in_values(decoded)
        {:ok, cleaned_decoded}

      {:error, _} ->
        # If it still fails, return the error
        {:error, "Failed to decode JSON"}
    end
  end

  @doc """
  Sanitizes a JSON string by handling problematic character sequences.
  Handles issues like escaped quotes within string values.
  """
  def sanitize_json_string(json_string) do
    # Replace problematic escaped quotes with single quotes
    # This regex finds field values and replaces any unescaped quotes inside them
    Regex.replace(~r/:\s*"(.*?)(?<!\\)"(?=,|\s*\}|\s*\n|$)/s, json_string, fn whole, content ->
      # Replace escaped quotes with single quotes
      sanitized_content = String.replace(content, "\\\"", "")
      ": \"#{sanitized_content}\""
    end)
  end

  @doc """
  Cleans any escaped quotes in the already decoded map values.
  """
  def clean_escaped_quotes_in_values(decoded) when is_map(decoded) do
    decoded
    |> Enum.map(fn {key, value} ->
      cleaned_value =
        if is_binary(value) do
          String.replace(value, "\\\"", "'")
        else
          value
        end

      {key, cleaned_value}
    end)
    |> Map.new()
  end

  def clean_escaped_quotes_in_values(value), do: value

  @doc """
  Formats a single record with the category name and filters out empty values.
  Handles additional sanitization for display in the formatted text output.
  """
  def format_record(record, category) do
    formatted_text =
      "Tabelle: #{category}\n" <>
        (record
         |> Enum.filter(fn {key, value} ->
           value != "" &&
             !(key in @entries_without_zero_values && (value == "0" || value == "0.00"))
         end)
         |> Enum.map(fn {key, value} -> "#{key}: #{value}" end)
         |> Enum.join("\n"))

    Jason.encode!(%{"text" => formatted_text})
  end

  @doc """
  Saves records for a specific category to a JSONL file.
  """
  def save_category_records(records, category) do
    file_path =
      Path.join([
        @dataset_output_dir,
        "datasets_by_category",
        "#{category}.jsonl"
      ])

    # Join records with newlines and write to file
    File.write!(file_path, Enum.join(records, "\n"))
  end

  @doc """
  Splits records across categories into training (75%), validation (20%), and test (5%) sets,
  maintaining category distribution using a round-robin approach.
  Reuses functionality from DatasetBuilder module.
  """
  def split_and_save_datasets(instructions, output_path) do
    # Create the combined datasets directory if it doesn't exist
    combined_dir = Path.join(output_path, "combined_datasets")
    File.mkdir_p!(combined_dir)

    # Group instructions by category
    grouped = Enum.group_by(instructions, fn {_instruction, category} -> category end)

    # Initialize empty lists for each split
    training_set = []
    validation_set = []
    test_set = []

    # For each category, apply round-robin split with priority for training
    {final_training, final_validation, final_test} =
      Enum.reduce(grouped, {training_set, validation_set, test_set}, fn {category,
                                                                         cat_instructions},
                                                                        {train_acc, val_acc,
                                                                         test_acc} ->
        # Calculate split sizes
        total = length(cat_instructions)

        # Shuffle the category instructions
        shuffled = Enum.shuffle(cat_instructions)

        # Apply round-robin distribution with priority
        {train, val, test} = distribute_round_robin(shuffled, total)

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
    IO.puts("Total records: #{length(instructions)}")
    IO.puts("Training set size: #{length(final_training)}")
    IO.puts("Validation set size: #{length(final_validation)}")
    IO.puts("Test set size: #{length(final_test)}")

    # Save each split to a JSONL file
    save_split_to_file(final_training, Path.join(combined_dir, "training_set.jsonl"))
    save_split_to_file(final_validation, Path.join(combined_dir, "validation_set.jsonl"))
    save_split_to_file(final_test, Path.join(combined_dir, "test_set.jsonl"))
  end

  @doc """
  Distributes items using a round-robin approach with priority for training,
  followed by validation, and then test sets.
  Reused from DatasetBuilder module.
  """
  defp distribute_round_robin(items, total) do
    # Initialize accumulators
    training = []
    validation = []
    test = []

    # Distribution pattern based on total items
    distribute_items_by_count(items, training, validation, test, total)
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

  @doc """
  Saves a list of instructions to a JSONL file.
  Reused from DatasetBuilder module.
  """
  defp save_split_to_file(instructions, file_path) do
    jsonl_content =
      instructions
      |> Enum.map(fn {instruction, _} -> instruction end)
      |> Enum.join("\n")

    File.write!(file_path, jsonl_content)
  end
end
