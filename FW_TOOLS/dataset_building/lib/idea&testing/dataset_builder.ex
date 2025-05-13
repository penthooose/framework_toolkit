defmodule DatasetBuilder do
  @moduledoc """
  Module to construct datasets for fine-tuning, from extracted summaries and report contents.
  """
  @processed_reports_first_stage "N:/Thesis/data_prepare/datasets_building/first_stage"
  @processed_reports_second_stage "N:/Thesis/data_prepare/datasets_building/second_stage"
  @datasets_output_first_stage "N:/Thesis/data_prepare/first_stage/datasets_ready"
  @datasets_output_second_stage "N:/Thesis/data_prepare/second_stage/datasets_ready"

  alias InstructionCreation

  def create_dataset_first_stage do
    parent_subfolders =
      File.ls!(@processed_reports_first_stage)
      |> Enum.filter(fn dir ->
        File.dir?(Path.join(@processed_reports_first_stage, dir))
      end)

    # Process each parent subfolder
    # [{instruction, category}]
    {instructions, _} =
      Enum.reduce(parent_subfolders, {[], 1}, fn parent_folder,
                                                 {acc_instructions, batch_counter} ->
        parent_path = Path.join(@processed_reports_first_stage, parent_folder)
        chapter_map = create_chapter_map(parent_path)
        IO.puts("Processing folder: #{parent_folder}\n")
        # Set initial inst_counter to batch_counter at the beginning of each subfolder
        inst_counter = batch_counter

        # For each chapter in the chapter map, create instructions
        {chapter_instructions, final_inst_counter} =
          Enum.reduce(chapter_map, {[], inst_counter}, fn {_chapter_num, chapter_data},
                                                          {chapter_acc, i_counter} ->
            task = String.to_atom("create_chapter_b#{i_counter}")
            chapter_name = chapter_data["chapter_name"]
            categories = Map.keys(chapter_data["categories"])
            meta_data = chapter_data["meta_data"]
            content = chapter_data["chapter_content"]
            summary = chapter_data["summary"]
            qa = chapter_data["categories"]

            # Create instruction for the current chapter
            {instruction, category} =
              InstructionCreation.create_instruction(
                task,
                chapter_name,
                meta_data,
                categories,
                summary,
                qa,
                content
              )

            # Add to accumulated instructions with category
            new_instructions = chapter_acc ++ [{instruction, category}]

            # Increment instruction counter and reset if needed
            next_i_counter = if i_counter >= 7, do: 1, else: i_counter + 1

            {new_instructions, next_i_counter}
          end)

        # Increment batch counter for the next parent folder and reset if needed
        next_batch_counter = if batch_counter >= 7, do: 1, else: batch_counter + 1

        # Combine all instructions
        {acc_instructions ++ chapter_instructions, next_batch_counter}
      end)

    # Group instructions by category and save them
    save_by_category(instructions, @datasets_output_first_stage)

    # Split and save datasets into training, validation, and test sets
    split_and_save_datasets(instructions, @datasets_output_first_stage)

    # Return the final list of instructions with their categories
    # instructions
  end

  def create_chapter_map(parent_path) do
    # Read metadata and summary files
    meta_data_path = Path.join(parent_path, "extracted_meta_info.json")
    summary_path = Path.join(parent_path, "extracted_summary.json")

    meta_data =
      if File.exists?(meta_data_path) do
        meta_data_path
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    summary_data =
      if File.exists?(summary_path) do
        summary_path
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    # Get and sort chapter files
    chapters_path = Path.join(parent_path, "single_chapters")

    chapter_files =
      if File.exists?(chapters_path) and File.dir?(chapters_path) do
        chapters_path
        |> File.ls!()
        |> Enum.filter(fn file -> String.ends_with?(file, ".md") end)
        |> Enum.sort_by(fn file ->
          # Extract the number prefix for sorting (e.g., "1." from "1. Vorwort")
          case Regex.run(~r/^(\d+)\./, file) do
            [_, num] -> String.to_integer(num)
            # If no number found, sort to the end
            nil -> 999
          end
        end)
      else
        []
      end

    # Create a map entry for each chapter
    chapter_files
    |> Enum.with_index(1)
    |> Enum.map(fn {file, index} ->
      chapter_content = File.read!(Path.join(chapters_path, file))

      # Extract categories and summary from the summary data
      chapter_summary_data = Map.get(summary_data, file, %{})

      # Get summary if available
      summary =
        case Map.get(chapter_summary_data, "Summary") do
          nil ->
            ""

          summary_data ->
            case Map.get(summary_data, "response") do
              [response | _] -> response
              _ -> ""
            end
        end

      # Get categories (all keys except "Summary")
      categories =
        chapter_summary_data
        |> Map.keys()
        |> Enum.reject(fn key -> key == "Summary" end)
        |> Enum.reduce(%{}, fn category, acc ->
          category_data = Map.get(chapter_summary_data, category)
          Map.put(acc, category, category_data)
        end)

      # Create the chapter map
      {"chapter_#{index}",
       %{
         "chapter_name" => file,
         "chapter_content" => chapter_content,
         "meta_data" => meta_data,
         "categories" => categories,
         "summary" => summary
       }}
    end)
    |> Map.new()
  end

  @doc """
  Groups instructions by category and saves them to separate JSONL files.
  """
  def save_by_category(instructions, output_path) do
    # Create the category directory if it doesn't exist
    category_dir = Path.join(output_path, "datasets_by_category")
    File.mkdir_p!(category_dir)

    # Group instructions by category
    instructions
    |> Enum.group_by(fn {_instruction, category} -> category end)
    |> Enum.each(fn {category, category_instructions} ->
      # Create JSONL file for this category
      file_path = Path.join(category_dir, "#{category}.jsonl")

      # Convert instructions to JSONL format
      jsonl_content =
        category_instructions
        |> Enum.map(fn {instruction, _} ->
          # Parse the instruction string as JSON and then re-encode it properly
          case Jason.decode(instruction) do
            {:ok, parsed_instruction} -> Jason.encode!(parsed_instruction)
            _ -> instruction
          end
        end)
        |> Enum.join("\n")

      # Write to file
      File.write!(file_path, jsonl_content)
    end)
  end

  @doc """
  Splits the instructions across categories into training (75%), validation (20%), and test (5%) sets,
  maintaining category distribution using a round-robin approach.
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
    IO.puts("Total instructions: #{length(instructions)}")
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
  """
  defp distribute_round_robin(items, total) do
    # Initialize accumulators
    training = []
    validation = []
    test = []

    # Distribution pattern based on total items
    {train, val, test} = distribute_items_by_count(items, training, validation, test, total)

    {train, val, test}
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
  """
  defp save_split_to_file(instructions, file_path) do
    jsonl_content =
      instructions
      |> Enum.map(fn {instruction, _} ->
        # Parse the instruction string as JSON and then re-encode it properly
        case Jason.decode(instruction) do
          {:ok, parsed_instruction} -> Jason.encode!(parsed_instruction)
          _ -> instruction
        end
      end)
      |> Enum.join("\n")

    File.write!(file_path, jsonl_content)
  end
end
