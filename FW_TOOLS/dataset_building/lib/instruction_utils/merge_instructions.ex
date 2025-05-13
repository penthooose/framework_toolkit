defmodule MergeInstructions do
  @dataset_dir [
    "N:/Thesis/data_prepare/datasets_ready/unsupervised/single_chapters",
    "N:/Thesis/data_prepare/datasets_ready/unsupervised/multiple_chapters"
  ]
  @dataset_output_dir "N:/Thesis/data_prepare/datasets_ready/unsupervised/mixed_chapters"

  # Target split percentages
  @split_ratio %{
    "training_set.jsonl" => 0.80,
    "validation_set.jsonl" => 0.15,
    "test_set.jsonl" => 0.05
  }

  def merge_instructions do
    # Create output combined_datasets directory
    output_combined_dir = Path.join(@dataset_output_dir, "combined_datasets")
    File.mkdir_p!(output_combined_dir)

    # Define the dataset file names we want to merge
    dataset_files = ["training_set.jsonl", "validation_set.jsonl", "test_set.jsonl"]

    # Collect all instructions from all files in all source directories
    IO.puts("Collecting all instructions from source directories...")

    all_instructions =
      Enum.flat_map(@dataset_dir, fn source_dir ->
        source_combined_dir = Path.join(source_dir, "combined_datasets")

        Enum.flat_map(dataset_files, fn dataset_file ->
          source_file_path = Path.join(source_combined_dir, dataset_file)

          if File.exists?(source_file_path) do
            # Read the file content as lines
            content = File.read!(source_file_path)
            lines = String.split(content, "\n", trim: true)
            IO.puts("  Found #{length(lines)} instructions in #{source_file_path}")
            lines
          else
            []
          end
        end)
      end)

    total_instructions = length(all_instructions)
    IO.puts("\nTotal instructions collected: #{total_instructions}")

    # Shuffle all instructions
    IO.puts("Shuffling instructions...")
    shuffled_instructions = Enum.shuffle(all_instructions)

    # Calculate split sizes
    training_size = floor(total_instructions * @split_ratio["training_set.jsonl"])
    validation_size = floor(total_instructions * @split_ratio["validation_set.jsonl"])
    # Ensure we account for all instructions
    test_size = total_instructions - training_size - validation_size

    # Split the instructions
    {training_set, remaining} = Enum.split(shuffled_instructions, training_size)
    {validation_set, test_set} = Enum.split(remaining, validation_size)

    # Create a map of dataset file to instructions
    splits = %{
      "training_set.jsonl" => training_set,
      "validation_set.jsonl" => validation_set,
      "test_set.jsonl" => test_set
    }

    # Write each split to its file
    Enum.each(splits, fn {filename, instructions} ->
      output_file_path = Path.join(output_combined_dir, filename)
      File.write!(output_file_path, Enum.join(instructions, "\n"))
      IO.puts("Wrote #{length(instructions)} instructions to #{output_file_path}")
    end)

    IO.puts("\nSplitting completed successfully!")

    # Display distribution
    IO.puts("\nDistribution of instructions:")

    Enum.each(splits, fn {filename, instructions} ->
      count = length(instructions)
      percentage = count / total_instructions * 100.0

      IO.puts("  #{filename}: #{count} (#{Float.round(percentage, 1)}%)")
    end)

    # Verify total matches
    total_after_split = Enum.sum(Enum.map(splits, fn {_, instrs} -> length(instrs) end))

    if total_after_split == total_instructions do
      IO.puts("\nVerification passed: All instructions accounted for.")
    else
      IO.puts(
        "\nWarning: Instruction count mismatch. Before: #{total_instructions}, After: #{total_after_split}"
      )
    end
  end
end
