defmodule MixedCoherenceDS do
  alias InstructionPreparation

  @datasets_supervised [
    "N:/Thesis/data_prepare/datasets_ready/supervised/multiple_chapters_format3/combined_datasets",
    "N:/Thesis/data_prepare/datasets_ready/supervised/multiple_chapters_format4/combined_datasets"
  ]
  @datasets_unsupervised [
    "N:/Thesis/data_prepare/datasets_ready/unsupervised/mixed_chapters/combined_datasets"
  ]
  @dataset_output "N:/Thesis/data_prepare/datasets_ready/mixed_coherence/combined_datasets"
  @dataset_output_sp "N:/Thesis/data_prepare/datasets_ready/supervised_coherence/combined_datasets"
  @dataset_output_us "N:/Thesis/data_prepare/datasets_ready/unsupervised_coherence/combined_datasets"

  def create_supervised_coherence_dataset(num_instructions \\ 3000, max_tokens \\ 3200) do
    # Ensure output directory exists
    File.mkdir_p!(@dataset_output_sp)

    # Collect instructions from supervised sources
    supervised_instructions =
      @datasets_supervised
      |> Enum.flat_map(fn source_dir ->
        collect_filtered_instructions(source_dir, max_tokens)
      end)
      |> Enum.take_random(num_instructions)

    # Calculate split counts
    total_count = length(supervised_instructions)
    training_count = floor(total_count * 0.7)
    validation_count = floor(total_count * 0.18)
    test_count = total_count - training_count - validation_count

    # Split instructions
    {training_set, remaining} = Enum.split(supervised_instructions, training_count)
    {validation_set, test_set} = Enum.split(remaining, validation_count)

    # Write output files
    write_jsonl(@dataset_output_sp, "training_set.jsonl", training_set)
    write_jsonl(@dataset_output_sp, "validation_set.jsonl", validation_set)
    write_jsonl(@dataset_output_sp, "test_set.jsonl", test_set)

    # Return stats
    %{
      total_instructions: total_count,
      training: length(training_set),
      validation: length(validation_set),
      test: length(test_set),
      supervised_count: total_count
    }
  end

  def create_unsupervised_coherence_dataset(num_instructions \\ 3000, max_tokens \\ 3200) do
    # Ensure output directory exists
    File.mkdir_p!(@dataset_output_us)

    # Collect instructions from unsupervised sources
    unsupervised_instructions =
      @datasets_unsupervised
      |> Enum.flat_map(fn source_dir ->
        collect_filtered_instructions(source_dir, max_tokens)
      end)
      |> Enum.take_random(num_instructions)

    # Calculate split counts
    total_count = length(unsupervised_instructions)
    training_count = floor(total_count * 0.7)
    validation_count = floor(total_count * 0.18)
    test_count = total_count - training_count - validation_count

    # Split instructions
    {training_set, remaining} = Enum.split(unsupervised_instructions, training_count)
    {validation_set, test_set} = Enum.split(remaining, validation_count)

    # Write output files
    write_jsonl(@dataset_output_us, "training_set.jsonl", training_set)
    write_jsonl(@dataset_output_us, "validation_set.jsonl", validation_set)
    write_jsonl(@dataset_output_us, "test_set.jsonl", test_set)

    # Return stats
    %{
      total_instructions: total_count,
      training: length(training_set),
      validation: length(validation_set),
      test: length(test_set),
      unsupervised_count: total_count
    }
  end

  def create_mixed_coherence_dataset(num_instructions \\ 4300, max_tokens \\ 3200) do
    # Calculate instruction counts for each source
    supervised_count = floor(num_instructions * 0.6)
    unsupervised_count = num_instructions - supervised_count

    # Calculate per-source counts for supervised datasets
    instructions_per_supervised = div(supervised_count, length(@datasets_supervised))

    # Ensure output directory exists
    File.mkdir_p!(@dataset_output)

    # Collect instructions from supervised sources
    supervised_instructions =
      @datasets_supervised
      |> Enum.flat_map(fn source_dir ->
        collect_filtered_instructions(source_dir, max_tokens)
      end)
      |> sample_instructions_from_source(supervised_count, length(@datasets_supervised))

    # Collect instructions from unsupervised sources
    unsupervised_instructions =
      @datasets_unsupervised
      |> Enum.flat_map(fn source_dir ->
        collect_filtered_instructions(source_dir, max_tokens)
      end)
      |> Enum.take_random(unsupervised_count)

    # Combine and shuffle all instructions
    all_instructions =
      (supervised_instructions ++ unsupervised_instructions)
      |> Enum.shuffle()

    # Calculate split counts
    total_count = length(all_instructions)
    training_count = floor(total_count * 0.7)
    validation_count = floor(total_count * 0.18)
    test_count = total_count - training_count - validation_count

    # Split instructions
    {training_set, remaining} = Enum.split(all_instructions, training_count)
    {validation_set, test_set} = Enum.split(remaining, validation_count)

    # Write output files
    write_jsonl(@dataset_output, "training_set.jsonl", training_set)
    write_jsonl(@dataset_output, "validation_set.jsonl", validation_set)
    write_jsonl(@dataset_output, "test_set.jsonl", test_set)

    # Return stats
    %{
      total_instructions: total_count,
      training: length(training_set),
      validation: length(validation_set),
      test: length(test_set),
      supervised_count: length(supervised_instructions),
      unsupervised_count: length(unsupervised_instructions)
    }
  end

  # Collect and filter instructions from a source directory
  defp collect_filtered_instructions(source_dir, max_tokens) do
    # Get validation and test sets
    validation_path = Path.join(source_dir, "validation_set.jsonl")
    test_path = Path.join(source_dir, "test_set.jsonl")

    # Read and filter instructions
    read_and_filter_instructions(validation_path, max_tokens) ++
      read_and_filter_instructions(test_path, max_tokens)
  end

  # Read and filter instructions from a JSONL file
  defp read_and_filter_instructions(file_path, max_tokens) do
    if File.exists?(file_path) do
      File.stream!(file_path)
      |> Stream.map(&String.trim/1)
      |> Stream.filter(fn line -> line != "" end)
      |> Stream.map(&Jason.decode!/1)
      |> Stream.filter(fn instruction ->
        # Determine if supervised or unsupervised format
        cond do
          Map.has_key?(instruction, "input") && Map.has_key?(instruction, "output") ->
            # Supervised format
            input_text = Map.get(instruction, "input", "")
            output_text = Map.get(instruction, "output", "")
            full_text = input_text <> output_text

            InstructionPreparation.estimate_token_length(full_text) <= max_tokens

          Map.has_key?(instruction, "text") ->
            # Unsupervised format
            text = Map.get(instruction, "text", "")

            InstructionPreparation.estimate_token_length(text) <= max_tokens

          true ->
            false
        end
      end)
      |> Enum.to_list()
    else
      []
    end
  end

  # Sample instructions ensuring equal distribution from multiple sources
  defp sample_instructions_from_source(instructions, total_count, source_count) do
    # Sort by source to ensure we get an equal distribution
    instructions_per_source = div(total_count, source_count)

    instructions
    |> Enum.take_random(total_count)
  end

  # Write instructions to a JSONL file
  defp write_jsonl(dir, filename, instructions) do
    file_path = Path.join(dir, filename)

    content =
      instructions
      |> Enum.map(&Jason.encode!/1)
      |> Enum.join("\n")

    File.write!(file_path, content)
  end
end
