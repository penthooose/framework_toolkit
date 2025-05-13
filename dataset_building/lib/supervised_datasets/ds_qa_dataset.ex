defmodule SupervisedDatasets.QADataset do
  @building_dir "N:/Thesis/data_prepare/datasets_building/supervised/questions_and_answers"
  @dataset_output_dir "N:/Thesis/data_prepare/datasets_ready/supervised/questions_and_answers"

  def create_supervised_dataset_questions_and_answers do
    parent_subfolders =
      File.ls!(@building_dir)
      |> Enum.filter(fn dir -> File.dir?(Path.join(@building_dir, dir)) end)

    # Process each parent subfolder
    # [{instruction, category}] where category is the parent folder name
    {instructions, _} =
      Enum.reduce(parent_subfolders, {[], 1}, fn parent_folder,
                                                 {acc_instructions, batch_counter} ->
        parent_path = Path.join(@building_dir, parent_folder)
        instruction_maps = create_instruction_map(parent_path)
        # Print the first entry of instruction_maps for debugging
        # IO.inspect(instruction_maps, label: "Instruction Maps")

        IO.puts("Processing folder: #{parent_folder}\n")
        # Set initial inst_counter to batch_counter at the beginning of each subfolder
        inst_counter = batch_counter

        # For each instruction map, create instructions
        {chapter_instructions, final_inst_counter} =
          Enum.reduce(instruction_maps, {[], inst_counter}, fn instruction_map,
                                                               {chapter_acc, i_counter} ->
            task = String.to_atom("qa_b#{i_counter}")

            # Create instruction for the current chapter
            # Use parent_folder as the category
            {instruction, category} =
              {SupervisedDatasets.QAInstructionCreation.create_instruction(
                 Map.put(instruction_map, :task, task)
               ), parent_folder}

            # IO.inspect(instruction, label: "Instruction")

            # Add to accumulated instructions with category
            new_instructions = chapter_acc ++ [{instruction, category}]

            # Increment instruction counter and reset if needed
            next_i_counter = if i_counter >= 6, do: 1, else: i_counter + 1

            {new_instructions, next_i_counter}
          end)

        # Increment batch counter for the next parent folder and reset if needed
        next_batch_counter = if batch_counter >= 6, do: 1, else: batch_counter + 1

        # Combine all instructions
        {acc_instructions ++ chapter_instructions, next_batch_counter}
      end)

    IO.puts("Total instructions created: #{length(instructions)}")

    # Group instructions by category and save them
    save_by_category(instructions, @dataset_output_dir)

    # Split and save datasets into training, validation, and test sets
    split_and_save_datasets(instructions, @dataset_output_dir)
  end

  def create_instruction_map(parent_path) do
    # Read the finetuning_layout.json
    layout_path = Path.join(parent_path, "finetuning_layout.json")

    layout_data =
      if File.exists?(layout_path) do
        layout_path
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    # Define the chapters directory
    chapters_dir = Path.join(parent_path, "single_chapters")

    # Process each key in layout_data to create instruction maps
    layout_data
    |> Enum.filter(fn {_, chapter_info} ->
      # Filter out entries with token value > 3000 or empty chapter_files
      chapter_info["combined_token_value"] <= 3000 &&
        length(chapter_info["chapter_files"]) > 0
    end)
    |> Enum.flat_map(fn {key, chapter_info} ->
      # Extract the content from each file in chapter_files
      context_entries =
        chapter_info["chapter_files"]
        |> Enum.map(fn filename ->
          file_path = Path.join(chapters_dir, filename)

          if File.exists?(file_path) do
            content = File.read!(file_path)

            # Remove chapter title (first one or two lines + following newlines)
            content = Regex.replace(~r/\A(.+\n){1,2}\s*\n*/, content, "")
            content = Regex.replace(~r/\n+\z/, content, "")

            # Apply all sanitization functions
            content
            |> remove_footnote_tags()
            |> remove_image_lines()
            |> remove_special_tags()
            |> normalize_newlines()
          else
            ""
          end
        end)
        |> Enum.filter(fn content -> content != "" end)

      # Apply sanitization to questions
      questions =
        chapter_info["questions"] || []

      # Transform and sanitize answers
      answers =
        (chapter_info["response"] || [])
        |> Enum.map(fn answer ->
          # First process N.A./NA answers
          processed_answer =
            case Regex.run(~r/^([A-Z]\.\s+)(N\.A\.)$/, answer) do
              [_, prefix, _] ->
                # Replace with new text while keeping the letter prefix
                replacement = Enum.random(na_replacements())
                "#{prefix}#{replacement}"

              _ ->
                # Not matching our pattern, return unchanged
                answer
            end

          # Then check for N.A. appearing elsewhere in the answer
          processed_answer =
            Regex.replace(~r/N\.A\./, processed_answer, fn _match ->
              Enum.random(na_short_replacements())
            end)

          # Then apply all sanitization functions
          processed_answer
          |> remove_footnote_tags()
          |> remove_image_lines()
          |> remove_special_tags()
          |> normalize_newlines()
        end)

      # Split questions and answers into groups of at most 3 pairs per map
      question_count = length(questions)

      if question_count <= 3 do
        # If 3 or fewer questions, create a single map
        # Shuffle the context entries right before creating the map
        shuffled_context = Enum.shuffle(context_entries)

        [
          %{
            filename: key,
            context: shuffled_context,
            questions: questions,
            answers: answers
          }
        ]
      else
        # For more than 3 questions, distribute them across multiple maps
        create_qa_group_maps(context_entries, questions, answers, [], key)
      end
    end)
  end

  defp na_replacements do
    [
      "Die vorliegenden Quellen enthalten keine Informationen zu dieser Fragestellung.",
      "Zu diesem Aspekt liegen in den Quellen keine Angaben vor.",
      "Diese Information ist in den vorliegenden Daten nicht enthalten.",
      "In den verfügbaren Quellen wird diese Frage nicht behandelt.",
      "Hierzu finden sich keine Angaben in den vorliegenden Textauszügen.",
      "Eine Beantwortung ist anhand der vorhandenen Daten nicht möglich.",
      "Die zur Verfügung stehenden Informationsquellen geben darüber keine Auskunft.",
      "Zu diesem Punkt fehlen entsprechende Informationen in den Textquellen.",
      "Aus den vorliegenden Daten geht diese Information nicht hervor.",
      "Die extrahierten Informationen enthalten hierzu keine spezifischen Angaben."
    ]
  end

  defp na_short_replacements do
    [
      "Keine Angaben in den vorliegenden Daten.",
      "In den Quellen nicht genannt.",
      "Keine Informationen hierzu gefunden.",
      "Aus den gesammelten Daten nicht ersichtlich.",
      "Hierzu liegen keine Informationen vor."
    ]
  end

  # Helper function to create multiple maps with at most 3 question-answer pairs each
  defp create_qa_group_maps(context_entries, questions, answers, acc, filename)
       when length(questions) <= 3 do
    # Add the remaining questions (3 or fewer) to the accumulator
    if length(questions) > 0 do
      # Relabel the questions and answers to start with "A."
      relabeled_questions = relabel_qa_items(questions)
      relabeled_answers = relabel_qa_items(answers)

      # Shuffle the context entries for this specific map
      shuffled_context = Enum.shuffle(context_entries)

      acc ++
        [
          %{
            filename: filename,
            context: shuffled_context,
            questions: relabeled_questions,
            answers: relabeled_answers
          }
        ]
    else
      acc
    end
  end

  defp create_qa_group_maps(context_entries, questions, answers, acc, filename) do
    # Determine how many questions to take for the current map (2 or 3)
    # If we have more than 3 left and the remainder would be 1, take 2 now to ensure at least 2 per group
    take_count =
      if length(questions) > 3 && rem(length(questions), 3) == 1 do
        2
      else
        3
      end

    # Take the first batch of questions and answers
    {curr_questions, rest_questions} = Enum.split(questions, take_count)
    {curr_answers, rest_answers} = Enum.split(answers, take_count)

    # Relabel the current batch of questions and answers to start with "A."
    relabeled_questions = relabel_qa_items(curr_questions)
    relabeled_answers = relabel_qa_items(curr_answers)

    # Shuffle the context entries for this specific map
    shuffled_context = Enum.shuffle(context_entries)

    # Add the current map to the accumulator
    current_map = %{
      filename: filename,
      context: shuffled_context,
      questions: relabeled_questions,
      answers: relabeled_answers
    }

    # Continue with the rest of the questions and answers
    create_qa_group_maps(
      context_entries,
      rest_questions,
      rest_answers,
      acc ++ [current_map],
      filename
    )
  end

  # Helper function to relabel items to start with "A.", "B.", etc.
  defp relabel_qa_items(items) do
    items
    |> Enum.with_index()
    |> Enum.map(fn {item, index} ->
      # Get the letter for the new label (A, B, C, etc.)
      # ASCII 'A' is 65, 'B' is 66, etc.
      new_letter = <<65 + index::utf8>>

      # Replace the existing letter prefix with the new one
      case Regex.run(~r/^[A-Z]\.\s+(.+)$/, item) do
        [_, content] ->
          # If item matches the pattern, replace the prefix
          "#{new_letter}. #{content}"

        _ ->
          # If not matching (unlikely), return the original
          item
      end
    end)
  end

  # Helper functions for saving datasets
  defp save_by_category(instructions, output_path) do
    # Create the category directory if it doesn't exist
    category_dir = Path.join(output_path, "datasets_by_category")
    File.mkdir_p!(category_dir)

    # Group instructions by category (parent folder name)
    instructions
    |> Enum.group_by(fn {_instruction, category} -> category end)
    |> Enum.each(fn {category, category_instructions} ->
      # Create JSONL file with the parent folder name
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

  defp normalize_newlines(content) when is_binary(content) do
    content
    |> String.replace(~r/\n\s*\n/, "\n")
    |> String.replace(~r/\r\n/, "\n")
    |> String.replace(~r/\r/, "\n")
    |> String.replace(~r/\n\s+\n/, "\n\n")
    |> String.replace(~r/^\s+/, "")
    |> String.replace(~r/\s+$/, "")
  end

  defp remove_footnote_tags(content) when is_binary(content) do
    # This pattern will match both standalone footnotes and embedded footnotes
    Regex.replace(~r/\[\^(\d+)\]/, content, "")
  end

  defp remove_footnote_tags(nil), do: ""

  defp remove_image_lines(content) when is_binary(content) do
    content
    |> String.split("\n")
    |> Enum.reject(fn line -> String.starts_with?(String.trim(line), "![") end)
    |> Enum.join("\n")
  end

  defp remove_image_lines(nil), do: ""

  defp remove_special_tags(content) when is_binary(content) do
    tag_regex =
      ~r/(?i)###\s*(FALL ENDE|FALL BEGINN|ZUSAMMENFASSUNG|TEXT|FALL ZUSAMMENFASSUNG|FALL|KERNAUSSAGEN?|STICHWORTE?|SCHLUSSFOLGERUNGE?N?|EMPFEHLUNGE?N?|)\s*\n*/

    Regex.replace(tag_regex, content, "")
  end

  defp remove_special_tags(nil), do: ""

  defp split_and_save_datasets(instructions, output_path) do
    # Create the combined datasets directory if it doesn't exist
    combined_dir = Path.join(output_path, "combined_datasets")
    File.mkdir_p!(combined_dir)

    # Extract instructions from tuples, ignoring categories for splitting
    all_instructions = instructions |> Enum.map(fn {instruction, _category} -> instruction end)

    # Get total count
    total = length(all_instructions)

    # Shuffle all instructions
    shuffled_instructions = Enum.shuffle(all_instructions)

    # Calculate split sizes
    train_count = max(1, floor(total * 0.8))
    val_count = max(1, floor(total * 0.15))
    test_count = total - train_count - val_count

    # Split the shuffled instructions
    {training_set, rest} = Enum.split(shuffled_instructions, train_count)
    {validation_set, test_set} = Enum.split(rest, val_count)

    # Log the split
    IO.puts("Total instructions: #{total}")
    IO.puts("  Training: #{length(training_set)}")
    IO.puts("  Validation: #{length(validation_set)}")
    IO.puts("  Test: #{length(test_set)}")

    # Save each split to a JSONL file
    save_split_to_file(training_set, Path.join(combined_dir, "training_set.jsonl"))
    save_split_to_file(validation_set, Path.join(combined_dir, "validation_set.jsonl"))
    save_split_to_file(test_set, Path.join(combined_dir, "test_set.jsonl"))
  end

  # Modified to accept instructions without category
  defp save_split_to_file(instructions, file_path) do
    jsonl_content =
      instructions
      |> Enum.map(fn instruction ->
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
