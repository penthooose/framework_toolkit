defmodule DP.PrepareQAData do
  @partitioned_files "N:/Thesis/data_prepare/partitioned_md_files"
  @building_supervised_dir "N:/Thesis/data_prepare/datasets_building/supervised"
  @building_supervised_qa_old "N:/Thesis/data_prepare/datasets_building/supervised/questions_and_answers_old"
  @building_supervised_qa "N:/Thesis/data_prepare/datasets_building/supervised/questions_and_answers"

  def prepare_qa_data do
    # prepare_partitioned_files()
    extract_well_formed_QA_data()
    prepare_finetuning_layout()
  end

  def prepare_finetuning_layout(report_dir \\ @building_supervised_qa) do
    # Get all directories in the report_dir
    {:ok, dirs} = File.ls(report_dir)

    dirs
    |> Enum.filter(fn dir -> File.dir?(Path.join(report_dir, dir)) end)
    |> Enum.each(fn dir ->
      dir_path = Path.join(report_dir, dir)
      json_path = Path.join(dir_path, "finetuning_layout.json")
      single_chapters_dir = Path.join(dir_path, "single_chapters")

      if File.exists?(json_path) do
        # Read and parse the JSON file
        {:ok, json_content} = File.read(json_path)
        {:ok, data} = Jason.decode(json_content)

        # Get all files in the single_chapters directory if it exists
        single_chapter_files =
          if File.dir?(single_chapters_dir) do
            {:ok, files} = File.ls(single_chapters_dir)
            files
          else
            []
          end

        # Update each entry with parent_chapter_num and chapter_files
        updated_data =
          Enum.reduce(data, %{}, fn {filename, file_data}, acc ->
            # Extract parent chapter number (e.g., "11." from "11. Zusammenfassung der Ergebnisse.md")
            parent_chapter_num = extract_parent_chapter_num(filename)

            # Find all files in single_chapters that start with the same parent chapter num
            # and calculate token values for each file
            {filtered_chapter_files, combined_token_value} =
              Enum.filter(single_chapter_files, fn file ->
                String.starts_with?(file, parent_chapter_num)
              end)
              |> Enum.reduce({[], 0}, fn chapter_file, {valid_files, token_sum} ->
                # Read file content
                file_path = Path.join(single_chapters_dir, chapter_file)

                if File.exists?(file_path) do
                  case File.read(file_path) do
                    {:ok, chapter_content} ->
                      # Process content to skip chapter title
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
                          [_single] -> ""
                          # If empty, return empty string
                          [] -> ""
                        end

                      # Calculate token count for the content
                      chapter_tokens = estimate_token_length(chapter_content_without_first_lines)

                      # Only include files with at least 10 tokens
                      if chapter_tokens >= 10 do
                        {valid_files ++ [chapter_file], token_sum + chapter_tokens}
                      else
                        {valid_files, token_sum}
                      end

                    # Error reading file, skip it
                    _ ->
                      {valid_files, token_sum}
                  end
                else
                  # File doesn't exist, skip it
                  {valid_files, token_sum}
                end
              end)

            # Add new fields to the file data
            updated_file_data =
              Map.put(file_data, "parent_chapter_num", parent_chapter_num)
              |> Map.put("chapter_files", filtered_chapter_files)
              |> Map.put("combined_token_value", combined_token_value)

            Map.put(acc, filename, updated_file_data)
          end)

        # Write updated data back to file
        {:ok, output_json} = Jason.encode(updated_data, pretty: true)
        File.write!(json_path, output_json)

        IO.puts("Updated finetuning_layout.json for #{dir}")
      else
        IO.puts("No finetuning_layout.json found in #{dir_path}")
      end
    end)
  end

  # Extract the parent chapter number from a filename
  defp extract_parent_chapter_num(filename) do
    # Match patterns like "11." from "11. Chapter name.md" or "3." from "3.3.2 Chapter name.md"
    case Regex.run(~r/^(\d+)\./, filename) do
      [_, number] -> "#{number}."
      # Return empty string if no match
      _ -> ""
    end
  end

  def extract_well_formed_QA_data(report_dir \\ @building_supervised_qa) do
    markers = ["### BEISPIEL", "### ZUSAMMENFASSUNG", "### ANWEISUNG", "### FALL BEGINN"]

    # Get all directories in the report_dir
    {:ok, dirs} = File.ls(report_dir)

    dirs
    |> Enum.filter(fn dir -> File.dir?(Path.join(report_dir, dir)) end)
    |> Enum.each(fn dir ->
      dir_path = Path.join(report_dir, dir)
      json_path = Path.join(dir_path, "extracted_summary.json")

      if File.exists?(json_path) do
        # Read and parse the JSON file
        {:ok, json_content} = File.read(json_path)
        {:ok, data} = Jason.decode(json_content)

        # Extract valid QA pairs
        extracted_data = extract_valid_qa_pairs(data, markers)

        # Write to new file
        output_path = Path.join(dir_path, "finetuning_layout.json")
        {:ok, output_json} = Jason.encode(extracted_data, pretty: true)
        File.write!(output_path, output_json)
      end
    end)
  end

  def estimate_token_length(content, char_per_token \\ 3.5) do
    # Estimate token length based on character count (1 token ≈ 3.5 characters)
    char_count = String.length(content)
    token_estimate = Float.ceil(char_count / char_per_token, 1)
  end

  defp extract_valid_qa_pairs(data, markers) do
    Enum.reduce(data, %{}, fn {filename, file_data}, acc ->
      # Gather all valid QA pairs from all categories in this file
      {valid_questions, valid_responses, _} =
        Enum.reduce(file_data, {[], [], MapSet.new()}, fn {_category, category_data},
                                                          {all_questions, all_responses,
                                                           used_letters} ->
          questions = Map.get(category_data, "questions", [])
          responses = Map.get(category_data, "response", [])

          # Pre-process responses to merge subitems with their parent lettered items
          processed_responses = preprocess_responses_with_subitems(responses)

          # Filter valid QA pairs
          valid_pairs =
            Enum.zip(questions, processed_responses)
            |> Enum.filter(fn {question, response} ->
              # Check if question starts with uppercase letter followed by a dot
              question_valid = String.match?(question, ~r/^[A-Z]\. /)

              # Check if response doesn't contain any markers
              response_valid = not Enum.any?(markers, &String.contains?(response, &1))

              question_valid and response_valid
            end)

          if Enum.empty?(valid_pairs) do
            {all_questions, all_responses, used_letters}
          else
            # Adjust letter prefixes to avoid duplicates when merging
            {new_questions, new_responses, updated_used_letters} =
              adjust_letter_prefixes(valid_pairs, used_letters)

            {all_questions ++ new_questions, all_responses ++ new_responses, updated_used_letters}
          end
        end)

      # Only include files that have valid QA pairs
      if valid_questions != [] do
        Map.put(acc, filename, %{
          "questions" => valid_questions,
          "response" => valid_responses
        })
      else
        acc
      end
    end)
  end

  # Process responses to identify and merge any subitems with their parent lettered items
  defp preprocess_responses_with_subitems(responses) do
    # Group the responses by identifying parent-child relationships
    {processed, _} =
      Enum.reduce(responses, {[], nil}, fn response, {acc, current_parent} ->
        cond do
          # If this is a parent item (starts with letter and dot)
          String.match?(response, ~r/^[A-Z]\. /) ->
            {acc ++ [response], response}

          # If this is any type of subitem (doesn't start with uppercase letter and dot)
          # and we have a current parentnil ->
          current_parent != nil ->
            # Get the last item (parent) and append this subitem with newlines and a tab
            {prev_items, [parent | rest]} = Enum.split(acc, -1)
            new_parent = parent <> "\n\t" <> response
            {prev_items ++ [new_parent | rest], current_parent}

          # Otherwise, treat as a regular item
          true ->
            {acc ++ [response], nil}
        end
      end)

    processed
  end

  # Adjust letter prefixes to avoid duplicates when merging
  defp adjust_letter_prefixes(valid_pairs, used_letters) do
    Enum.reduce(valid_pairs, {[], [], used_letters}, fn {question, response},
                                                        {questions_acc, responses_acc,
                                                         letters_set} ->
      # Extract the letter prefix (like "A.", "B.")
      [letter_prefix | _] = String.split(question, ". ", parts: 2)
      letter = String.first(letter_prefix)

      # Find the next available letter that isn't in used_letters
      new_letter = find_next_available_letter(letter, letters_set)

      # Replace the letter prefix if needed
      {new_question, new_response} =
        if letter == new_letter do
          {question, response}
        else
          {
            String.replace(question, ~r/^[A-Z]\. /, "#{new_letter}. ", global: false),
            String.replace(response, ~r/^[A-Z]\. /, "#{new_letter}. ", global: false)
          }
        end

      # Update the accumulator
      updated_letters_set = MapSet.put(letters_set, new_letter)
      {questions_acc ++ [new_question], responses_acc ++ [new_response], updated_letters_set}
    end)
  end

  # Find the next available letter starting from base_letter
  defp find_next_available_letter(base_letter, used_letters) do
    letter_code = :binary.first(String.upcase(base_letter))

    if MapSet.member?(used_letters, <<letter_code::utf8>>) do
      # If the letter is already used, try the next one
      next_letter_code = letter_code + 1
      # If we go beyond 'Z', wrap around to 'A'
      next_letter_code = if next_letter_code > 90, do: 65, else: next_letter_code
      find_next_available_letter(<<next_letter_code::utf8>>, used_letters)
    else
      <<letter_code::utf8>>
    end
  end

  def prepare_partitioned_files(path_partitioned_files \\ @building_supervised_qa) do
    # Step 1: Copy all folders and contents from @partitioned_files to path_partitioned_files
    # copy_directories(@partitioned_files, path_partitioned_files)

    IO.puts(
      "Completed copying directories from #{@partitioned_files} to #{path_partitioned_files}"
    )

    # Step 2: (Separate operation) Copy specific JSON files from each folder in @building_supervised_qa_old
    copy_json_from_old_to_new(@building_supervised_qa_old, path_partitioned_files)

    IO.puts(
      "Completed copying JSON files from #{@building_supervised_qa_old} to #{path_partitioned_files}"
    )
  end

  # New helper function to handle the separate step of copying JSON files
  defp copy_json_from_old_to_new(source_dir, dest_dir) do
    # Check if source directory exists
    if File.dir?(source_dir) do
      # Get all directories in the source
      case File.ls(source_dir) do
        {:ok, dirs} ->
          dirs
          |> Enum.filter(fn dir -> File.dir?(Path.join(source_dir, dir)) end)
          |> Enum.each(fn dir ->
            src_path = Path.join(source_dir, dir)
            dst_path = Path.join(dest_dir, dir)

            # Create destination directory if it doesn't exist
            File.mkdir_p(dst_path)

            # Copy JSON files for this directory
            copy_json_files(src_path, dst_path)

            # Handle subdirectories
            process_subdirectories(src_path, dst_path)
          end)

        {:error, reason} ->
          IO.puts("Error listing files in #{source_dir}: #{inspect(reason)}")
      end
    else
      IO.puts("Source directory #{source_dir} does not exist. Skipping JSON file copy step.")
    end
  end

  # Helper function to copy JSON files from source to destination
  defp copy_json_files(source_dir, dest_dir) do
    # Copy extracted_meta_info.json if exists
    meta_info_path = Path.join(source_dir, "extracted_meta_info.json")

    if File.exists?(meta_info_path) do
      File.copy(meta_info_path, Path.join(dest_dir, "extracted_meta_info.json"))
    end

    # Copy extracted_summary.json if exists
    summary_path = Path.join(source_dir, "extracted_summary.json")

    if File.exists?(summary_path) do
      File.copy(summary_path, Path.join(dest_dir, "extracted_summary.json"))
    end
  end

  # Helper function to process subdirectories
  defp process_subdirectories(source_dir, dest_dir) do
    # Get all subdirectories
    {:ok, subdirs} = File.ls(source_dir)

    subdirs
    |> Enum.filter(fn subdir -> File.dir?(Path.join(source_dir, subdir)) end)
    |> Enum.each(fn subdir ->
      source_subdir = Path.join(source_dir, subdir)
      dest_subdir = Path.join(dest_dir, subdir)

      # Create destination subdirectory if it doesn't exist
      File.mkdir_p(dest_subdir)

      # Copy JSON files
      copy_json_files(source_subdir, dest_subdir)

      # Recursively process further subdirectories
      process_subdirectories(source_subdir, dest_subdir)
    end)
  end

  # Helper function to recursively copy directories
  defp copy_directories(source, destination) do
    # Create destination directory if it doesn't exist
    File.mkdir_p(destination)

    # Get all files and folders in the source directory
    {:ok, files} = File.ls(source)

    # Copy each file/directory
    Enum.each(files, fn file ->
      source_path = Path.join(source, file)
      dest_path = Path.join(destination, file)

      cond do
        # If it's a directory, recursively copy it
        File.dir?(source_path) ->
          copy_directories(source_path, dest_path)

        # If it's a file, copy it
        true ->
          File.copy(source_path, dest_path)
      end
    end)
  end
end
