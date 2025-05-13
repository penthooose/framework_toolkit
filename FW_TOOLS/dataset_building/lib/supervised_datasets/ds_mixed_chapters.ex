defmodule SupervisedDatasets.MixedChapters do
  @building_dir "N:/Thesis/data_prepare/datasets_building/supervised/multiple_chapters_format4"
  @dataset_output_dir "N:/Thesis/data_prepare/datasets_ready/supervised/multiple_chapters_format4"
  @excluded_categories ["overlong", "single"]

  def create_supervised_dataset_mixed_multiple_chapters do
    parent_subfolders =
      File.ls!(@building_dir)
      |> Enum.filter(fn dir -> File.dir?(Path.join(@building_dir, dir)) end)

    # Process each parent subfolder
    # [{instruction, category}]
    {instructions, _} =
      Enum.reduce(parent_subfolders, {[], 1}, fn parent_folder,
                                                 {acc_instructions, batch_counter} ->
        parent_path = Path.join(@building_dir, parent_folder)
        instruction_maps = create_instruction_map(parent_path)

        IO.puts("Processing folder: #{parent_folder}\n")
        # Set initial inst_counter to batch_counter at the beginning of each subfolder
        inst_counter = batch_counter

        # For each instruction map, create instructions
        {chapter_instructions, final_inst_counter} =
          Enum.reduce(instruction_maps, {[], inst_counter}, fn instruction_map,
                                                               {chapter_acc, i_counter} ->
            task = String.to_atom("create_chapter_b#{i_counter}")

            # Create instruction for the current chapter
            {instruction, category} =
              SupervisedDatasets.MixedInstructionCreation.create_instruction(
                Map.put(instruction_map, :task, task)
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

    IO.puts("Total instructions created: #{length(instructions)}")

    # Group instructions by category and save them
    save_by_category(instructions, @dataset_output_dir)

    # Split and save datasets into training, validation, and test sets
    split_and_save_datasets(instructions, @dataset_output_dir)
  end

  def create_instruction_map(parent_path, debug \\ false) do
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

    # Read the extracted_summary.json
    summary_path = Path.join(parent_path, "extracted_summary.json")

    summary_data =
      if File.exists?(summary_path) do
        summary_path
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    # Read the extracted_statistics.json for metadata
    stats_path = Path.join(parent_path, "extracted_statistics.json")

    stats_data =
      if File.exists?(stats_path) do
        stats_path
        |> File.read!()
        |> Jason.decode!()
      else
        %{}
      end

    # Chapters directory
    chapters_dir = Path.join(parent_path, "single_chapters")

    # Process each chapter entry in the layout
    layout_data
    |> Enum.map(fn {filename, chapter_layout} ->
      # Format chapter number correctly (e.g., 5.21 -> 5.2.1)
      chapter_num = format_chapter_number(chapter_layout["sanitized_chapter_num"])
      chapter_name = chapter_layout["sanitized_filename"]
      category = chapter_layout["category"]
      type = chapter_layout["type"]

      # Get chapter content
      content_path = Path.join(chapters_dir, filename)

      content =
        if File.exists?(content_path) do
          File.read!(content_path)
        else
          ""
        end

      # Get chapter summary from extracted_summary.json - set to nil for technical chapters
      summary =
        if type == "technical" do
          nil
        else
          case Map.get(summary_data, filename) do
            %{"summary" => summary_text} -> summary_text
            _ -> ""
          end
        end

      # Get metadata from included_meta_data in summary_data
      meta_data =
        case Map.get(summary_data, filename) do
          %{"included_meta_data" => meta} -> meta
          _ -> %{}
        end

      # Extract and prepare the list of chapters or summaries with their types
      processed_content_list =
        if chapter_layout["contained_summaries"] == [] do
          # Use chapter contents
          prepare_chapter_content_list(chapter_layout["contained_chapters"], chapters_dir)
        else
          # Use summaries, but handle technical chapters specially
          prepare_summary_content_list(
            chapter_layout["contained_summaries"],
            chapters_dir,
            summary_data
          )
        end

      # Determine chapter types (main_chapter or sub_chapter) based on chapter number relationships
      previous_content = determine_chapter_types(processed_content_list)

      # If category is "only_summaries", set all types in previous_content to "sub_chapter"
      previous_content =
        if category == "only_summaries" do
          previous_content
          |> Enum.map(fn {idx, content} ->
            {idx, Map.put(content, :type, "sub_chapter")}
          end)
          |> Map.new()
        else
          # For "only_chapters" category, identify parent-child relationships
          if category == "only_chapters" do
            update_parent_chapters_types(previous_content, chapter_num)
          else
            previous_content
          end
        end

      # Create the instruction map
      instruction_map =
        %{
          chapter_num: chapter_num,
          chapter_name: chapter_name,
          category: category,
          type: type,
          content: content,
          summary: summary,
          meta_data: meta_data,
          previous_content: previous_content
        }

      if debug do
        # Inspect instruction map without content, summary, and meta_data
        # Create a version for inspection that hides content fields
        instruction_map_for_inspect =
          instruction_map
          |> Map.drop([:content, :summary, :meta_data])
          |> Map.update(:previous_content, %{}, fn prev ->
            prev
            |> Enum.map(fn {idx, entry} ->
              {idx, Map.drop(entry, [:content])}
            end)
            |> Map.new()
          end)

        Map.drop(instruction_map, [:content, :summary, :meta_data])

        IO.inspect(instruction_map_for_inspect, label: "Instruction Map")
      end

      instruction_map
    end)
  end

  # Helper function to identify parent-child relationships and update parent chapter types
  defp update_parent_chapters_types(previous_content, current_chapter_num, debug \\ false) do
    # Get all chapter numbers and their indices - create map with chapter_num as key and idx as value
    chapter_nums_with_indices =
      previous_content
      |> Enum.map(fn {idx, content} -> {content.chapter_num, idx} end)
      |> Enum.into(%{})

    # Create a map to track which chapters have children
    chapters_with_children =
      previous_content
      |> Enum.reduce(%{}, fn {_, content}, acc ->
        chapter_num = content.chapter_num

        # If this is a multi-level chapter (contains a dot)
        if String.contains?(chapter_num, ".") do
          # Find its parent by taking everything before the last dot
          parent_num = chapter_num |> String.split(".") |> Enum.drop(-1) |> Enum.join(".")

          # Add this parent to the map of chapters with children
          Map.put(acc, parent_num, true)
        else
          acc
        end
      end)

    # First pass: Process all single-level chapters to identify direct parents of current chapter
    updated_content =
      Enum.reduce(previous_content, previous_content, fn {idx, content}, acc ->
        chapter_num = content.chapter_num
        current_type = Map.get(content, :type, "unknown")
        chapter_type = Map.get(content, :chapter_type, "unknown")

        # Determine if this chapter is a direct parent of the current chapter
        is_direct_parent = is_direct_parent?(chapter_num, current_chapter_num)

        if debug do
          IO.puts(
            "\nExamining previous chapter #{chapter_num} (current type: #{current_type}, chapter_type: #{chapter_type})"
          )

          IO.puts(
            "  Is direct parent of current chapter #{current_chapter_num}? #{is_direct_parent}"
          )
        end

        # Check if this is a single-level chapter that is a direct parent of current chapter
        if !String.contains?(chapter_num, ".") && is_direct_parent do
          if debug do
            IO.puts(
              "  Found direct parent: #{chapter_num} for current chapter: #{current_chapter_num}"
            )
          end

          # Determine the appropriate type based on the chapter type and current type
          new_type =
            cond do
              # For only_heading or heading_only chapters
              chapter_type == "only_heading" || chapter_type == "heading_only" ||
                current_type == "main_chapter_no_content" ||
                  current_type == "heading_only" ->
                "main_chapter_no_closing_no_content"

              # For other direct parents
              true ->
                "main_chapter_no_closing"
            end

          if debug do
            IO.puts("  Setting direct parent chapter #{chapter_num} to #{new_type}")
          end

          Map.put(acc, idx, Map.put(content, :type, new_type))
        else
          acc
        end
      end)

    # Second pass: Process multi-level chapters as before
    # Process each previous content entry
    Enum.reduce(previous_content, updated_content, fn {idx, content}, acc ->
      # Skip if this chapter was already processed in the first pass
      if Map.get(acc, idx) != content do
        # This chapter was already updated in first pass
        acc
      else
        chapter_num = content.chapter_num
        current_type = Map.get(content, :type, "unknown")
        chapter_type = Map.get(content, :chapter_type, "unknown")

        # For multi-level chapters (e.g., "3.1"), find and process their parent chapters
        new_acc =
          if String.contains?(chapter_num, ".") do
            # Process parent-child relationships for multi-level chapters
            parent_chapters = identify_parent_chapters(chapter_num, chapter_nums_with_indices)

            if length(parent_chapters) > 0 do
              if debug do
                IO.puts("  Found #{length(parent_chapters)} parent chapters for #{chapter_num}")
              end

              # Update parent chapter types
              Enum.reduce(parent_chapters, acc, fn {parent_idx, parent_num}, inner_acc ->
                parent_content = Map.get(inner_acc, parent_idx)

                if parent_content do
                  parent_current_type = Map.get(parent_content, :type, "unknown")
                  parent_chapter_type = Map.get(parent_content, :chapter_type, "unknown")
                  parent_is_direct_parent = is_direct_parent?(parent_num, current_chapter_num)
                  has_children = Map.has_key?(chapters_with_children, parent_num)

                  if debug do
                    IO.puts(
                      "    Parent #{parent_num} (type: #{parent_chapter_type}, current_type: #{parent_current_type})"
                    )

                    IO.puts("    Is direct parent of current chapter? #{parent_is_direct_parent}")
                    IO.puts("    Has children? #{has_children}")
                  end

                  # Determine type based on parent's type and relationship with current chapter
                  new_type =
                    cond do
                      # If parent already has no_content suffix, preserve it
                      String.contains?(parent_current_type, "no_content") ->
                        if parent_is_direct_parent do
                          "main_chapter_no_closing_no_content"
                        else
                          parent_current_type
                        end

                      # Only_heading type or heading_only
                      parent_content[:chapter_type] == "only_heading" ||
                        parent_content[:chapter_type] == "heading_only" ||
                          parent_current_type == "heading_only" ->
                        if parent_is_direct_parent do
                          "main_chapter_no_closing_no_content"
                        else
                          "main_chapter_no_content"
                        end

                      # Previously marked no_content types
                      parent_current_type == "main_chapter_no_content" ->
                        if parent_is_direct_parent do
                          "main_chapter_no_closing_no_content"
                        else
                          "main_chapter_no_content"
                        end

                      # Direct parent-child relationship
                      parent_is_direct_parent ->
                        "main_chapter_no_closing"

                      # Has children chapters - should be main_chapter_opening
                      has_children ->
                        "main_chapter_opening"

                      # Other cases - regular main chapter
                      true ->
                        "main_chapter"
                    end

                  if debug do
                    IO.puts("    Setting type to: #{new_type}")
                  end

                  Map.put(inner_acc, parent_idx, Map.put(parent_content, :type, new_type))
                else
                  inner_acc
                end
              end)
            else
              acc
            end
          else
            acc
          end

        # For single-level chapters (like "3"), determine type based on relationship with current chapter
        if !String.contains?(chapter_num, ".") &&
             (chapter_type == "only_heading" || chapter_type == "heading_only") do
          content_from_acc = Map.get(new_acc, idx)
          current_assigned_type = Map.get(content_from_acc, :type, "unknown")

          # Don't change types if they've already been set to no_closing variants in the first pass
          if String.contains?(current_assigned_type, "no_closing") do
            new_acc
          else
            # If not already processed as a main chapter but has only_heading type
            if !String.starts_with?(current_assigned_type, "main_chapter") do
              if debug do
                IO.puts(
                  "  Setting single-level chapter #{chapter_num} to main_chapter_no_content"
                )
              end

              Map.put(new_acc, idx, Map.put(content_from_acc, :type, "main_chapter_no_content"))
            else
              new_acc
            end
          end
        else
          # Check if this chapter has children
          has_children = Map.has_key?(chapters_with_children, chapter_num)
          content_from_acc = Map.get(new_acc, idx)
          current_assigned_type = Map.get(content_from_acc, :type, "unknown")

          # If it has children and is a main chapter but not opening/closing variant, update to opening
          if has_children && current_assigned_type == "main_chapter" do
            if debug do
              IO.puts("  Setting chapter with children #{chapter_num} to main_chapter_opening")
            end

            Map.put(new_acc, idx, Map.put(content_from_acc, :type, "main_chapter_opening"))
          else
            new_acc
          end
        end
      end
    end)
  end

  # Helper function to parse chapter number into a comparable numeric value
  defp parse_chapter_num(chapter_num) do
    chapter_num
    |> String.split(".")
    |> Enum.map(fn part ->
      case Integer.parse(part) do
        {num, _} -> num
        :error -> 0
      end
    end)
  end

  # Helper function to determine if a chapter is a direct parent of another
  # e.g., "4" is direct parent of "4.1" but not of "5.1"
  defp is_direct_parent?(parent_num, child_num) do
    String.starts_with?(child_num, "#{parent_num}.")
  end

  # Identify parent chapters of a given chapter number
  defp identify_parent_chapters(chapter_num, chapter_nums_with_indices) do
    parts = String.split(chapter_num, ".")

    if length(parts) <= 1 do
      # No parent chapters for top-level chapters
      []
    else
      # Generate all possible parent chapter numbers
      1..(length(parts) - 1)
      |> Enum.map(fn i ->
        # Create parent chapter number by taking i parts
        parent_num = Enum.take(parts, i) |> Enum.join(".")

        # Find index of this parent chapter if it exists (using direct map lookup)
        parent_idx = Map.get(chapter_nums_with_indices, parent_num)

        if parent_idx, do: {parent_idx, parent_num}, else: nil
      end)
      |> Enum.reject(&is_nil/1)
    end
  end

  # Helper function to prepare chapter content list with positions
  defp prepare_chapter_content_list(chapter_list, chapters_dir) do
    chapter_list
    |> Enum.sort_by(fn chapter -> chapter["position"] end)
    |> Enum.map(fn chapter ->
      # Extract relevant information
      filename = chapter["filename"]
      type_info = chapter["type"]
      position = chapter["position"]
      chapter_path = Path.join(chapters_dir, filename)

      # Extract chapter number from filename (e.g., "5.1" from "5.1 Chapter Name.md")
      chapter_num = extract_chapter_number(filename)

      # Get sanitized values from the chapter layout
      sanitized_filename = chapter["sanitized_filename"]
      sanitized_chapter_num = format_chapter_number(chapter["sanitized_chapter_num"])

      chapter_content =
        if File.exists?(chapter_path) do
          File.read!(chapter_path)
        else
          ""
        end

      %{
        position: position,
        filename: filename,
        content: chapter_content,
        chapter_num: chapter_num,
        chapter_type: type_info,
        sanitized_filename: sanitized_filename,
        sanitized_chapter_num: sanitized_chapter_num
      }
    end)
  end

  # Helper function to prepare summary content list with positions
  defp prepare_summary_content_list(summary_list, chapters_dir, summary_data) do
    summary_list
    |> Enum.sort_by(fn chapter -> chapter["position"] end)
    |> Enum.map(fn chapter ->
      # Extract relevant information
      filename = chapter["filename"]
      type_info = chapter["type"]
      position = chapter["position"]

      # Extract chapter number from filename
      chapter_num = extract_chapter_number(filename)

      # Get sanitized values from the chapter layout
      sanitized_filename = chapter["sanitized_filename"]
      sanitized_chapter_num = format_chapter_number(chapter["sanitized_chapter_num"])

      content =
        if type_info == "technical" do
          # For technical chapters, use content
          chapter_path = Path.join(chapters_dir, filename)

          if File.exists?(chapter_path) do
            File.read!(chapter_path)
          else
            ""
          end
        else
          # For regular chapters, use summary
          case Map.get(summary_data, filename) do
            %{"summary" => summary_text} -> summary_text
            _ -> ""
          end
        end

      %{
        position: position,
        filename: filename,
        content: content,
        chapter_num: chapter_num,
        chapter_type: type_info,
        sanitized_filename: sanitized_filename,
        sanitized_chapter_num: sanitized_chapter_num
      }
    end)
  end

  # Extract chapter number from filename (e.g., "5.1" from "5.1 Chapter Name.md")
  defp extract_chapter_number(filename) do
    case Regex.run(~r/^(\d+(?:\.\d+)*)/, filename) do
      [_, num_str] -> num_str
      _ -> ""
    end
  end

  # Determine if each chapter is a main_chapter or sub_chapter based on its relationship to the next chapter
  defp determine_chapter_types(chapter_list) do
    # Process items with next item context
    chapter_list_with_types =
      chapter_list
      |> Enum.with_index()
      |> Enum.map(fn {current_item, index} ->
        # Determine if this is the last item
        next_item =
          if index < length(chapter_list) - 1, do: Enum.at(chapter_list, index + 1), else: nil

        # Determine if this is a main chapter or sub-chapter
        chapter_type = determine_single_chapter_type(current_item, next_item)

        # Return chapter with its type
        Map.put(current_item, :chapter_hierarchy_type, chapter_type)
      end)

    # Convert the enriched list to the required map format
    chapter_list_with_types
    |> Enum.with_index(1)
    |> Enum.map(fn {chapter, idx} ->
      {idx,
       %{
         content: chapter.content,
         type: chapter.chapter_hierarchy_type,
         chapter_num: format_chapter_number(chapter.sanitized_chapter_num),
         sanitized_filename: chapter.sanitized_filename
       }}
    end)
    |> Map.new()
  end

  # Determine if a chapter is a main_chapter or sub_chapter based on its relationship with the next chapter
  defp determine_single_chapter_type(current_item, next_item) do
    # Check if the current item is a heading-only chapter
    if current_item.chapter_type == "heading_only" do
      "main_chapter_no_content"
    else
      # If there's no next item, default to sub_chapter
      if next_item == nil do
        "sub_chapter"
      else
        current_num = current_item.chapter_num
        next_num = next_item.chapter_num

        # Compare chapter numbers to determine relationship
        compare_chapter_numbers(current_num, next_num)
      end
    end
  end

  # Compare chapter numbers to determine their relationship and proper chapter type
  defp compare_chapter_numbers(current_num, next_num) do
    # If either number is empty, they can't be related
    if current_num == "" || next_num == "" do
      "sub_chapter"
    else
      # Split chapter numbers into parts
      current_parts = String.split(current_num, ".")
      next_parts = String.split(next_num, ".")

      # Calculate level difference
      level_difference = calculate_level_difference(current_parts, next_parts)

      cond do
        # Main chapter that opens multiple subchapter levels
        level_difference < -1 ->
          # Determine how many levels are opened
          opening_levels = abs(level_difference)
          "main_chapter_opening" <> String.duplicate("_opening", opening_levels - 1)

        # Main chapter that opens one subchapter level
        level_difference == -1 ->
          "main_chapter_opening"

        # Subchapter that closes multiple levels up
        level_difference > 1 ->
          # Determine how many levels are closed
          closing_levels = level_difference
          "subchapter_closing" <> String.duplicate("_closing", closing_levels - 1)

        # Subchapter that closes one level up
        level_difference == 1 ->
          "subchapter_closing"

        # Same level but not sequential - changing from main_chapter to sub_chapter
        level_difference == 0 && !is_next_sequential_chapter(current_num, next_num) ->
          "sub_chapter"

        # Default - sequential chapters at same level
        true ->
          "sub_chapter"
      end
    end
  end

  # Calculate level difference between chapter numbers (positive: next is higher level, negative: next is lower level)
  defp calculate_level_difference(current_parts, next_parts) do
    # Calculate basic level difference based on number of parts
    basic_level_diff = length(current_parts) - length(next_parts)

    # Check if prefix match to refine the determination
    if basic_level_diff == 0 do
      # If same level, check if they share the same prefix except the last part
      prefix_current = Enum.drop(current_parts, -1)
      prefix_next = Enum.drop(next_parts, -1)

      if prefix_current == prefix_next do
        # Same branch, sequential chapters
        0
      else
        # Same level but different branch - determine if it's closing one branch and opening another
        common_prefix_length = find_common_prefix_length(current_parts, next_parts)

        # Calculate implicit level changes
        length(current_parts) - common_prefix_length - (length(next_parts) - common_prefix_length)
      end
    else
      # Different levels - check if they're related
      common_prefix_length = find_common_prefix_length(current_parts, next_parts)

      if common_prefix_length > 0 do
        # Related chapters, normal level difference applies
        basic_level_diff
      else
        # Completely different branches - treat as significant level change
        # If current has more parts, it's closing multiple levels
        # If next has more parts, it's opening multiple levels
        basic_level_diff
      end
    end
  end

  # Find length of common prefix between two chapter numbers
  defp find_common_prefix_length(parts1, parts2) do
    parts1
    |> Enum.zip(parts2)
    |> Enum.reduce_while(0, fn {a, b}, count ->
      if a == b, do: {:cont, count + 1}, else: {:halt, count}
    end)
  end

  # Check if next_num is the next sequential chapter after current_num (e.g., "2" -> "3" or "5.1" -> "5.2")
  defp is_next_sequential_chapter(current_num, next_num) do
    # Split both numbers by dots to get their parts
    current_parts = String.split(current_num, ".")
    next_parts = String.split(next_num, ".")

    # If they have different number of parts, they're not sequential at the same level
    if length(current_parts) != length(next_parts) do
      false
    else
      # Check if all parts except the last are identical
      {last_current, last_next} =
        if length(current_parts) > 1 do
          init_current = Enum.drop(current_parts, -1)
          init_next = Enum.drop(next_parts, -1)

          if init_current == init_next do
            {List.last(current_parts), List.last(next_parts)}
          else
            {nil, nil}
          end
        else
          {List.first(current_parts), List.first(next_parts)}
        end

      # If we could extract comparable parts
      if last_current != nil && last_next != nil do
        # Try to convert them to integers and check if they're sequential
        case {Integer.parse(last_current), Integer.parse(last_next)} do
          {{current_int, ""}, {next_int, ""}} ->
            next_int == current_int + 1

          _ ->
            false
        end
      else
        false
      end
    end
  end

  # Format chapter number from float to proper string format (e.g., 5.21 -> 5.2.1)
  defp format_chapter_number(num) when is_float(num) do
    # Convert float to string
    str_num = Float.to_string(num)

    # Check if it has decimal part that needs formatting
    if String.contains?(str_num, ".") do
      [whole, decimal] = String.split(str_num, ".")

      # If decimal part has multiple digits, insert dots
      formatted_decimal =
        if String.length(decimal) > 1 do
          decimal
          |> String.graphemes()
          |> Enum.join(".")
        else
          decimal
        end

      "#{whole}.#{formatted_decimal}"
    else
      str_num
    end
  end

  defp format_chapter_number(num) when is_integer(num) do
    Integer.to_string(num)
  end

  defp format_chapter_number(num) do
    "#{num}"
  end

  # Helper functions for saving datasets
  defp save_by_category(instructions, output_path) do
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

  defp split_and_save_datasets(instructions, output_path) do
    # Create the combined datasets directory if it doesn't exist
    combined_dir = Path.join(output_path, "combined_datasets")
    File.mkdir_p!(combined_dir)

    # Filter out instructions with excluded categories using @excluded_categories
    filtered_instructions =
      Enum.filter(instructions, fn {_, category} ->
        category not in @excluded_categories
      end)

    # Group filtered instructions by category
    grouped = Enum.group_by(filtered_instructions, fn {_instruction, category} -> category end)

    # Initialize empty lists for each split
    training_set = []
    validation_set = []
    test_set = []

    # Apply round-robin split with priority for training
    {final_training, final_validation, final_test} =
      Enum.reduce(grouped, {training_set, validation_set, test_set}, fn {category,
                                                                         cat_instructions},
                                                                        {train_acc, val_acc,
                                                                         test_acc} ->
        # Calculate split sizes
        total = length(cat_instructions)

        # Shuffle instructions
        shuffled = Enum.shuffle(cat_instructions)

        # Apply distribution
        {train, val, test} = distribute_items(shuffled, total)

        # Log the split
        IO.puts("Category: #{category}")
        IO.puts("  Total: #{total}")
        IO.puts("  Training: #{length(train)}")
        IO.puts("  Validation: #{length(val)}")
        IO.puts("  Test: #{length(test)}")

        # Add to accumulators
        {train_acc ++ train, val_acc ++ val, test_acc ++ test}
      end)

    # Count and log excluded items
    excluded_count = length(instructions) - length(filtered_instructions)

    if excluded_count > 0 do
      excluded_categories_str = Enum.join(@excluded_categories, ", ")

      IO.puts(
        "Excluded #{excluded_count} items with categories [#{excluded_categories_str}] from combined datasets"
      )
    end

    # Save each split to a JSONL file
    save_split_to_file(final_training, Path.join(combined_dir, "training_set.jsonl"))
    save_split_to_file(final_validation, Path.join(combined_dir, "validation_set.jsonl"))
    save_split_to_file(final_test, Path.join(combined_dir, "test_set.jsonl"))
  end

  defp distribute_items(items, total) do
    cond do
      total <= 0 ->
        {[], [], []}

      total == 1 ->
        {items, [], []}

      total == 2 ->
        [first, second] = items
        {[first], [second], []}

      total == 3 ->
        [first, second, third] = items
        {[first], [second], [third]}

      true ->
        # Calculate target proportions (80%, 15%, 5%)
        target_train = max(1, floor(total * 0.8))
        target_val = max(1, floor(total * 0.15))
        target_test = total - target_train - target_val

        # Ensure test set gets at least one item if possible
        target_test = max(1, target_test)

        # Adjust if needed
        target_train = total - target_val - target_test

        # Split the items
        {train_items, rest} = Enum.split(items, target_train)
        {val_items, test_items} = Enum.split(rest, target_val)

        {train_items, val_items, test_items}
    end
  end

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
