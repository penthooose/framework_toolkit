defmodule SE.MetaDataProcessor do
  @moduledoc """
  Detects existence of extracted metadata in each subchapter and appends information to extracted_summary file.
  """
  @report_dir "N:/Thesis/data_prepare/information_extraction/symbolic_checked_meta_data"

  def extract_metadata_from_subchapters do
    # Get all directories in the report directory
    case File.ls(@report_dir) do
      {:ok, dirs} ->
        Enum.each(dirs, fn dir ->
          parent_dir = Path.join(@report_dir, dir)
          process_directory(parent_dir)
        end)

      {:error, reason} ->
        IO.puts("Error reading directory #{@report_dir}: #{reason}")
    end
  end

  defp process_directory(parent_dir) do
    # Check if this is a directory
    if File.dir?(parent_dir) do
      # Path to the single_chapters folder
      single_chapters_dir = Path.join(parent_dir, "single_chapters")

      # Path to the meta info file
      meta_info_file = Path.join(parent_dir, "extracted_meta_info.json")

      # Path to the summary file
      summary_file = Path.join(parent_dir, "extracted_summary.json")

      if File.exists?(meta_info_file) && File.dir?(single_chapters_dir) do
        # Read meta info file
        case File.read(meta_info_file) do
          {:ok, meta_info_content} ->
            meta_info = Jason.decode!(meta_info_content)

            # Read summary file if it exists, otherwise create empty summary
            summary =
              if File.exists?(summary_file) do
                case File.read(summary_file) do
                  {:ok, summary_content} -> Jason.decode!(summary_content)
                  {:error, _} -> %{}
                end
              else
                %{}
              end

            # Process each markdown file in single_chapters directory
            case File.ls(single_chapters_dir) do
              {:ok, files} ->
                # Create an accumulator to collect all changes
                updated_summary =
                  Enum.reduce(files, summary, fn file, acc_summary ->
                    if String.ends_with?(file, ".md") do
                      md_file_path = Path.join(single_chapters_dir, file)
                      process_md_file(md_file_path, file, meta_info, acc_summary)
                    else
                      acc_summary
                    end
                  end)

                # Write updated summary back to file only once at the end
                File.write!(summary_file, Jason.encode!(updated_summary, pretty: true))

              {:error, reason} ->
                IO.puts("Error reading directory #{single_chapters_dir}: #{reason}")
            end

          {:error, reason} ->
            IO.puts("Error reading meta info file #{meta_info_file}: #{reason}")
        end
      end
    end
  end

  defp process_md_file(md_file_path, filename, meta_info, summary) do
    case File.read(md_file_path) do
      {:ok, content} ->
        lowercase_content = String.downcase(content)

        # Find metadata matches in the content
        matches = find_metadata_matches(lowercase_content, meta_info)

        # Update summary with matches
        update_summary(summary, filename, matches)

      {:error, reason} ->
        IO.puts("Error reading markdown file #{md_file_path}: #{reason}")
        summary
    end
  end

  defp find_metadata_matches(lowercase_content, meta_info) do
    Enum.reduce(meta_info, %{}, fn {key, value}, acc ->
      if is_binary(value) do
        # Convert value to lowercase
        lowercase_value = String.downcase(value)

        # First check if the full lowercase value is contained in the content
        if String.contains?(lowercase_content, lowercase_value) do
          # If the full value is found, it's a direct hit
          Map.put(acc, key, value)
        else
          # If not found, continue with word-by-word matching
          # Split into words
          words =
            lowercase_value
            |> String.split(~r/[\n,;\s\t]+/, trim: true)
            |> Enum.filter(fn word ->
              # Filter out single special characters and ensure word has content
              String.length(word) > 0 &&
                !(String.length(word) == 1 && !String.match?(word, ~r/^[a-zäöüßéèêëàáâçñ]$/u))
            end)

          # Count how many words are contained in the content
          matches =
            Enum.count(words, fn word ->
              # Check if the full word is contained in content
              full_match = String.contains?(lowercase_content, word)

              # Only allow slicing for alphabetical words (including German umlauts) of at least 5 chars
              can_slice =
                String.length(word) >= 5 && String.match?(word, ~r/^[a-zäöüßéèêëàáâçñ]+$/u)

              if full_match do
                true
              else
                if can_slice do
                  # Calculate max chars to slice (up to 25% of string length)
                  slice_amount = max(1, trunc(String.length(word) * 0.25))
                  sliced_word = String.slice(word, 0..(String.length(word) - slice_amount - 1))

                  # Check if the sliced word matches
                  String.contains?(lowercase_content, sliced_word)
                else
                  false
                end
              end
            end)

          # Calculate match percentage
          match_percentage = if length(words) > 0, do: matches / length(words), else: 0

          # If more than 50% of words match, add to results
          if match_percentage > 0.5 do
            Map.put(acc, key, value)
          else
            acc
          end
        end
      else
        acc
      end
    end)
  end

  defp update_summary(summary, filename, matches) do
    # Get or create the entry for this filename
    file_entry =
      Map.get(summary, filename, %{
        "included_meta_data" => %{},
        "summary" => ""
      })

    # Update included_meta_data while preserving other fields
    updated_file_entry = Map.put(file_entry, "included_meta_data", matches)

    # Update the summary with the new file entry
    Map.put(summary, filename, updated_file_entry)
  end
end
