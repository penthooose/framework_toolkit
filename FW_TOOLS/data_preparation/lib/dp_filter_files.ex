defmodule DP.FilterFiles do
  @md_files "N:/Thesis/data_prepare/md_files"
  @filtered_md_files "N:/Thesis/data_prepare/filtered_md_files"
  @temp_path "N:/Thesis/temp"

  def filter_md_files do
    # Create directories if they don't exist
    File.mkdir_p!(@filtered_md_files)
    File.mkdir_p!(@temp_path)

    # Get all markdown files
    md_files = Path.wildcard(Path.join(@md_files, "*.md"))

    # Process each file
    Enum.each(md_files, fn file_path ->
      # Read first 20 lines of the file
      content =
        file_path
        |> File.stream!()
        |> Enum.take(20)
        |> Enum.join(" ")
        |> String.downcase()

      # Convert content to a version without spaces for comparison
      no_spaces_content = String.replace(content, " ", "")

      # Create space-tolerant patterns for matching
      gutachten_pattern = ~r/g\s*u\s*t\s*a\s*c\s*h\s*t\s*e\s*n/i

      ergaenzung_pattern =
        ~r/e\s*r\s*g\s*(ä|a\s*e)\s*n\s*z\s*u\s*n\s*g\s*s\s*g\s*u\s*t\s*a\s*c\s*h\s*t\s*e\s*n/i

      # Check if file contains target words
      contains_keywords =
        Regex.match?(gutachten_pattern, content) ||
          Regex.match?(ergaenzung_pattern, content) ||
          String.contains?(no_spaces_content, "gutachten") ||
          String.contains?(no_spaces_content, "ergänzungsgutachten") ||
          String.contains?(no_spaces_content, "ergaenzungsgutachten")

      # Get the file name
      file_name = Path.basename(file_path)

      # Determine destination and move the file
      destination =
        if contains_keywords do
          Path.join(@filtered_md_files, file_name)
        else
          Path.join(@temp_path, file_name)
        end

      File.cp!(file_path, destination)

      IO.puts(
        "Copied #{file_name} to #{if contains_keywords, do: "filtered files", else: "temp"}"
      )
    end)

    IO.puts("File filtering complete!")
  end
end
