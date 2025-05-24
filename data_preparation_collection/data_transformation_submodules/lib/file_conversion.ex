defmodule DP.ConvertFiles do
  @input_path "N:/Thesis/data_prepare/docm_files"
  @output_path "N:/Thesis/data_prepare/md_files"
  @pandoc_path "C:/Program Files/Pandoc/pandoc.exe"

  def convert_docm_to_md do
    # Create output directory if it doesn't exist
    File.mkdir_p!(@output_path)

    # Log paths for debugging
    IO.puts("Searching in path: #{@input_path}")
    IO.puts("Output path: #{@output_path}")
    IO.puts("Using Pandoc at: #{@pandoc_path}")

    # Check if Pandoc exists
    if !File.exists?(@pandoc_path) do
      IO.puts("Warning: Pandoc not found at #{@pandoc_path}!")
      {:error, "Pandoc executable not found"}
    else
      # Check if base path exists
      if !File.exists?(@input_path) do
        IO.puts("Warning: Base path #{@input_path} does not exist!")
        {:error, "Base path does not exist!"}
      else
        files = Path.wildcard("#{@input_path}/**/*.docm")
        IO.puts("Found #{length(files)} files")

        if files != [] do
          IO.puts("First matched file: #{inspect(Enum.take(files, 1))}")
        end

        # Convert each file to markdown format using Pandoc
        results =
          Enum.map(files, fn file ->
            filename = Path.basename(file, ".docm")
            target_path = Path.join(@output_path, "#{filename}.md")

            IO.puts("Converting #{file} to #{target_path}")

            case convert_with_pandoc(file, target_path) do
              :ok ->
                {:ok, filename}

              {:error, reason} ->
                IO.puts("Error converting #{filename}: #{inspect(reason)}")
                {:error, filename, reason}
            end
          end)

        # Count successes and failures
        {:ok,
         %{
           total_files_found: length(files),
           files_converted:
             Enum.count(results, fn
               {:ok, _} -> true
               _ -> false
             end),
           failed_files:
             Enum.filter(results, fn
               {:error, _, _} -> true
               _ -> false
             end),
           output_path: @output_path
         }}
      end
    end
  end

  # Convert DOCM file to Markdown using Pandoc
  defp convert_with_pandoc(source_path, target_path) do
    try do
      # Convert paths to Windows backslash format
      source_win_path = String.replace(source_path, "/", "\\")
      target_win_path = String.replace(target_path, "/", "\\")
      pandoc_win_path = String.replace(@pandoc_path, "/", "\\")

      IO.puts("  Source: #{source_win_path}")
      IO.puts("  Target: #{target_win_path}")
      IO.puts("  Pandoc: #{pandoc_win_path}")

      # Try direct command execution
      case System.cmd(
             pandoc_win_path,
             [
               "-f",
               "docx",
               "-t",
               "markdown",
               "-o",
               target_win_path,
               source_win_path
             ],
             stderr_to_stdout: true
           ) do
        {output, 0} ->
          IO.puts("Conversion successful")
          :ok

        {output, exit_code} ->
          IO.puts("Pandoc direct call failed with exit code #{exit_code}: #{output}")

          # If direct call fails, try through cmd.exe
          cmd =
            "\"#{pandoc_win_path}\" -f docx -t markdown -o \"#{target_win_path}\" \"#{source_win_path}\""

          IO.puts("Trying through cmd.exe: #{cmd}")

          case System.cmd("cmd.exe", ["/c", cmd], stderr_to_stdout: true) do
            {cmd_output, 0} ->
              IO.puts("Conversion through cmd.exe successful")
              :ok

            {cmd_output, cmd_exit_code} ->
              IO.puts("Cmd execution failed with exit code #{cmd_exit_code}: #{cmd_output}")
              {:error, "Pandoc conversion failed: #{cmd_output}"}
          end
      end
    rescue
      e ->
        IO.puts("Error during conversion: #{inspect(e)}")
        {:error, e}
    end
  end
end
