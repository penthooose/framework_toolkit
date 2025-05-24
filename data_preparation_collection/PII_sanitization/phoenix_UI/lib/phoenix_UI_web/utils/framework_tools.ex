defmodule PhoenixUIWeb.Utils.FrameworkTools do
  def conv_file_to_txt(input_path, output_path) do
    # Remove the automatic extension changing - let the caller handle it
    input = check_win_or_unix_path(input_path)
    output = check_win_or_unix_path(output_path)

    IO.puts("Converting Word to TXT...")
    IO.puts("Input path: #{input_path}")
    IO.puts("Output path: #{output_path}")

    case validate_file_type(input) do
      :ok ->
        case convert_file(input, output) do
          {:ok, _} ->
            IO.puts("Conversion successful")
            {:ok, "Conversion successful"}

          {:error, error} ->
            IO.puts("Conversion failed: #{error}")
            {:error, "Conversion failed: #{error}"}
        end

      {:error, error} ->
        IO.puts("Invalid file type: #{error}")
        {:error, "Invalid file type: #{error}"}
    end
  end

  def conv_all_files_in_dir_to_txt(input_path, output_path, type) do
    input = check_win_or_unix_path(input_path)
    output = check_win_or_unix_path(output_path)

    IO.puts("Converting files in directory: #{input}")
    IO.puts("Selected type: #{type}")

    # Use normalized path for search
    search_dir = Path.absname(input)
    IO.puts("Search directory: #{search_dir}")

    # Get all files recursively first
    all_files = Path.wildcard(Path.join(search_dir, "**/*"))

    # Filter by extension
    valid_extensions =
      case String.downcase(type) do
        "pdf" -> [".pdf"]
        "docx" -> [".doc", ".docx"]
        "pdf_docx" -> [".pdf", ".doc", ".docx"]
      end

    files =
      all_files
      |> Enum.filter(&File.regular?/1)
      |> Enum.filter(fn file ->
        ext = String.downcase(Path.extname(file))
        ext in valid_extensions
      end)

    IO.puts("Found files: #{inspect(files)}")
    IO.puts("Total files found: #{length(files)}")

    case files do
      [] ->
        {:error, "No files found to convert"}

      files ->
        results =
          Enum.map(files, fn file ->
            relative_path = Path.relative_to(file, input)
            out_dir = Path.join(output, Path.dirname(relative_path))
            File.mkdir_p!(out_dir)
            out_file = Path.join(out_dir, Path.basename(file, Path.extname(file)) <> ".txt")
            IO.puts("Converting file: #{file} to #{out_file}")
            conv_file_to_txt(file, out_file)
          end)

        # Check if any conversion was successful
        successful =
          Enum.count(results, fn
            {:ok, _} -> true
            _ -> false
          end)

        if successful > 0 do
          {:ok, "Successfully converted #{successful} files"}
        else
          {:error, "Failed to convert any files"}
        end
    end
  end

  def conv_all_files_to_single_dir(input_path, output_path, type) do
    input = check_win_or_unix_path(input_path)

    # Create flat output directory
    flat_output_dir = Path.join(output_path, "converted_files")
    File.mkdir_p!(flat_output_dir)

    # Get all files recursively
    all_files = Path.wildcard(Path.join(input, "**/*"))

    # Filter valid files
    valid_files =
      all_files
      |> Enum.filter(&File.regular?/1)
      |> Enum.filter(fn file ->
        ext = String.downcase(Path.extname(file))

        case String.downcase(type) do
          "pdf" -> ext == ".pdf"
          "docx" -> ext in [".doc", ".docx"]
          "pdf_docx" -> ext in [".pdf", ".doc", ".docx"]
        end
      end)

    case valid_files do
      [] ->
        {:error, "No valid files found to convert"}

      files ->
        results =
          Enum.map(files, fn file ->
            # Use unique names to avoid conflicts
            base_name = Path.basename(file, Path.extname(file))
            unique_name = "#{base_name}_#{:rand.uniform(10000)}.txt"
            output_file = Path.join(flat_output_dir, unique_name)

            conv_file_to_txt(file, output_file)
          end)

        successful = Enum.count(results, &match?({:ok, _}, &1))

        if successful > 0 do
          {:ok, "Successfully converted #{successful} files"}
        else
          {:error, "Failed to convert any files"}
        end
    end
  end

  def check_win_or_unix_path(path) when is_binary(path) do
    case :os.type() do
      {:unix, _} ->
        String.replace(path, "\\", "/")

      {:win32, _} ->
        String.replace(path, "/", "\\")
    end
  end

  defp convert_file(input, output) do
    case String.downcase(Path.extname(input)) do
      ext when ext in [".docx", ".doc"] ->
        try do
          output_dir = Path.dirname(output)
          File.mkdir_p!(output_dir)

          case System.cmd(libreoffice_command(), [
                 "--headless",
                 "--convert-to",
                 "txt",
                 input,
                 "--outdir",
                 output_dir
               ]) do
            {_, 0} ->
              # Ensure the output file has the correct name
              output_file = Path.join(output_dir, "#{Path.basename(input, ext)}.txt")

              if File.exists?(output_file) do
                File.rename!(output_file, output)
                IO.puts("Conversion successful")
                IO.puts("Output file exists: #{File.exists?(output)}")
                {:ok, output}
              else
                {:error, "Output file not found after conversion"}
              end

            {error, _} ->
              {:error, error}
          end
        rescue
          e in ErlangError -> {:error, Exception.message(e)}
        end

      ".pdf" ->
        try do
          case System.cmd(pdf_command(), [input, output]) do
            {_, 0} -> {:ok, output}
            {error, _} -> {:error, error}
          end
        rescue
          e in ErlangError -> {:error, Exception.message(e)}
        end

      ext ->
        {:error, "Unsupported file type: #{ext}"}
    end
  end

  defp libreoffice_command do
    case :os.type() do
      {:unix, _} ->
        System.find_executable("libreoffice") || raise "LibreOffice not found"

      {:win32, _} ->
        # Search in common installation paths
        paths = [
          "C:/Program Files/LibreOffice/program/soffice.exe",
          "C:/Program Files (x86)/LibreOffice/program/soffice.exe",
          System.find_executable("soffice"),
          System.find_executable("soffice.exe")
        ]

        Enum.find(paths, fn path ->
          path && File.exists?(path)
        end) ||
          raise "LibreOffice not found. Please ensure LibreOffice is installed and 'soffice.exe' is in your PATH"
    end
  end

  def cleanup_files_input_output(input_path, output_path) do
    try do
      # Clean up input files
      if input_path && File.exists?(input_path) do
        if File.dir?(input_path) do
          File.rm_rf(input_path)
        else
          File.rm(input_path)
        end
      end

      # Clean up output files
      if output_path && File.exists?(output_path) do
        if File.dir?(output_path) do
          File.rm_rf(output_path)
        else
          File.rm(output_path)
        end
      end

      :ok
    rescue
      e in File.Error ->
        IO.puts("Warning: Could not cleanup files: #{Exception.message(e)}")
        {:error, "Could not cleanup files"}
    end
  end

  def cleanup_all_files(input_dir, output_dir) do
    try do
      # Only remove contents of directories, not the directories themselves
      if File.exists?(input_dir) do
        input_dir
        |> File.ls!()
        |> Enum.each(fn file ->
          path = Path.join(input_dir, file)
          if File.exists?(path), do: File.rm_rf!(path)
        end)
      end

      if File.exists?(output_dir) do
        output_dir
        |> File.ls!()
        |> Enum.each(fn file ->
          path = Path.join(output_dir, file)
          if File.exists?(path), do: File.rm_rf!(path)
        end)
      end

      # Ensure directories exist
      File.mkdir_p!(input_dir)
      File.mkdir_p!(output_dir)
      :ok
    rescue
      e in File.Error ->
        IO.puts("Warning: Could not cleanup directories: #{Exception.message(e)}")
        {:error, "Could not cleanup directories"}
    end
  end

  defp pdf_command do
    case :os.type() do
      {:unix, _} -> "pdftotext"
      {:win32, _} -> "pdftotext.exe"
    end
  end

  defp validate_file_type(file_path) do
    case String.downcase(Path.extname(file_path)) do
      ext when ext in [".docx", ".doc", ".pdf"] -> :ok
      ext -> {:error, "Unsupported file type: #{ext}"}
    end
  end

  def filter_and_clean_directory(dir_path, type) do
    # Get valid extensions based on type
    valid_extensions =
      case String.downcase(type) do
        "pdf" -> [".pdf"]
        "docx" -> [".doc", ".docx"]
        "pdf_docx" -> [".pdf", ".doc", ".docx"]
        "txt" -> [".txt"]
      end

    # Find all files recursively using proper pattern
    search_pattern =
      case :os.type() do
        {:win32, _} -> Path.join([dir_path, "**", "*.*"])
        {:unix, _} -> Path.join(dir_path, "**/*.*")
      end

    # Get all files
    all_files = Path.wildcard(search_pattern)

    # Filter and delete invalid files
    {valid_files, invalid_files} =
      all_files
      |> Enum.filter(&File.regular?/1)
      |> Enum.split_with(fn file ->
        String.downcase(Path.extname(file)) in valid_extensions
      end)

    # Delete invalid files
    Enum.each(invalid_files, &File.rm!/1)

    # Remove empty directories
    clean_empty_directories(dir_path)

    if valid_files == [] do
      {:error, "No valid files found for type #{type}"}
    else
      {:ok, valid_files}
    end
  end

  defp clean_empty_directories(dir_path) do
    # Use proper pattern for directory search
    pattern =
      case :os.type() do
        {:win32, _} -> Path.join([dir_path, "**", "*"])
        {:unix, _} -> Path.join(dir_path, "**/*")
      end

    pattern
    |> Path.wildcard()
    |> Enum.filter(&File.dir?/1)
    |> Enum.sort_by(&String.length/1, :desc)
    |> Enum.each(fn dir ->
      if dir != dir_path and File.ls!(dir) == [] do
        File.rmdir!(dir)
      end
    end)
  end

  def create_zip_archive(dir_path, zip_path) do
    try do
      case :os.type() do
        {:unix, _} ->
          {_, 0} = System.cmd("zip", ["-r", zip_path, "."], cd: dir_path)
          :ok

        {:win32, _} ->
          seven_zip_path = find_7zip()
          {_, 0} = System.cmd(seven_zip_path, ["a", "-tzip", zip_path, "*"], cd: dir_path)
          :ok
      end
    rescue
      e ->
        IO.puts("Error creating ZIP: #{Exception.message(e)}")
        {:error, Exception.message(e)}
    end
  end

  def find_7zip do
    common_paths = [
      "C:/Program Files/7-Zip/7z.exe",
      "C:/Program Files (x86)/7-Zip/7z.exe",
      System.find_executable("7z"),
      System.find_executable("7z.exe")
    ]

    case Enum.find(common_paths, &(not is_nil(&1) and File.exists?(&1))) do
      nil -> raise "ZIP utility not found"
      path -> path
    end
  end

  def get_next_available_name(base_name, taken_names, dir) do
    Enum.reduce_while(Stream.iterate(1, &(&1 + 1)), {nil, taken_names}, fn i, {_, names} ->
      candidate =
        if i == 1,
          do: Path.join(dir, "#{base_name}.txt"),
          else: Path.join(dir, "#{base_name}_#{i}.txt")

      if candidate not in names do
        {:halt, {candidate, MapSet.put(names, candidate)}}
      else
        {:cont, {nil, names}}
      end
    end)
  end
end
