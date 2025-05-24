defmodule DP.ExtractFiles do
  # Use forward slashes for better cross-platform compatibility
  @db_path "N:/Thesis/Gutachten"
  @all_docm_files "N:/Thesis/data_prepare/docm_files"

  def extract_docm_files do
    # Create output directory if it doesn't exist
    File.mkdir_p!(@all_docm_files)

    # Log paths for debugging
    IO.puts("Searching in path: #{@db_path}")
    IO.puts("Output path: #{@all_docm_files}")

    # Check if base path exists and debug directory contents
    if !File.exists?(@db_path) do
      IO.puts("Warning: Base path #{@db_path} does not exist!")

      %{
        total_files_found: 0,
        files_copied: 0,
        failed_files: [],
        all_docm_files: @all_docm_files
      }
    else
      IO.puts("Base path exists. Checking directory contents:")

      # List top-level files to verify access
      case File.ls(@db_path) do
        {:ok, files} ->
          IO.puts("Total items in directory: #{length(files)}")

        {:error, reason} ->
          IO.puts("Error reading directory: #{inspect(reason)}")
      end

      files = Path.wildcard("#{@db_path}/**/*GA*01.docm")
      IO.puts("Found #{length(files)} files")

      if files != [] do
        IO.puts("First few matched files: #{inspect(Enum.take(files, 3))}")
      end

      # Copy each file to the output path
      results =
        Enum.map(files, fn file ->
          filename = Path.basename(file)
          target_path = Path.join(@all_docm_files, filename)

          IO.puts("Copying #{file} to #{target_path}")

          case File.cp(file, target_path) do
            :ok ->
              {:ok, filename}

            {:error, reason} ->
              IO.puts("Error copying #{filename}: #{inspect(reason)}")
              {:error, filename, reason}
          end
        end)

      # Count successes and failures
      successes =
        Enum.count(results, fn
          {:ok, _} -> true
          _ -> false
        end)

      failures =
        Enum.filter(results, fn
          {:error, _, _} -> true
          _ -> false
        end)

      # Return a summary
      %{
        total_files_found: length(files),
        files_copied: successes,
        failed_files: failures,
        all_docm_files: @all_docm_files
      }
    end
  end
end
