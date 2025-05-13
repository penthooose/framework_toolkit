defmodule Phoenix_UIWeb.Tools do
  @moduledoc """
    IDEA: If other modules also need "Tools" functions, consider moving them to a separate project.
  """
  use Phoenix_UIWeb, :live_view
  alias PhoenixUIWeb.Utils.FrameworkTools

  @input_dir "files/conversion_input"
  @output_dir "files/conversion_output"
  @priv_dir :phoenix_UI |> :code.priv_dir() |> to_string()
  @downloads_dir Path.join([@priv_dir, "static", "downloads"])

  @impl true
  def mount(_params, _session, socket) do
    # Create necessary directories at mount with absolute paths
    input_dir = Path.absname(@input_dir)
    output_dir = Path.absname(@output_dir)
    downloads_dir = Path.absname(@downloads_dir)

    File.mkdir_p!(input_dir)
    File.mkdir_p!(output_dir)
    File.mkdir_p!(downloads_dir)

    {:ok,
     socket
     |> assign(
       mode_single_file: true,
       selected_input_type: "DOCX",
       input_path: nil,
       flash_message: nil,
       flash_type: nil,
       flash_timer: nil,
       download_path: nil,
       is_uploading: false,
       is_processing: false,
       upload_complete: false,
       show_download_section: false,
       file_accept: ".doc,.docx",
       current_upload_dir: nil,
       processed_files: 0,
       total_files: 0,
       retain_structure: true
     )
     |> allow_upload(:directory,
       accept: [".zip"],
       max_entries: 1,
       max_file_size: 30_000_000_000,
       auto_upload: true,
       progress: &handle_progress/3
     )
     |> allow_upload(:file,
       accept: [".pdf", ".doc", ".docx"],
       max_entries: 1,
       max_file_size: 30_000_000_000,
       auto_upload: true,
       progress: &handle_progress/3,
       validate: &validate_file/2
     )}
  end

  @impl true
  def handle_event("validate", _params, socket) do
    # Reset all conversion and upload states
    cleanup_conversion_directories()

    {:noreply,
     socket
     |> assign(
       input_path: nil,
       upload_complete: false,
       show_download_section: false,
       download_path: nil,
       is_processing: false
     )}
  end

  def handle_event("cancel-upload", %{"ref" => ref}, socket) do
    {:noreply, cancel_upload(socket, :file, ref)}
  end

  def handle_event("toggle_mode", _params, socket) do
    case cleanup_conversion_directories() do
      :ok ->
        {:noreply,
         assign(socket,
           mode_single_file: !socket.assigns.mode_single_file,
           input_path: nil,
           output_path: nil,
           show_download_section: false
         )}

      {:error, msg} ->
        {:noreply,
         socket
         |> put_timed_flash(:error, "Error: #{msg}")
         |> assign(is_processing: false)}
    end
  end

  def handle_event("select_input_type", %{"type" => type}, socket) do
    IO.puts("Selected input type: #{type}")

    file_accept =
      case type do
        "PDF" -> ".pdf"
        "DOCX" -> ".doc,.docx"
        "PDF_DOCX" -> ".pdf,.doc,.docx"
      end

    socket =
      socket
      |> assign(selected_input_type: type, file_accept: file_accept)
      |> allow_upload(:file,
        accept:
          case type do
            "PDF" -> [".pdf"]
            "DOCX" -> [".doc", ".docx"]
            "PDF_DOCX" -> [".pdf", ".doc", ".docx"]
          end,
        max_entries: 1,
        max_file_size: 300_000_000,
        auto_upload: true,
        progress: &handle_progress/3,
        validate: &validate_file/2
      )
      |> allow_upload(:directory,
        accept: [".zip"],
        max_entries: 1,
        max_file_size: 300_000_000,
        auto_upload: true,
        progress: &handle_progress/3
      )

    {:noreply, socket}
  end

  # Handle file upload result
  def handle_event("set_input_file_path", %{"file" => file}, socket) when is_map(file) do
    IO.inspect(file, label: "Received file data")

    socket = assign(socket, is_uploading: true, upload_complete: false)

    if file["path"] do
      {:noreply,
       socket
       |> assign(input_path: file["path"])
       |> assign(is_uploading: false)
       |> assign(upload_complete: true)}
    else
      {:noreply,
       socket
       |> assign(is_uploading: false)
       |> put_timed_flash(:error, "Upload failed", 5000)}
    end
  end

  # Handle directory selection
  def handle_event("set_input_directory_path", %{"directory" => files} = params, socket) do
    IO.inspect(files, label: "Received files")

    try do
      # Create a unique directory name
      dir_name = "upload_#{System.system_time(:second)}"
      input_dir = Path.join(@input_dir, dir_name)

      # Create base directory
      File.mkdir_p!(input_dir)

      # Track upload progress
      socket = assign(socket, is_uploading: true)

      # Process each file, maintaining directory structure
      files
      |> Enum.filter(fn file -> is_map(file) end)
      |> Enum.each(fn file ->
        IO.inspect(file, label: "Processing file")

        try do
          relative_path =
            cond do
              is_binary(file["webkitRelativePath"]) and file["webkitRelativePath"] != "" ->
                file["webkitRelativePath"]

              is_binary(file["relativePath"]) and file["relativePath"] != "" ->
                file["relativePath"]

              true ->
                file["name"]
            end

          if relative_path && file["path"] do
            dest_path = Path.join(input_dir, relative_path)
            File.mkdir_p!(Path.dirname(dest_path))
            contents = File.read!(file["path"])
            File.write!(dest_path, contents)
            IO.puts("Copied file to: #{dest_path}")
          end
        rescue
          e ->
            IO.puts("Error processing file: #{inspect(e)}")
        end
      end)

      {:noreply,
       socket
       |> assign(input_path: input_dir)
       |> assign(is_uploading: false)
       |> assign(upload_complete: true)
       |> assign(show_download_section: false)
       |> put_timed_flash(:info, "Directory uploaded successfully")}
    rescue
      e ->
        IO.puts("Directory upload error: #{inspect(e)}")

        {:noreply,
         socket
         |> assign(is_uploading: false)
         |> put_timed_flash(:error, "Failed to upload directory: #{Exception.message(e)}")}
    end
  end

  def handle_event("start_conversion", _params, socket) do
    socket =
      socket
      |> assign(is_processing: true)
      |> assign(show_download_section: false)

    send(self(), :do_conversion)
    {:noreply, socket}
  end

  def handle_info(:do_conversion, socket) do
    try do
      {result, output_path} =
        if socket.assigns.mode_single_file do
          input_file = socket.assigns.input_path
          input_filename = Path.basename(input_file)
          base_name = Path.basename(input_filename, Path.extname(input_filename))
          output_filename = "#{base_name}.txt"

          # Save to downloads directory
          output_file = Path.join(@downloads_dir, output_filename)
          File.mkdir_p!(Path.dirname(output_file))

          {FrameworkTools.conv_file_to_txt(input_file, output_file), output_file}
        else
          input_dir = socket.assigns.input_path
          output_dir = String.replace(input_dir, "conversion_input", "conversion_output")
          File.mkdir_p!(output_dir)

          conversion_result =
            if socket.assigns.retain_structure do
              case FrameworkTools.conv_all_files_in_dir_to_txt(
                     input_dir,
                     output_dir,
                     socket.assigns.selected_input_type
                   ) do
                {:ok, msg} ->
                  # Create ZIP of the output directory
                  zip_name = "converted_files.zip"
                  zip_path = Path.join(@downloads_dir, zip_name)
                  File.mkdir_p!(@downloads_dir)

                  if File.exists?(zip_path), do: File.rm!(zip_path)

                  case FrameworkTools.create_zip_archive(output_dir, zip_path) do
                    :ok -> {:ok, zip_path}
                    error -> error
                  end

                error ->
                  error
              end
            else
              # First check if there are valid files and copy them to output directory
              case FrameworkTools.filter_and_clean_directory(
                     input_dir,
                     socket.assigns.selected_input_type
                   ) do
                {:ok, []} ->
                  {:error, "No valid files found to convert"}

                {:ok, valid_files} ->
                  IO.puts("Found #{length(valid_files)} valid files to process")

                  # Create converted_files directory for output
                  converted_dir = Path.join(output_dir, "converted_files")
                  File.mkdir_p!(converted_dir)

                  # First step: Generate all unique filenames and track taken names
                  {file_mappings, _taken_names} =
                    Enum.reduce(valid_files, {[], MapSet.new()}, fn file,
                                                                    {mappings, taken_names} ->
                      base_name = Path.basename(file, Path.extname(file))

                      # Find the next available name that hasn't been taken
                      {output_file, new_taken_names} =
                        FrameworkTools.get_next_available_name(
                          base_name,
                          taken_names,
                          converted_dir
                        )

                      {[{file, output_file} | mappings], new_taken_names}
                    end)

                  # Second step: Perform all conversions
                  result =
                    Enum.reduce_while(file_mappings, {:ok, []}, fn {input_file, output_file},
                                                                   {:ok, acc} ->
                      case FrameworkTools.conv_file_to_txt(input_file, output_file) do
                        {:ok, _} -> {:cont, {:ok, [output_file | acc]}}
                        error -> {:halt, error}
                      end
                    end)

                  case result do
                    {:ok, converted_files} when length(converted_files) > 0 ->
                      # Create ZIP directly from converted_files directory
                      zip_name = "converted_files.zip"
                      zip_path = Path.join(@downloads_dir, zip_name)
                      File.mkdir_p!(@downloads_dir)

                      if File.exists?(zip_path), do: File.rm!(zip_path)

                      case FrameworkTools.create_zip_archive(converted_dir, zip_path) do
                        # Changed from {{:ok, msg}, path} to {:ok, path}
                        :ok -> {:ok, zip_path}
                        error -> error
                      end

                    {:ok, []} ->
                      {:error, "No files were converted"}

                    error ->
                      error
                  end

                {:error, reason} ->
                  {:error, reason}
              end
            end

          case conversion_result do
            # Changed to match new return format
            {:ok, zip_path} ->
              # Create ZIP of converted files directly from conversion result
              {{:ok, "Conversion successful"}, zip_path}

            {:error, reason} ->
              {{:error, reason}, nil}
          end
        end

      case result do
        {:ok, _} ->
          download_path =
            if socket.assigns.mode_single_file do
              "/downloads/#{Path.basename(output_path)}"
            else
              # Remove potential double .zip extension
              base_name =
                Path.basename(output_path)
                |> String.replace(~r/\.zip\.zip$/, ".zip")
                |> String.replace(~r/\.zip$/, ".zip")

              "/downloads/#{base_name}"
            end

          Process.send_after(self(), {:cleanup_download, output_path}, :timer.minutes(5))

          {:noreply,
           socket
           |> assign(is_processing: false)
           |> assign(download_path: download_path)
           |> assign(upload_complete: false)
           |> assign(show_download_section: true)
           |> put_timed_flash(:info, "Conversion complete! Click the download link to download.")}

        {:error, msg} ->
          FrameworkTools.cleanup_files_input_output(socket.assigns.input_path, output_path)

          {:noreply,
           socket
           |> assign(is_processing: false)
           |> put_timed_flash(:error, "Conversion failed: #{msg}")}
      end
    rescue
      e ->
        FrameworkTools.cleanup_files_input_output(socket.assigns.input_path, nil)

        {:noreply,
         socket
         |> assign(is_processing: false)
         |> put_timed_flash(:error, "Error: #{Exception.message(e)}")}
    end
  end

  def handle_event("toggle_structure", _params, socket) do
    {:noreply, assign(socket, retain_structure: !socket.assigns.retain_structure)}
  end

  def handle_info(:clear_flash, socket) do
    {:noreply,
     socket
     |> assign(:flash_timer, nil)
     |> assign(flash_message: nil, flash_type: nil)}
  end

  def handle_info(:cleanup_files, socket) do
    FrameworkTools.cleanup_files_input_output(
      socket.assigns.input_path,
      socket.assigns.output_path
    )

    {:noreply, socket}
  end

  def handle_info({:cleanup_download, file_path}, socket) do
    if File.exists?(file_path), do: File.rm!(file_path)
    {:noreply, socket}
  end

  # Handle progress for both single files and directory uploads
  defp handle_progress(:directory, entry, socket) do
    if entry.done? do
      try do
        cleanup_conversion_directories()

        # Get original directory name from ZIP
        dir_name = Path.basename(entry.client_name, ".zip")
        input_dir = Path.absname(Path.join(@input_dir, dir_name))
        File.mkdir_p!(input_dir)

        consume_uploaded_entry(socket, entry, fn %{path: temp_path} ->
          IO.puts("Processing ZIP file: #{temp_path}")

          case extract_zip(temp_path, input_dir) do
            :ok ->
              IO.puts("ZIP extracted successfully to: #{input_dir}")

              case FrameworkTools.filter_and_clean_directory(
                     input_dir,
                     socket.assigns.selected_input_type
                   ) do
                {:ok, valid_files} ->
                  IO.puts("Found #{length(valid_files)} valid files")
                  {:ok, input_dir}

                {:error, reason} ->
                  {:error, reason}
              end

            {:error, reason} ->
              {:error, reason}
          end
        end)

        {:noreply,
         socket
         |> assign(input_path: input_dir)
         |> assign(is_uploading: false)
         |> assign(upload_complete: true)}
      rescue
        e ->
          cleanup_conversion_directories()
          IO.puts("Error in handle_progress: #{Exception.message(e)}")

          {:noreply,
           socket
           |> assign(is_uploading: false)
           |> assign(upload_complete: false)
           |> put_timed_flash(:error, "Error processing ZIP: #{Exception.message(e)}")}
      end
    else
      {:noreply, assign(socket, is_uploading: true)}
    end
  end

  defp handle_progress(:file, entry, socket) do
    if entry.done? do
      # Clean up existing files first
      cleanup_conversion_directories()

      dest_path = Path.join(@input_dir, entry.client_name)

      # Ensure parent directory exists
      File.mkdir_p!(Path.dirname(dest_path))

      try do
        consume_uploaded_entry(socket, entry, fn %{path: path} ->
          File.cp!(path, dest_path)
          {:ok, dest_path}
        end)

        {:noreply,
         socket
         |> assign(input_path: dest_path)
         |> assign(is_uploading: false)
         |> assign(upload_complete: true)
         |> assign(show_download_section: false)}
      rescue
        e ->
          {:noreply,
           socket
           |> assign(is_uploading: false)
           |> put_timed_flash(:error, "Upload error: #{Exception.message(e)}")}
      end
    else
      {:noreply, assign(socket, is_uploading: true)}
    end
  end

  defp put_timed_flash(socket, type, message, link_type \\ nil) do
    if socket.assigns.flash_timer, do: Process.cancel_timer(socket.assigns.flash_timer)
    timer_ref = Process.send_after(self(), :clear_flash, 5000)

    socket
    |> assign(:flash_timer, timer_ref)
    |> assign(:flash_message, message)
    |> assign(:flash_type, type)
    |> assign(:flash_link_type, link_type)
  end

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup_files, 5000)
  end

  defp validate_file(_upload, socket) do
    if socket.assigns.mode_single_file do
      case socket.assigns.selected_input_type do
        "DOCX" -> [extension: ~w(.doc .docx)]
        "PDF" -> [extension: ~w(.pdf)]
        "PDF_DOCX" -> [extension: ~w(.pdf .doc .docx)]
      end
    else
      []
    end
  end

  defp validate_directory(_entry, _socket) do
    # Accept all ZIP files, we'll filter contents after extraction
    []
  end

  defp cleanup_conversion_directories do
    case FrameworkTools.cleanup_all_files(@input_dir, @output_dir) do
      :ok -> :ok
      {:error, msg} -> {:error, msg}
    end
  end

  # Add helper function to sanitize directory names
  defp sanitize_directory_name(name) when is_binary(name) do
    name
    # Remove special characters
    |> String.replace(~r/[^\w\s-]/, "")
    |> String.trim()
    # Replace spaces with underscores
    |> String.replace(~r/\s+/, "_")
  end

  defp sanitize_directory_name(_), do: nil

  defp get_unique_filename(base_name, ext, dir) do
    Enum.reduce_while(2..1000, Path.join(dir, "#{base_name}#{ext}"), fn i, acc ->
      if File.exists?(acc) do
        new_name = Path.join(dir, "#{base_name}_#{i}#{ext}")
        {:cont, new_name}
      else
        {:halt, acc}
      end
    end)
  end

  defp extract_zip(zip_path, dest_path) do
    try do
      IO.puts("Extracting ZIP from #{zip_path} to #{dest_path}")

      case :os.type() do
        {:unix, _} ->
          case System.cmd("unzip", ["-o", zip_path, "-d", dest_path]) do
            {output, 0} ->
              IO.puts("Unzip successful: #{output}")
              :ok

            {error, code} ->
              IO.puts("Unzip failed with code #{code}: #{error}")
              {:error, error}
          end

        {:win32, _} ->
          seven_zip_path = FrameworkTools.find_7zip()
          IO.puts("Using 7-Zip at: #{seven_zip_path}")

          case System.cmd(seven_zip_path, ["x", "-y", "-o#{dest_path}", zip_path]) do
            {output, 0} ->
              IO.puts("7-Zip extraction successful: #{output}")
              :ok

            {error, code} ->
              IO.puts("7-Zip extraction failed with code #{code}: #{error}")
              {:error, error}
          end
      end
    rescue
      e ->
        IO.puts("Error in extract_zip: #{Exception.message(e)}\n#{Exception.format_stacktrace()}")
        {:error, Exception.message(e)}
    end
  end
end
