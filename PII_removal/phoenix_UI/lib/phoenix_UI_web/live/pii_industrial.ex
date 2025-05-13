defmodule Phoenix_UIWeb.PIIIndustrial do
  use Phoenix_UIWeb, :live_view
  alias Phoenix_UI.State.PIIState
  alias PhoenixUIWeb.PIIHelpers
  alias PhoenixUIWeb.Utils.FrameworkTools
  use Phoenix_UIWeb.Components.CustomRecognizerBackend
  use Phoenix_UIWeb.Components.GeneralFunctionalities
  use Phoenix_UIWeb.Components.LabelsAndLabelSetsBackend
  import Phoenix_UIWeb.Components.CustomRecognizerForm
  import Phoenix_UIWeb.Components.LabelsAndLabelSetsForm
  import PhoenixUIWeb.PIIHelpers

  @protect_input_dir "files/protect_input"
  @protect_output_dir "files/protect_output"
  @priv_dir :phoenix_UI |> :code.priv_dir() |> to_string()
  @downloads_dir Path.join([@priv_dir, "static", "downloads"])

  @impl true
  def mount(_params, _session, socket) do
    # Create directories and clean them
    File.mkdir_p!(@protect_input_dir)
    File.mkdir_p!(@protect_output_dir)
    FrameworkTools.cleanup_files_input_output(@protect_input_dir, @protect_output_dir)

    PIIState.ensure_running()
    current_mode = PIIState.get_mode()

    # Get the lists and normalize them
    all_labels = PIIState.get_labels() |> Enum.map(&normalize_label/1)
    all_label_sets = PIIState.get_label_sets_names()
    active_label_sets = PIIState.get_active_label_sets_names()
    active_labels = PIIState.get_active_labels_names() |> Enum.map(&to_string/1)

    agent_status = PIIState.get_status()
    IO.puts("\nAgent Status:")

    Enum.each(agent_status, fn {name, status} ->
      IO.puts("\t#{name}: #{inspect(status)}")
    end)

    {:ok,
     socket
     |> assign(
       upload_complete: false,
       is_uploading: false,
       input_path: nil,
       current_mode: :anonymize,
       loading_protect: false,
       protected_segments: [],
       output_expanded: false,
       label_sets_expanded: false,
       single_labels_expanded: false,
       all_labels: all_labels,
       all_label_sets: all_label_sets,
       active_labels: active_labels,
       active_label_sets: active_label_sets,
       flash_message: nil,
       flash_type: nil,
       selected_language: "de",
       flash_timer: nil,
       recognizer_type: "example_based",
       generated_regex: nil,
       current_examples: [],
       editing_regex: false,
       recognizer_name: nil,
       recognizer_context: [],
       recognizer_language: nil,
       recognizer_valid: false,
       current_regex_input: nil,
       custom_recognizer_expanded: false,
       has_valid_files: false,
       txt_files_count: 0,
       loading_add_recognizer: false,
       add_recognizer_failed: false,
       deny_list_lines: [],
       deny_list_set_name: nil,
       deny_list_language: nil,
       deny_list_context: nil,
       retain_structure: true,
       show_download_section: false,
       selected_segment_id: nil,
       claimed_false_positives: []
     )
     |> allow_upload(:directory,
       accept: [".zip"],
       max_entries: 1,
       max_file_size: 30_000_000_000,
       auto_upload: true,
       progress: &handle_progress/3
     )}
  end

  @impl true
  def handle_event("toggle_output", _params, socket) do
    {:noreply, assign(socket, output_expanded: !socket.assigns.output_expanded)}
  end

  def handle_event("toggle_section", %{"section" => section}, socket) do
    key = String.to_existing_atom("#{section}_expanded")
    {:noreply, assign(socket, key, !socket.assigns[key])}
  end

  @impl true
  def handle_event("validate", _params, socket) do
    cleanup_protect_directories()
    {:noreply, socket}
  end

  def handle_event("save", _params, socket) do
    # This handler is needed for the form submission
    {:noreply, socket}
  end

  def handle_event("protect_all_files", _params, socket) do
    if socket.assigns.has_valid_files do
      socket =
        socket
        |> assign(loading_protect: true)
        |> assign(show_download_section: false)

      send(self(), {:do_protect_all_files})
      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  def handle_info({:do_protect_all_files}, socket) do
    active_labels = socket.assigns.active_labels
    language = socket.assigns.selected_language

    try do
      {result, collected_segments} =
        if socket.assigns.retain_structure do
          process_files_with_structure(socket, active_labels, language)
        else
          process_files_flat(socket, active_labels, language)
        end

      case result do
        {:ok, zip_path} ->
          handle_successful_protection(socket, zip_path, collected_segments)

        {:error, msg} ->
          {:noreply,
           socket
           |> assign(loading_protect: false)
           |> put_timed_flash(:error, "Protection failed: #{msg}")}
      end
    rescue
      e ->
        {:noreply,
         socket
         |> assign(loading_protect: false)
         |> put_timed_flash(:error, "Error: #{Exception.message(e)}")}
    end
  end

  defp process_files_with_structure(socket, active_labels, language) do
    case FrameworkTools.filter_and_clean_directory(@protect_input_dir, "txt") do
      {:ok, valid_files} ->
        {protected_files, collected_segments} =
          Enum.reduce(valid_files, {[], []}, fn file, {files, segments} ->
            rel_path = Path.relative_to(file, @protect_input_dir)
            output_file = Path.join(@protect_output_dir, rel_path)
            File.mkdir_p!(Path.dirname(output_file))

            case process_single_file(file, output_file, active_labels, language) do
              {:ok, file_segments} ->
                protected_only = extract_protected_segments(file_segments, file)
                {[{:ok, output_file} | files], segments ++ protected_only}

              {:error, reason} ->
                {[{:error, reason} | files], segments}
            end
          end)

        if Enum.all?(protected_files, fn {status, _} -> status == :ok end) do
          zip_result = create_zip_file(@protect_output_dir)
          {zip_result, collected_segments}
        else
          {{:error, "Failed to process some files"}, []}
        end

      {:error, reason} ->
        {{:error, reason}, []}
    end
  end

  defp process_files_flat(socket, active_labels, language) do
    case FrameworkTools.filter_and_clean_directory(@protect_input_dir, "txt") do
      {:ok, valid_files} ->
        # Create flat output directory
        protected_dir = Path.join(@protect_output_dir, "protected_files")
        File.mkdir_p!(protected_dir)

        # Use a MapSet to track taken filenames across all iterations
        {protected_files, collected_segments, _} =
          Enum.reduce(valid_files, {[], [], MapSet.new()}, fn file,
                                                              {files, segments, taken_names} ->
            base_name = Path.basename(file, ".txt")

            # Get next available name that hasn't been used
            {output_file, new_taken_names} =
              FrameworkTools.get_next_available_name(base_name, taken_names, protected_dir)

            case process_single_file(file, output_file, active_labels, language) do
              {:ok, file_segments} ->
                protected_only = extract_protected_segments(file_segments, file)
                {[{:ok, output_file} | files], segments ++ protected_only, new_taken_names}

              {:error, reason} ->
                {[{:error, reason} | files], segments, new_taken_names}
            end
          end)

        if Enum.all?(protected_files, fn {status, _} -> status == :ok end) do
          zip_result = create_zip_file(protected_dir)
          {zip_result, collected_segments}
        else
          {{:error, "Failed to process some files"}, []}
        end

      {:error, reason} ->
        {{:error, reason}, []}
    end
  end

  defp create_zip_file(source_dir) do
    zip_path = Path.join(@downloads_dir, "protected_files.zip")
    File.mkdir_p!(@downloads_dir)
    if File.exists?(zip_path), do: File.rm!(zip_path)

    case FrameworkTools.create_zip_archive(source_dir, zip_path) do
      :ok -> {:ok, zip_path}
      error -> error
    end
  end

  defp handle_successful_protection(socket, zip_path, collected_segments) do
    download_path = "/downloads/#{Path.basename(zip_path)}"
    Process.send_after(self(), {:cleanup_download, zip_path}, :timer.minutes(5))

    {:noreply,
     socket
     |> assign(
       loading_protect: false,
       protected_segments: collected_segments,
       output_expanded: true,
       download_path: download_path,
       show_download_section: true
     )
     |> put_timed_flash(:info, "Files protected successfully! Click to download.")}
  end

  def extract_text_from_file(file_input_path) do
    case File.read(file_input_path) do
      {:ok, text} -> text
      {:error, reason} -> raise "Failed to read file: #{reason}"
    end
  end

  def protect_extracted_text(text, active_labels, language) do
    case PIIHelpers.protect_text(text, active_labels, language, true) do
      {:ok, response} ->
        processed_response = PIIHelpers.process_protection_response_with_analysis(response)
        formatted_segments = PIIHelpers.format_protected_text_with_analysis(processed_response)
        {:ok, formatted_segments}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def handle_event("toggle_structure", _params, socket) do
    {:noreply, assign(socket, retain_structure: !socket.assigns.retain_structure)}
  end

  def handle_event("toggle_segment_details", %{"id" => id}, socket) do
    current_selected = socket.assigns.selected_segment_id
    new_selected = if current_selected == id, do: nil, else: id
    {:noreply, assign(socket, selected_segment_id: new_selected)}
  end

  defp handle_progress(:directory, entry, socket) do
    if entry.done? do
      try do
        cleanup_protect_directories()

        # Consume the uploaded file and extract it
        consume_uploaded_entry(socket, entry, fn %{path: temp_path} ->
          case extract_zip(temp_path, @protect_input_dir) do
            :ok ->
              # Count TXT files after successful extraction
              txt_count = count_txt_files(@protect_input_dir)
              IO.puts("Found #{txt_count} TXT files in directory")
              {:ok, txt_count}

            {:error, reason} ->
              {:error, reason}
          end
        end)

        # After extraction, update socket based on file count
        txt_count = count_txt_files(@protect_input_dir)

        if txt_count > 0 do
          {:noreply,
           socket
           |> assign(
             upload_complete: true,
             is_uploading: false,
             has_valid_files: true,
             txt_files_count: txt_count,
             input_path: @protect_input_dir
           )}
        else
          {:noreply,
           socket
           |> assign(
             upload_complete: false,
             is_uploading: false,
             has_valid_files: false,
             txt_files_count: 0
           )
           |> put_timed_flash(:error, "No TXT files found in the uploaded ZIP")}
        end
      rescue
        e ->
          cleanup_protect_directories()

          {:noreply,
           socket
           |> assign(
             is_uploading: false,
             has_valid_files: false,
             txt_files_count: 0
           )
           |> put_timed_flash(:error, "Error processing ZIP: #{Exception.message(e)}")}
      end
    else
      {:noreply, assign(socket, is_uploading: true)}
    end
  end

  # Helper functions for ZIP handling
  defp extract_zip(zip_path, dest_path) do
    try do
      case :os.type() do
        {:unix, _} ->
          case System.cmd("unzip", ["-o", zip_path, "-d", dest_path]) do
            {_, 0} -> :ok
            {error, _} -> {:error, error}
          end

        {:win32, _} ->
          seven_zip_path = FrameworkTools.find_7zip()

          case System.cmd(seven_zip_path, ["x", "-y", "-o#{dest_path}", zip_path]) do
            {_, 0} -> :ok
            {error, _} -> {:error, error}
          end
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp count_txt_files(dir) do
    case Path.wildcard(Path.join([dir, "**", "*.txt"])) do
      [] -> 0
      files -> length(files)
    end
  end

  defp cleanup_protect_directories do
    FrameworkTools.cleanup_files_input_output(@protect_input_dir, @protect_output_dir)
  end

  defp process_single_file(input_file, output_file, active_labels, language) do
    with {:ok, raw_text} <- File.read(input_file),
         normalized_text <- PIIHelpers.normalize_input_text(raw_text),
         {:ok, response} <-
           PIIHelpers.protect_text(normalized_text, active_labels, language, true) do
      # Get segments from the response
      IO.inspect(response, label: "\nProtection response")

      # Handle response with charlist keys
      formatted_segments =
        case response do
          %{~c"single_results" => results} when is_list(results) ->
            results
            |> Enum.with_index()
            |> Enum.map(fn {segment, index} ->
              %{
                "id" => "segment-#{Path.basename(input_file)}-#{index}",
                "start" => get_in(segment, [~c"start"]) || 0,
                "end" => get_in(segment, [~c"end"]) || 0,
                "original_text" => to_string(get_in(segment, [~c"original_text"]) || ""),
                "protected_text" => to_string(get_in(segment, [~c"protected_text"]) || ""),
                "recognizer_name" => to_string(get_in(segment, [~c"recognizer_name"]) || ""),
                "score" => get_in(segment, [~c"score"]) || 0.0,
                "pattern" => get_in(segment, [~c"pattern"]) || 0.0
                # "validation_result" => get_in(segment, [~c"validation_result"])
              }
            end)

          _ ->
            []
        end

      # IO.inspect(formatted_segments, label: "Formatted segments")

      # Get anonymized text from response
      anonymized_text =
        case response do
          %{~c"anonymized_text" => text} when is_list(text) -> to_string(text)
          %{~c"anonymized_text" => text} -> to_string(text)
          _ -> normalized_text
        end

      # Write the protected text to file
      case File.write(output_file, anonymized_text) do
        :ok -> {:ok, formatted_segments}
        error -> error
      end
    end
  end

  defp apply_replacements_ordered(text, segments) do
    # Sort segments by position in descending order to ensure earlier replacements don't affect later positions
    sorted_segments = Enum.sort_by(segments, fn s -> -(s["start"] || 0) end)
    text_length = String.length(text)

    # Apply each replacement
    Enum.reduce(sorted_segments, text, fn segment, current_text ->
      start_pos = segment["start"] || 0
      end_pos = segment["end"] || 0
      replacement = segment["protected_text"] || "<#{segment["recognizer_name"]}>"

      # Move condition outside guard clause
      if start_pos >= 0 and end_pos > start_pos and end_pos <= text_length do
        String.slice(current_text, 0, start_pos) <>
          replacement <>
          String.slice(current_text, end_pos..-1)
      else
        current_text
      end
    end)
  end

  defp extract_protected_segments(segments, filename) when is_list(segments) do
    segments
    |> Enum.map(fn segment ->
      {Path.basename(filename),
       %{
         id: segment["id"],
         original_text: segment["original_text"],
         protected_text: "<#{segment["recognizer_name"]}>",
         recognizer_name: segment["recognizer_name"],
         score: segment["score"],
         pattern: segment["pattern"],
         # validation_result: segment["validation_result"],
         start: segment["start"],
         end: segment["end"]
       }}
    end)
  end

  def handle_info({:cleanup_download, file_path}, socket) do
    if File.exists?(file_path), do: File.rm!(file_path)
    {:noreply, socket}
  end
end
