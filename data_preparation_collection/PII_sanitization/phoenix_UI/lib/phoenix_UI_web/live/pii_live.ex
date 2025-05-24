defmodule Phoenix_UIWeb.PIILive do
  use Phoenix_UIWeb, :live_view
  use Phoenix_UIWeb.Components.CustomRecognizerBackend
  use Phoenix_UIWeb.Components.GeneralFunctionalities
  use Phoenix_UIWeb.Components.LabelsAndLabelSetsBackend
  import Phoenix_UIWeb.Components.CustomRecognizerForm
  import Phoenix_UIWeb.Components.LabelsAndLabelSetsForm
  import PhoenixUIWeb.PIIHelpers
  alias Phoenix_UI.State.PIIState
  alias PhoenixUIWeb.PIIHelpers

  def mount(_params, _session, socket) do
    PIIState.ensure_running()
    current_mode = PIIState.get_mode()

    # Get all Single Recognizers (aka labels) and Recognizer Sets (aka label sets)
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
     assign(socket,
       input_text: "",
       display_text: "",
       analyzed_text: [],
       dropdown_visible: %{},
       highlights: [],
       loading_analyze: false,
       loading_protect: false,
       loading_add_recognizer: false,
       flash_message: nil,
       flash_type: nil,
       current_mode: current_mode,
       protected_text: "",
       protected_segments: [],
       label_sets_expanded: false,
       single_labels_expanded: false,
       all_labels: all_labels,
       all_label_sets: all_label_sets,
       active_labels: active_labels,
       active_label_sets: active_label_sets,
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
       claimed_false_positives: []
     )}
  end

  def handle_event("text_changed", %{"key" => _key, "value" => text}, socket) do
    IO.puts("Text changed: #{text}")
    normalized_text = PIIHelpers.normalize_input_text(text)
    socket = assign(socket, input_text: normalized_text)
    {:noreply, socket}
  end

  def handle_event("analyze", _params, socket) do
    case socket.assigns.input_text do
      "" ->
        put_timed_flash(socket, :error, "Please enter text to analyze")

      text ->
        socket = assign(socket, loading_analyze: true, claimed_false_positives: [])
        send(self(), {:do_analysis, text})
        {:noreply, socket}
    end
  end

  def handle_info({:do_analysis, text}, socket) do
    active_labels = socket.assigns.active_labels
    analyzed = MainPii.analyze_text_erlport(text, active_labels, socket.assigns.selected_language)

    case analyzed do
      {:error, _} = error ->
        {:noreply,
         socket
         |> assign(loading_analyze: false)
         |> put_timed_flash(:error, "Error analyzing text: #{inspect(error)}")}

      _ ->
        formatted = PIIHelpers.format_analysis(analyzed)

        {:noreply,
         assign(socket,
           display_text: text,
           analyzed_text: formatted,
           highlights: formatted,
           dropdown_visible: %{},
           loading_analyze: false
         )}
    end
  end

  def handle_event("protect_text", _params, socket) do
    case socket.assigns.input_text do
      "" ->
        put_timed_flash(socket, :error, "Please enter text to protect")

      text ->
        socket = assign(socket, loading_protect: true)
        send(self(), {:do_protect, text})
        {:noreply, socket}
    end
  end

  def handle_info({:do_protect, text}, socket) do
    active_labels = socket.assigns.active_labels
    language = socket.assigns.selected_language

    IO.inspect(language, label: "Language")
    IO.inspect(active_labels, label: "Active Labels")

    case PIIHelpers.protect_text(text, active_labels, language) do
      {:ok, response} ->
        processed_response = PIIHelpers.process_protection_response(response)
        formatted_segments = PIIHelpers.format_protected_text(processed_response)

        {:noreply,
         socket
         |> assign(protected_segments: formatted_segments)
         |> assign(loading_protect: false)}

      {:error, reason} ->
        {:noreply,
         socket
         |> assign(loading_protect: false)
         |> put_timed_flash(:error, "Error protecting text: #{inspect(reason)}")}
    end
  end

  def handle_event("toggle_dropdown", %{"index" => index}, socket) do
    index = String.to_integer(index)
    current = Map.get(socket.assigns.dropdown_visible, index, false)

    dropdown_visible = socket.assigns.dropdown_visible || %{}

    {:noreply,
     assign(socket,
       dropdown_visible: Map.put(dropdown_visible, index, !current)
     )}
  end
end
