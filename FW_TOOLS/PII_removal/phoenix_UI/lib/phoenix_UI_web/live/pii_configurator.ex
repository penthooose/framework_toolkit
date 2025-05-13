defmodule Phoenix_UIWeb.PIIConfigurator do
  use Phoenix_UIWeb, :live_view
  alias Phoenix_UI.State.PIIState
  use Phoenix_UIWeb.Components.CustomRecognizerBackend
  use Phoenix_UIWeb.Components.GeneralFunctionalities
  use Phoenix_UIWeb.Components.LabelsAndLabelSetsBackend
  import Phoenix_UIWeb.Components.CustomRecognizerForm
  import Phoenix_UIWeb.Components.LabelsAndLabelSetsForm
  import Phoenix.LiveView.JS

  def mount(_params, _session, socket) do
    PIIState.ensure_running()

    # Get initial data
    label_sets = PIIState.get_label_sets()
    all_labels = PIIState.get_labels_names() |> Enum.map(&to_string/1)

    custom_recognizers = MainPii.get_custom_recognizers() |> Enum.map(& &1[~c"name"])

    socket =
      assign(socket,
        label_sets: label_sets,
        all_labels: all_labels,
        selected_labels: [],
        new_set_name: "",
        flash_message: nil,
        flash_type: nil,
        # Now contains recognizer names
        custom_recognizers: custom_recognizers,
        custom_recognizer_expanded: false,
        recognizer_type: "example_based",
        current_examples: [],
        recognizer_context: [],
        recognizer_language: nil,
        generated_regex: nil,
        current_regex_input: nil,
        recognizer_valid: false,
        recognizer_name: nil,
        flash_timer: nil,
        loading_add_recognizer: false,
        loading_remove_recognizer: false,
        # Add this line
        selected_language: "any"
      )

    if connected?(socket) do
      script = JS.dispatch("js:init", detail: %{})
      {:ok, push_event(socket, "init", %{})}
    else
      {:ok, socket}
    end
  end

  def handle_event("toggle_label_selection", %{"label" => label}, socket) do
    selected = socket.assigns.selected_labels

    new_selected =
      if label in selected do
        List.delete(selected, label)
      else
        [label | selected]
      end

    {:noreply, assign(socket, selected_labels: new_selected)}
  end
end
