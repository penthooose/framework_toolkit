defmodule Phoenix_UIWeb.Components.LabelsAndLabelSetsForm do
  use Phoenix.Component

  attr(:label_sets_expanded, :boolean, required: true)
  attr(:single_labels_expanded, :boolean, required: true)
  attr(:all_label_sets, :list, required: true)
  attr(:active_label_sets, :list, required: true)
  attr(:all_labels, :list, required: true)
  attr(:active_labels, :list, required: true)

  def labels_section(assigns) do
    ~H"""
    <!-- Label Sets Section -->
    <div class="border rounded-lg mt-4">
      <div class="collapsible-header" phx-click="toggle_section" phx-value-section="label_sets">
        <h4 class="text-md font-medium">Recognizer Sets <%= if @label_sets_expanded, do: "▼", else: "▶" %></h4>
      </div>
      <%= if @label_sets_expanded do %>
        <div class="collapsible-content expanded">
          <%= for set_name <- @all_label_sets do %>
            <button class={"w-full px-4 py-2 text-sm font-medium rounded shadow-sm transition-all duration-200 #{if set_name in @active_label_sets, do: "bg-green-500 text-white", else: "bg-white text-gray-700 border border-gray-300"} hover:shadow-md"}
                    phx-click="toggle_label_sets"
                    phx-value-set={set_name}>
              <%= String.replace(set_name, "_", " ") |> String.capitalize() %>
            </button>
          <% end %>
        </div>
      <% end %>
    </div>

    <!-- Single Labels Section -->
    <div class="border rounded-lg mt-4">
      <div class="collapsible-header" phx-click="toggle_section" phx-value-section="single_labels">
        <h4 class="text-md font-medium">Single Recognizers <%= if @single_labels_expanded, do: "▼", else: "▶" %></h4>
      </div>
      <%= if @single_labels_expanded do %>
        <div class="collapsible-content expanded">
          <%= for label <- @all_labels do %>
            <button class={"w-full px-4 py-2 text-sm font-medium rounded shadow-sm transition-all duration-200 #{if label["recognizer_name"] in @active_labels, do: "bg-green-500 text-white", else: "bg-white text-gray-700 border border-gray-300"} hover:shadow-md"}
                    phx-click="toggle_labels"
                    phx-value-label={label["recognizer_name"]}>
              <%= label["recognizer_name"] |> String.replace("_", " ") %>
            </button>
          <% end %>
        </div>
      <% end %>
    </div>
    """
  end
end
