defmodule Phoenix_UIWeb.Components.LabelsAndLabelSetsBackend do
  defmacro __using__(_opts) do
    quote do
      alias Phoenix_UI.State.PIIState

      def handle_event("toggle_section", %{"section" => section}, socket) do
        IO.puts("Toggle section: #{section}")

        case section do
          "label_sets" ->
            new_state = !socket.assigns.label_sets_expanded

            IO.puts("Label sets expanded: #{new_state}")
            {:noreply, assign(socket, label_sets_expanded: new_state)}

          "single_labels" ->
            new_state = !socket.assigns.single_labels_expanded
            # Add debug output
            IO.puts("Single labels expanded: #{new_state}")
            {:noreply, assign(socket, single_labels_expanded: new_state)}
        end
      end

      def handle_event("toggle_labels", %{"label" => label}, socket) do
        IO.puts("Toggled label: #{inspect(label)}")

        label_charlist = String.to_charlist(label)
        PIIState.toggle_label_active(label_charlist)

        updated_labels = PIIState.get_active_labels_names() |> Enum.map(&to_string/1)

        {:noreply, assign(socket, active_labels: updated_labels)}
      end

      def handle_event("toggle_label_sets", %{"set" => set}, socket) do
        IO.puts("Toggled set: #{inspect(set)}")
        PIIState.toggle_label_set_active(set)

        updated_sets = PIIState.get_active_label_sets_names() |> Enum.map(&to_string/1)
        updated_labels = PIIState.get_active_labels_names() |> Enum.map(&to_string/1)
        {:noreply, assign(socket, active_label_sets: updated_sets, active_labels: updated_labels)}
      end

      def normalize_label(%{"active" => active, "recognizer_name" => name}) when is_list(name) do
        %{"active" => active, "recognizer_name" => to_string(name)}
      end

      def normalize_label(label), do: label

      def handle_event("create_label_set", %{"name" => name, "labels" => labels}, socket) do
        labels = Enum.map(labels, &String.to_charlist/1)

        case PIIState.create_new_label_set(name, labels) do
          :ok ->
            PIIState.reset_agents(:label_set_agent)

            {:noreply,
             socket
             |> assign(label_sets: PIIState.get_label_sets())
             |> put_timed_flash(:info, "Label set created successfully")}

          {:error, reason} ->
            {:noreply,
             put_timed_flash(socket, :error, "Error creating label set: #{inspect(reason)}")}
        end
      end

      def handle_event("remove_label_set", %{"name" => name}, socket) do
        case PIIState.remove_label_set(name) do
          :ok ->
            PIIState.reset_agents(:label_set_agent)

            socket =
              socket
              |> assign(label_sets: PIIState.get_label_sets())
              |> put_timed_flash(:info, "Label set removed successfully")

            {:noreply, socket}

          {:error, reason} ->
            {:noreply,
             put_timed_flash(socket, :error, "Error removing label set: #{inspect(reason)}")}
        end
      end

      def handle_event("validate_set_name", %{"name" => name}, socket) do
        {:noreply, assign(socket, new_set_name: name)}
      end

      def handle_event(
            "add_label_to_set",
            %{"set" => set_name, "label" => label} = params,
            socket
          ) do
        IO.puts("Event received with params: #{inspect(params)}")

        if label && label != "" do
          label_charlist = String.to_charlist(label)

          case PIIState.add_label_to_label_set(set_name, label_charlist) do
            :ok ->
              PIIState.reset_agents(:label_set_agent)

              {:noreply,
               socket
               |> assign(label_sets: PIIState.get_label_sets())
               |> put_timed_flash(:info, "Label #{label} added to #{set_name}")}

            {:error, reason} ->
              {:noreply,
               put_timed_flash(socket, :error, "Error adding label: #{inspect(reason)}")}
          end
        else
          {:noreply, put_timed_flash(socket, :error, "Please select a label to add")}
        end
      end

      def handle_event("remove_label_from_set", %{"set" => set_name, "label" => label}, socket) do
        label_charlist = String.to_charlist(label)
        IO.inspect(label_charlist)

        case PIIState.remove_label_from_label_set(set_name, label_charlist) do
          :ok ->
            PIIState.reset_agents(:label_set_agent)

            {:noreply,
             socket
             |> assign(label_sets: PIIState.get_label_sets())
             |> put_timed_flash(:info, "Label #{label} removed from #{set_name}")}

          {:error, reason} ->
            {:noreply,
             put_timed_flash(socket, :error, "Error removing label: #{inspect(reason)}")}
        end
      end

      def handle_event("remove_custom_recognizer", %{"label" => "ALL"}, socket) do
        socket = assign(socket, loading_remove_recognizer: true)
        send(self(), :do_remove_all_recognizers)
        {:noreply, socket}
      end

      def handle_event("remove_custom_recognizer", %{"label" => label}, socket)
          when label != "" do
        socket = assign(socket, loading_remove_recognizer: true)
        send(self(), {:do_remove_recognizer, label})
        {:noreply, socket}
      end

      def handle_event("remove_custom_recognizer", _, socket) do
        {:noreply, put_timed_flash(socket, :error, "Please select a recognizer to remove")}
      end

      def handle_info(:do_remove_all_recognizers, socket) do
        case PIIState.remove_all_custom_labels() do
          {:ok, message} ->
            custom_recognizers = MainPii.get_custom_recognizer_entities()
            all_labels = PIIState.get_labels_names() |> Enum.map(&to_string/1)
            label_sets = PIIState.get_label_sets()

            {:noreply,
             socket
             |> assign(loading_remove_recognizer: false)
             |> assign(
               custom_recognizers: custom_recognizers,
               all_labels: all_labels,
               label_sets: label_sets
             )
             |> put_timed_flash(:info, message)}

          {:error, reason} ->
            {:noreply,
             socket
             |> assign(loading_remove_recognizer: false)
             |> put_timed_flash(:error, "Error removing all recognizers: #{inspect(reason)}")}
        end
      end

      def handle_info({:do_remove_recognizer, label}, socket) do
        case PIIState.remove_custom_label(label) do
          {:ok, message} ->
            custom_recognizers = MainPii.get_custom_recognizer_entities()
            all_labels = PIIState.get_labels_names() |> Enum.map(&to_string/1)
            label_sets = PIIState.get_label_sets()

            {:noreply,
             socket
             |> assign(loading_remove_recognizer: false)
             |> assign(
               custom_recognizers: custom_recognizers,
               all_labels: all_labels,
               label_sets: label_sets
             )
             |> put_timed_flash(:info, message)}

          {:error, reason} ->
            {:noreply,
             socket
             |> assign(loading_remove_recognizer: false)
             |> put_timed_flash(:error, "Error removing recognizer: #{inspect(reason)}")}
        end
      end
    end
  end
end
