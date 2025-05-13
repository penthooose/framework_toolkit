defmodule Phoenix_UIWeb.Components.CustomRecognizerBackend do
  defmacro __using__(_opts) do
    alias Phoenix_UI.State.PIIState

    quote do
      def handle_event("toggle_custom_recognizer", _params, socket) do
        {:noreply,
         assign(socket, :custom_recognizer_expanded, !socket.assigns.custom_recognizer_expanded)}
      end

      def handle_event(
            "add_custom_regex",
            %{"name" => name, "regex" => regex, "context" => context, "language" => language},
            socket
          ) do
        context_list =
          context |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))

        cond do
          String.trim(name) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Pattern name cannot be empty")}

          String.trim(regex) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Regex pattern cannot be empty")}

          true ->
            case RegexGenerator.validate_regex(regex) do
              {:ok, _} ->
                socket =
                  assign(socket,
                    loading_add_recognizer: true,
                    recognizer_name: name,
                    recognizer_context: context_list,
                    recognizer_language: language
                  )

                send(
                  self(),
                  {:do_add_recognizer_regex, name, RegexGenerator.formate_regex(regex),
                   context_list, language}
                )

                {:noreply, socket}

              {:error, reason} ->
                {:noreply,
                 socket
                 |> assign(:current_regex_input, regex)
                 |> assign(:recognizer_name, name)
                 |> assign(:recognizer_context, context_list)
                 |> assign(:recognizer_language, language)
                 |> put_timed_flash(:error, "Invalid regex pattern: #{inspect(reason)}")}
            end
        end
      end

      def handle_info({:do_add_recognizer_regex, name, regex, context_list, language}, socket) do
        case MainPii.add_pattern_recognizer_with_regex_erlport(
               name,
               regex,
               context_list,
               language
             ) do
          result when is_list(result) and result == ~c"SUCCESS" ->
            handle_successful_recognizer_addition(socket)

          error ->
            {:noreply,
             socket
             |> assign(loading_add_recognizer: false)
             |> put_timed_flash(:error, "Error adding recognizer: #{inspect(error)}")}
        end
      end

      def handle_event(
            "generate_regex",
            %{
              "name" => name,
              "examples" => examples,
              "context" => context,
              "language" => language
            },
            socket
          ) do
        examples_list =
          examples |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))

        context_list =
          context |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))

        IO.puts("[PIILive] Examples: #{inspect(examples_list)}")
        IO.puts("[PIILive] Context: #{inspect(context_list)}")
        IO.puts("[PIILive] Language: #{inspect(language)}")

        cond do
          String.trim(name) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Pattern name cannot be empty")}

          length(examples_list) < 2 ->
            {:noreply, put_timed_flash(socket, :error, "Please provide at least 2 examples")}

          true ->
            regex = RegexGenerator.derive_regex(examples_list)

            {:noreply,
             assign(socket,
               generated_regex: regex,
               current_regex_input: regex,
               current_examples: examples_list,
               recognizer_name: name,
               recognizer_context: context_list,
               recognizer_language: language,
               recognizer_valid: false
             )}
        end
      end

      def handle_event("add_recognizer_regex", %{"name" => name, "regex" => regex}, socket) do
        case RegexGenerator.validate_regex(regex) do
          {:ok, _} ->
            socket = assign(socket, loading_add_recognizer: true)
            send(self(), {:do_add_recognizer_regex, name, regex})
            {:noreply, socket}

          {:error, _} ->
            {:noreply, put_timed_flash(socket, :error, "Invalid regex pattern")}
        end
      end

      def handle_event(
            "add_recognizer_deny_list",
            %{
              "name" => name,
              "deny_list" => deny_list,
              "context" => context,
              "language" => language
            },
            socket
          ) do
        # Process deny list - split by newlines and clean up
        deny_list_items =
          deny_list
          |> String.split("\n")
          |> Enum.map(&String.trim/1)
          |> Enum.reject(&(&1 == ""))
          # Ensure strings
          |> Enum.map(&to_string/1)

        # Process context
        context_list =
          context
          |> String.split(",")
          |> Enum.map(&String.trim/1)
          |> Enum.reject(&(&1 == ""))
          # Ensure strings
          |> Enum.map(&to_string/1)

        cond do
          String.trim(name) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Pattern name cannot be empty")}

          deny_list_items == [] ->
            {:noreply, put_timed_flash(socket, :error, "Deny list cannot be empty")}

          true ->
            socket = assign(socket, loading_add_recognizer: true)
            send(self(), {:do_add_deny_list, name, deny_list_items, context_list, language})
            {:noreply, socket}
        end
      end

      def handle_info({:do_add_deny_list, name, deny_list, context, language}, socket) do
        case MainPii.add_deny_list_recognizer_erlport(name, deny_list, context, language) do
          result when is_list(result) and result == ~c"SUCCESS" ->
            PIIState.reset_agents(:labels_agent)
            # Convert the labels to simple strings
            all_labels =
              PIIState.get_labels()
              |> Enum.map(&normalize_label/1)
              |> Enum.map(fn
                label when is_map(label) -> to_string(label["recognizer_name"])
                label -> to_string(label)
              end)

            # Convert custom recognizers to strings
            custom_recognizers =
              MainPii.get_custom_recognizer_entities()
              |> Enum.map(fn
                recognizer when is_map(recognizer) -> to_string(recognizer["recognizer_name"])
                recognizer -> to_string(recognizer)
              end)

            {:noreply,
             socket
             |> assign(loading_add_recognizer: false)
             |> assign(all_labels: all_labels)
             |> assign(custom_recognizers: custom_recognizers)
             |> assign(current_examples: [])
             |> assign(recognizer_name: nil)
             |> assign(recognizer_context: [])
             |> assign(recognizer_language: nil)
             |> put_timed_flash(:info, "Deny list recognizer added successfully")}

          error ->
            {:noreply,
             socket
             |> assign(loading_add_recognizer: false)
             |> put_timed_flash(:error, "Error adding deny list recognizer: #{inspect(error)}")}
        end
      end

      def handle_event(
            "add_not_recognizer",
            %{"name" => name, "deny_item" => deny_item, "language" => language},
            socket
          ) do
        deny_list_items = [deny_item]
        not_name = "NOT_#{String.downcase(name)}"

        cond do
          String.trim(name) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Pattern name cannot be empty")}

          deny_list_items == [] ->
            {:noreply, put_timed_flash(socket, :error, "Deny list cannot be empty")}

          true ->
            send(self(), {:do_add_not_recognizer, not_name, deny_list_items, [], language})
            # Add the deny_item to claimed_false_positives
            {:noreply, update(socket, :claimed_false_positives, &[deny_item | &1])}
        end
      end

      def handle_info({:do_add_not_recognizer, not_name, deny_list, context, language}, socket) do
        case MainPii.add_deny_list_recognizer_erlport(not_name, deny_list, context, language) do
          result when is_list(result) and result == ~c"SUCCESS" ->
            # Phoenix_UI.State.PIIState.reset_agents(:labels_agent)

            {:noreply,
             socket
             |> put_timed_flash(:info, "NOT recognizer added successfully")}

          error ->
            {:noreply,
             socket
             |> put_timed_flash(:error, "Error adding NOT recognizer: #{inspect(error)}")}
        end
      end

      def handle_successful_recognizer_addition(socket) do
        PIIState.reset_agents(:labels_agent)
        all_labels = PIIState.get_labels() |> Enum.map(&normalize_label/1)

        {:noreply,
         socket
         |> assign(loading_add_recognizer: false)
         |> assign(all_labels: all_labels)
         |> assign(generated_regex: nil)
         |> assign(current_examples: [])
         |> assign(recognizer_name: nil)
         |> assign(recognizer_context: [])
         |> assign(recognizer_language: nil)
         |> assign(current_regex_input: nil)
         |> assign(recognizer_valid: false)
         |> put_timed_flash(:info, "Recognizer added successfully", 1000)}
      end

      def handle_event("recognizer_type_changed", %{"type" => type}, socket) do
        IO.puts("Changed recognizer type to: #{type}")

        {
          :noreply,
          socket
          |> assign(:recognizer_type, type)
          |> assign(:generated_regex, nil)
          |> assign(:current_examples, [])
          |> assign(:recognizer_name, nil)
          |> assign(:recognizer_context, [])
          |> assign(:recognizer_language, socket.assigns.selected_language)
          |> assign(:recognizer_valid, false)
        }
      end

      def handle_event(
            "add_regex",
            %{"name" => name, "regex" => regex, "context" => context, "language" => language},
            socket
          ) do
        # Split context string into list and clean it
        context_list =
          context |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))

        cond do
          String.trim(name) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Pattern name cannot be empty")}

          String.trim(regex) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Regex pattern cannot be empty")}

          true ->
            case RegexGenerator.validate_regex(regex) do
              {:ok, _} ->
                socket =
                  assign(socket,
                    loading_add_recognizer: true,
                    recognizer_name: name,
                    recognizer_context: context_list,
                    recognizer_language: language
                  )

                send(
                  self(),
                  {:do_add_recognizer_regex, name, RegexGenerator.formate_regex(regex),
                   context_list, language}
                )

                {:noreply, socket}

              {:error, reason} ->
                {:noreply,
                 socket
                 |> assign(:current_regex_input, regex)
                 |> assign(:recognizer_name, name)
                 |> assign(:recognizer_context, context_list)
                 |> assign(:recognizer_language, language)
                 |> put_timed_flash(:error, "Invalid regex pattern: #{inspect(reason)}")}
            end
        end
      end

      def handle_event(
            "add_custom_regex",
            %{"name" => name, "regex" => regex, "context" => context, "language" => language},
            socket
          ) do
        # Split context string into list and clean it
        context_list =
          context |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))

        cond do
          String.trim(name) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Pattern name cannot be empty")}

          String.trim(regex) == "" ->
            {:noreply, put_timed_flash(socket, :error, "Regex pattern cannot be empty")}

          true ->
            case RegexGenerator.validate_regex(regex) do
              {:ok, _} ->
                socket =
                  assign(socket,
                    loading_add_recognizer: true,
                    recognizer_name: name,
                    recognizer_context: context_list,
                    recognizer_language: language
                  )

                send(
                  self(),
                  {:do_add_recognizer_regex, name, RegexGenerator.formate_regex(regex),
                   context_list, language}
                )

                {:noreply, socket}

              {:error, reason} ->
                {:noreply,
                 socket
                 |> assign(:current_regex_input, regex)
                 |> assign(:recognizer_name, name)
                 |> assign(:recognizer_context, context_list)
                 |> assign(:recognizer_language, language)
                 |> put_timed_flash(:error, "Invalid regex pattern: #{inspect(reason)}")}
            end
        end
      end

      def handle_info({:do_add_recognizer_regex, name, regex, context_list, language}, socket) do
        case MainPii.add_pattern_recognizer_with_regex_erlport(
               name,
               regex,
               context_list,
               language
             ) do
          result when is_list(result) and result == ~c"SUCCESS" ->
            handle_successful_recognizer_addition(socket)

          error ->
            {:noreply,
             socket
             |> assign(loading_add_recognizer: false)
             |> put_timed_flash(:error, "Error adding recognizer: #{inspect(error)}")}
        end
      end

      def handle_event("update_regex_input", %{"regex" => value}, socket) do
        {:noreply, assign(socket, :current_regex_input, value)}
      end

      def handle_event("restore_regex", _params, socket) do
        {:noreply,
         socket
         |> assign(:current_regex_input, socket.assigns.generated_regex)
         |> assign(:recognizer_valid, false)}
      end

      def handle_event("validate_modified_regex", %{"regex" => regex}, socket) do
        with {:ok, _} <- RegexGenerator.validate_regex(regex),
             true <- RegexGenerator.check_regex_fitting(regex, socket.assigns.current_examples) do
          {:noreply,
           socket
           |> assign(:recognizer_valid, true)
           |> put_timed_flash(:info, "Regex is valid and matches all examples")}
        else
          false ->
            {:noreply,
             socket
             |> assign(:recognizer_valid, false)
             |> put_timed_flash(:error, "Regex doesn't match all examples")}

          {:error, _} ->
            {:noreply,
             socket
             |> assign(:recognizer_valid, false)
             |> put_timed_flash(:error, "Invalid regex pattern")}
        end
      end

      def handle_event("remove_custom_recognizer", %{"recognizer_name" => name}, socket) do
        case name do
          "ALL" ->
            # Handle removing all recognizers
            # You'll need to implement this functionality in MainPii
            MainPii.remove_all_recognizers()
            PIIState.reset_agents(:labels_agent)

            {:noreply,
             socket
             |> assign(:custom_recognizers, [])
             |> put_timed_flash(:info, "All custom recognizers removed")}

          "" ->
            {:noreply, put_timed_flash(socket, :error, "Please select a recognizer to remove")}

          name ->
            case MainPii.remove_recognizer(name) do
              true ->
                # Refresh the list of recognizer names
                updated_recognizers = MainPii.get_custom_recognizers() |> Enum.map(& &1[~c"name"])

                {:noreply,
                 socket
                 |> assign(:custom_recognizers, updated_recognizers)
                 |> put_timed_flash(:info, "Recognizer #{name} removed successfully")}

              false ->
                {:noreply, put_timed_flash(socket, :error, "Failed to remove recognizer #{name}")}
            end
        end
      end
    end
  end
end
