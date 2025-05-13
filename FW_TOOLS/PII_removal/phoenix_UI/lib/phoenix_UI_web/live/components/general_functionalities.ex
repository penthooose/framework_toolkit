defmodule Phoenix_UIWeb.Components.GeneralFunctionalities do
  alias Phoenix_UI.State.PIIState

  defmacro __using__(_opts) do
    quote do
      def handle_event("language_changed", %{"language" => language}, socket) do
        IO.inspect(language, label: "Selected language")
        {:noreply, assign(socket, selected_language: language)}
      end

      def handle_event("update_mode", %{"value" => mode}, socket) do
        new_mode = String.to_atom(mode)
        IO.inspect(new_mode, label: "Mode")

        case PIIState.set_mode(new_mode) do
          :ok ->
            {:noreply,
             socket
             |> assign(:current_mode, new_mode)
             |> put_timed_flash(:success, "Protection mode updated to: #{mode}")}

          {:error, reason} ->
            {:noreply,
             socket
             |> put_timed_flash(:error, "Error updating mode: #{inspect(reason)}")}
        end
      end

      def put_timed_flash(socket, type, message, time \\ 5000) do
        # Cancel any existing timer
        if socket.assigns.flash_timer, do: Process.cancel_timer(socket.assigns.flash_timer)

        # Set new flash and timer
        timer_ref = Process.send_after(self(), :clear_flash, time)

        socket
        |> assign(:flash_timer, timer_ref)
        |> assign(flash_message: message, flash_type: type)
      end

      def handle_info(:clear_flash, socket) do
        {:noreply,
         socket
         |> assign(:flash_timer, nil)
         |> assign(flash_message: nil, flash_type: nil)}
      end
    end
  end
end
