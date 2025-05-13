defmodule Phoenix_UIWeb.ServerHelpers do
  def restart do
    IO.puts("Restarting Phoenix server...")
    Application.stop(:phoenix_UI)
    Application.stop(:pii_module)
    :ok = Application.start(:pii_module)
    :ok = Application.start(:phoenix_UI)
    IO.puts("Phoenix server restarted!")
  end
end
