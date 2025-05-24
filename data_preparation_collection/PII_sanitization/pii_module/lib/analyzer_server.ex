defmodule AnalyzerServer do
  use GenServer

  @presidio Path.join([File.cwd!(), "..", "Presidio"])

  def start_link(_) do
    if should_start_analyzer?() do
      GenServer.start_link(__MODULE__, %{python: nil}, name: __MODULE__)
    else
      :ignore
    end
  end

  def init(initial_state) do
    case :python.start_link([{:python, ~c"python"}, {:python_path, String.to_charlist(@presidio)}]) do
      {:ok, pid} ->
        # Give Python time to fully initialize
        Process.sleep(2000)
        :python.call(pid, :presidio_service, :create_analyzer_engine, [])
        {:ok, %{python: pid}}

      error ->
        IO.puts("Failed to start Python: #{inspect(error)}")
        {:stop, :python_start_failed}
    end
  end

  def handle_call(:get_analyzer, _from, state) do
    case state do
      %{python: pid} when is_pid(pid) -> {:reply, pid, state}
      _ -> {:reply, {:error, :no_python_process}, state}
    end
  end

  def handle_call(:restart, _from, _state) do
    {:ok, pid} =
      :python.start_link([{:python, ~c"python"}, {:python_path, String.to_charlist(@presidio)}])

    :python.call(pid, :presidio_service, :create_analyzer_engine, [])
    {:reply, :ok, pid}
  end

  def get_analyzer() do
    if should_start_analyzer?() do
      GenServer.call(__MODULE__, :get_analyzer)
    else
      :analyzer_disabled
    end
  end

  def restart(timeout \\ 15000) do
    IO.puts("[AnalyzerServer] Starting restart process...")

    current_pid = Process.whereis(__MODULE__)

    if current_pid do
      IO.puts("[AnalyzerServer] Stopping current server...")

      python_pid =
        try do
          GenServer.call(current_pid, :get_analyzer, 5000)
        catch
          :exit, _ -> nil
        end

      # Stop Python process if it exists
      if python_pid, do: :python.stop(python_pid)

      GenServer.stop(current_pid)
      # Give processes time to clean up
      Process.sleep(2000)
    end

    IO.puts("[AnalyzerServer] Starting new server instance...")

    # Just start the server and return :ok, don't wait for verification
    _result = start_link([])
    :ok
  end

  def ensure_running(timeout \\ 15000)

  def ensure_running(timeout) when is_integer(timeout) do
    case Process.whereis(__MODULE__) do
      nil ->
        IO.puts("[AnalyzerServer] No server running, starting new instance...")
        start_link([])
        :ok

      pid ->
        case GenServer.call(pid, :get_analyzer, timeout) do
          analyzer when is_pid(analyzer) -> :ok
          _error -> restart(timeout)
        end
    end
  end

  defp verify_analyzer(pid, timeout) do
    # Give Python time to initialize
    Process.sleep(2000)

    case GenServer.call(pid, :get_analyzer, timeout) do
      analyzer when is_pid(analyzer) ->
        IO.puts("[AnalyzerServer] Server successfully started and verified")
        :ok

      error ->
        IO.puts("[AnalyzerServer] Failed to verify analyzer: #{inspect(error)}")
        {:error, :analyzer_not_available}
    end
  end

  defp do_start_server(timeout, retries) when retries > 0 do
    case start_link([]) do
      {:ok, pid} ->
        {:ok, pid}

      {:error, {:already_started, _pid}} ->
        Process.sleep(1000)
        do_start_server(timeout, retries - 1)

      error ->
        error
    end
  end

  defp do_start_server(_timeout, 0), do: {:error, :max_retries_reached}

  defp purge_and_reload_modules(modules) do
    Enum.each(modules, fn module ->
      :code.purge(module)
      :code.delete(module)
      :code.load_file(module)
    end)
  end

  defp should_start_analyzer? do
    Application.get_env(:phoenix_UI, :analyzer, [])[:auto_start] || false
  end

  def terminate(_reason, state) do
    case state do
      %{python: pid} when is_pid(pid) ->
        :python.stop(pid)

      _ ->
        :ok
    end

    :ok
  end
end
