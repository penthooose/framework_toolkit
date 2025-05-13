defmodule GatewayAPI do
  @moduledoc """
  API to establish a connection to modules written in other languages, e.g. Python via ErlPort.
  Connection is established lazily only when needed.
  """

  use GenServer
  require Logger

  @python_path Path.expand(Path.join([__DIR__, "..", "external", "python"]))

  # Public API functions

  @doc """
  Ensures that the GatewayAPI is started and Python connection is established.
  Returns {:ok, pid} if successful or {:error, reason} if it fails.
  """
  def ensure_started(module) do
    case Process.whereis(__MODULE__) do
      nil ->
        # Gateway not started, start it manually
        Logger.info("GatewayAPI not found, starting manually")
        # Use start instead of start_link to avoid linking to the calling process
        GenServer.start(__MODULE__, [python_module: module], name: __MODULE__)

      pid when is_pid(pid) ->
        # Gateway is already started
        Logger.debug("GatewayAPI already running with pid #{inspect(pid)}")
        {:ok, pid}
    end
  end

  @doc """
  Restarts the GatewayAPI GenServer and its Python instance.
  This is useful when Python files have been modified while the application is running.

  Options:
  - file: Optional Python filename to reload specifically after a full restart

  Returns:
  - :ok if the restart was successful
  - {:error, reason} if restart failed
  """
  def restart_genserver(options \\ []) do
    file_to_reload = Keyword.get(options, :file)

    # Full restart logic - always performed
    Logger.info("Attempting to restart GatewayAPI GenServer")

    case Process.whereis(__MODULE__) do
      nil ->
        # GenServer isn't running, just start it
        Logger.info("GatewayAPI GenServer not found, starting fresh")
        ensure_started(file_to_reload)

      pid when is_pid(pid) ->
        # First stop the existing GenServer
        Logger.info("Stopping existing GatewayAPI GenServer")

        # Try graceful termination first
        try do
          GenServer.stop(pid, :normal, 5000)
        catch
          :exit, _ ->
            # If graceful termination fails, forcefully terminate
            Logger.warning("Graceful termination failed, forcing termination")
            Process.exit(pid, :kill)
        end

        # Give the system a moment to clean up
        Process.sleep(500)

        # Now start a new instance
        Logger.info("Starting fresh GatewayAPI GenServer")
        ensure_started(file_to_reload)
    end

    # Verify the restart was successful
    case Process.whereis(__MODULE__) do
      pid when is_pid(pid) ->
        # Check if the new instance works
        case test_connection(file_to_reload) do
          true ->
            Logger.info("GatewayAPI GenServer successfully restarted")
            :ok

          false ->
            Logger.error("GatewayAPI GenServer restarted but Python verification failed")

            {:error, :module_unavailable}
        end

      nil ->
        Logger.error("Failed to restart GatewayAPI GenServer")
        {:error, :restart_failed}
    end
  end

  @doc """
  Reloads a specific Python module without restarting the entire GenServer.
  This is useful for quick updates to individual Python files.

  Returns:
  - :ok if the module was reloaded successfully
  - {:error, reason} if the reload failed
  """
  def reload_python_module(module_name) when is_atom(module_name) do
    module_name_str = Atom.to_string(module_name)
    reload_python_module(module_name_str)
  end

  def reload_python_module(module_name) when is_binary(module_name) do
    # Remove .py extension if provided
    module_name = String.replace(module_name, ".py", "")

    Logger.info("Attempting to reload Python module: #{module_name}")

    with {:ok, pid} <- ensure_started(module_name),
         true <- Process.alive?(pid) do
      try do
        # Only try calling the module's own reload_module function
        case GenServer.call(__MODULE__, {:call, String.to_atom(module_name), :reload_module, []}) do
          {:ok, result} ->
            Logger.info("Custom reload for #{module_name}: #{result}")
            :ok

          {:error, _reason} ->
            Logger.info(
              "Function for reloading imports in Python file #{module_name} doesn't exist!"
            )

            :ok
        end
      catch
        kind, reason ->
          Logger.error("Error during Python module reload: #{inspect({kind, reason})}")
          {:error, {kind, reason}}
      end
    else
      false ->
        Logger.error("GatewayAPI GenServer not alive")
        {:error, :server_not_alive}

      error ->
        Logger.error("Failed to ensure GatewayAPI is started: #{inspect(error)}")
        error
    end
  end

  @doc """
  Check if a model is loaded in a specific Python module.
  Useful for determining if a reload would be expensive.
  """
  def check_model_loaded(module) do
    with {:ok, _pid} <- ensure_started(module) do
      try do
        case GenServer.call(__MODULE__, {:call, module, :is_model_loaded, []}) do
          {:ok, status} ->
            status = if is_list(status), do: List.to_string(status), else: "#{status}"
            {:ok, status}

          error ->
            error
        end
      catch
        _, _ -> {:error, :unavailable}
      end
    end
  end

  @doc """
  Call any Python function in a given module.
  module: atom name of the Python module
  function: atom name of the Python function
  args: list of arguments to pass
  options: map or boolean with the following supported keys:
    - restart: boolean, if true, restart the GenServer before the function call
    - reload: boolean, if true, reload the module before the function call
  timeout: optional timeout in milliseconds (defaults to standard GenServer timeout)

  For backward compatibility, a boolean can be passed instead of options map, which
  is equivalent to %{reload: value}

  Returns {:ok, result} or {:error, reason} from Python.
  """
  def call(module, function, args \\ [], options \\ nil, timeout \\ 5000) do
    # Handle the different ways options can be provided
    options =
      cond do
        # backward compatibility
        is_boolean(options) -> %{reload: options}
        is_map(options) -> options
        is_nil(options) -> %{}
        true -> %{}
      end

    # Check if restart is requested
    if Map.get(options, :restart, false) do
      Logger.info("Restarting GenServer before calling #{inspect(module)}.#{inspect(function)}")
      # Restart the entire GenServer
      case restart_genserver(file: module) do
        :ok ->
          Logger.debug("GenServer restarted successfully")

        {:error, reason} ->
          Logger.warning("Failed to restart GenServer: #{inspect(reason)}")
      end

      # If not restarting, check if reload is requested
    else
      if Map.get(options, :reload, false) do
        Logger.debug("Reloading module #{inspect(module)} before function call")
        # Just reload the module
        case reload_python_module(module) do
          :ok ->
            Logger.debug("Module #{inspect(module)} reloaded before function call")

          {:error, reason} ->
            Logger.warning("Failed to reload module #{inspect(module)}: #{inspect(reason)}")
        end
      end
    end

    # Now ensure the GenServer is started and call the function with the specified timeout
    with {:ok, _pid} <- ensure_started(module) do
      GenServer.call(__MODULE__, {:call, module, function, args}, timeout)
    end
  end

  @doc """
  Call any Python function in a given module with improved error formatting.
  Returns {:ok, result} on success or {:error, formatted_reason} on failure.
  """
  def call_with_formatted_errors(module, function, args \\ [], options \\ nil, timeout \\ 5000) do
    case call(module, function, args, options, timeout) do
      {:ok, result} ->
        {:ok, result}

      {:error, reason} ->
        {:error, format_python_error(reason)}
    end
  end

  @doc """
  Format Python errors into readable Elixir strings.
  """
  def format_python_error(error) do
    case error do
      {:exception, %ErlangError{original: {:python, error_type, error_msg, _traceback}}} ->
        error_type_str = Atom.to_string(error_type) |> String.replace("builtins.", "")

        error_msg_str =
          if is_list(error_msg),
            do: List.to_string(error_msg),
            else: "#{error_msg}"

        "Python #{error_type_str}: #{error_msg_str}"

      {:exception, %ErlangError{original: {:python, error_payload}}}
      when is_tuple(error_payload) ->
        "Python error: #{inspect(error_payload, pretty: true)}"

      # Handle other error formats
      {:exception, %ErlangError{original: original}} ->
        "Python error: #{inspect(original, pretty: true)}"

      # Catch-all for other error types
      err ->
        "Error: #{inspect(err, pretty: true)}"
    end
  end

  @doc """
  Register a callback function to receive progress updates from Python.
  The callback will be called with progress updates.
  """
  def register_progress_callback(pid_or_name \\ nil, module) do
    pid =
      cond do
        is_nil(pid_or_name) -> self()
        is_pid(pid_or_name) -> pid_or_name
        true -> Process.whereis(pid_or_name)
      end

    if is_nil(pid) do
      {:error, "Invalid process"}
    else
      with {:ok, _} <- ensure_started(module) do
        GenServer.call(__MODULE__, {:register_callback, pid, module})
      end
    end
  end

  @doc """
  Tests the connection to Python, optionally for a specific module.
  If module is provided, tests that specific Python module's connection.
  Returns true if connection is working, false otherwise.
  """
  def test_connection(module) do
    with {:ok, _pid} <- ensure_started(module) do
      Logger.info("Testing connection to Python module: #{module}")

      case GenServer.call(__MODULE__, {:test_connection, module}) do
        true ->
          Logger.info("Successfully connected to Python module: #{module}")
          true

        false ->
          Logger.error("Failed to connect to Python module: #{module}")
          false
      end
    else
      {:error, reason} ->
        Logger.error("Failed to ensure GatewayAPI is started: #{inspect(reason)}")
        false

      _ ->
        false
    end
  end

  # GenServer implementation

  def init(opts) do
    Logger.info("Starting Python via erlport with path: #{@python_path}")
    python_module = Keyword.get(opts, :python_module, nil)

    # Start Python instance with improved error handling
    try do
      python_options = [
        {:python_path, to_charlist(@python_path)},
        {:python, ~c"python"},
        {:cd, to_charlist(@python_path)}
      ]

      case :python.start(python_options) do
        {:ok, python} ->
          # Allow process to handle exits
          Process.flag(:trap_exit, true)

          # If a specific module was provided, test it, otherwise just succeed
          python_module = Keyword.get(opts, :python_module)

          if python_module do
            # Test the specific module
            case verify_python_connection(python, python_module) do
              true ->
                Logger.info("Python connection to #{python_module} established successfully")
                {:ok, %{python: python}}

              false ->
                Logger.error("Python verification failed for module: #{python_module}")
                :python.stop(python)
                {:stop, :python_verification_failed}
            end
          else
            # No specific module requested, just verify Python works
            Logger.info("Python connection established successfully (no specific module)")
            {:ok, %{python: python}}
          end

        {:error, reason} ->
          Logger.error("Failed to start Python: #{inspect(reason)}")
          {:stop, :python_start_failed}
      end
    rescue
      e ->
        Logger.error("Exception starting Python: #{inspect(e)}")
        {:stop, {:python_error, e}}
    end
  end

  # Add a verification function (missing in your general GatewayAPI)
  defp verify_python_connection(python, module) when is_atom(module) do
    verify_python_connection(python, Atom.to_string(module))
  end

  defp verify_python_connection(python, module) when is_binary(module) do
    try do
      # Create an atom from the module name
      module_atom = String.to_atom(module)

      # Try the module's own test_connection function first
      case :python.call(python, module_atom, :test_connection, []) do
        result when is_list(result) ->
          result_str = List.to_string(result)
          result_str == "ok"

        "ok" ->
          true

        _ ->
          false
      end
    rescue
      e ->
        Logger.error("Error verifying Python module #{module}: #{inspect(e)}")
        false
    catch
      kind, reason ->
        Logger.error(
          "Error caught when verifying Python module #{module}: #{inspect({kind, reason})}"
        )

        false
    end
  end

  def handle_call({:call, mod, fun, args}, _from, %{python: python} = state) do
    reply =
      try do
        {:ok, :python.call(python, mod, fun, args)}
      rescue
        e -> {:error, {:exception, e}}
      catch
        kind, reason -> {:error, {kind, reason}}
      end

    {:reply, reply, state}
  end

  # Add a new function clause to handle the case with extra arguments
  def handle_call({:call, mod, fun, args, _extra_args}, from, state) do
    # Forward to the standard handler, ignoring extra arguments
    handle_call({:call, mod, fun, args}, from, state)
  end

  def handle_call({:test_connection, module}, _from, %{python: python} = state) do
    module_atom = if is_atom(module), do: module, else: String.to_atom("#{module}")

    result =
      try do
        # Try calling the test_connection function of the module
        case :python.call(python, module_atom, :test_connection, []) do
          result when is_list(result) ->
            result_str = List.to_string(result)
            Logger.debug("Module #{module} test_connection result: #{result_str}")
            result_str == "ok"

          "ok" ->
            Logger.debug("Module #{module} test_connection returned ok")
            true

          other ->
            Logger.warning(
              "Module #{module} test_connection returned unexpected: #{inspect(other)}"
            )

            false
        end
      rescue
        e ->
          Logger.error("Error testing connection to #{module}: #{inspect(e)}")
          false
      catch
        kind, reason ->
          Logger.error("Error testing connection to #{module}: #{inspect({kind, reason})}")
          false
      end

    {:reply, result, state}
  end

  def handle_call({:reload_module, module_name}, _from, state) do
    # No longer trying to use importlib directly
    module_atom = String.to_atom(module_name)

    reply =
      try do
        case :python.call(state.python, module_atom, :reload_module, []) do
          result -> {:ok, result}
        end
      rescue
        e ->
          Logger.info(
            "Function for reloading imports in Python file #{module_name} doesn't exist!"
          )

          {:error, {:exception, e}}
      catch
        kind, reason ->
          Logger.info(
            "Function for reloading imports in Python file #{module_name} doesn't exist!"
          )

          {:error, {kind, reason}}
      end

    {:reply, reply, state}
  end

  # Add handler for registering callbacks
  def handle_call({:register_callback, pid, module}, _from, state) do
    # Register the callback process
    new_state = Map.put(state, :callback_pid, pid)
    module_atom = if is_atom(module), do: module, else: String.to_atom("#{module}")

    # Try to call the module's register_callback function directly
    reply =
      try do
        :python.call(state.python, module_atom, :register_progress_callback, [])
        {:ok, "Callback registered"}
      rescue
        e ->
          Logger.info("Function for registering callback in Python file #{module} doesn't exist!")
          {:error, {:exception, e}}
      catch
        kind, reason ->
          Logger.info("Function for registering callback in Python file #{module} doesn't exist!")
          {:error, {kind, reason}}
      end

    {:reply, reply, new_state}
  end

  # Handle process exits
  def handle_info({:EXIT, python_pid, reason}, %{python: python_pid} = state) do
    Logger.error("Python process exited: #{inspect(reason)}")
    # Attempt to restart Python
    try do
      :python.stop(python_pid)
    catch
      _, _ -> :ok
    end

    case :python.start([{:python_path, to_charlist(@python_path)}]) do
      {:ok, new_python} ->
        Logger.info("Successfully restarted Python process")
        {:noreply, %{state | python: new_python}}

      {:error, restart_reason} ->
        Logger.error("Failed to restart Python: #{inspect(restart_reason)}")
        {:stop, {:python_restart_failed, restart_reason}, state}
    end
  end

  def handle_info(msg, state) do
    Logger.debug("Unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  def terminate(_reason, %{python: python}) do
    :python.stop(python)
    :ok
  end
end
