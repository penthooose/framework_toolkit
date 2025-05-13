defmodule Phoenix_UI.State.PIIState do
  use Agent

  # either :anonymize or :pseudonymize
  @mode %{
    "mode" => :anonymize
  }

  # Update the agents list to use functions instead of module attributes
  @agents [
    {:operating_mode_agent, &__MODULE__.get_initial_mode/0},
    {:labels_agent, &__MODULE__.get_initial_labels/0},
    {:recognizers_agent, &__MODULE__.get_initial_recognizers/0},
    {:label_set_agent, &__MODULE__.get_initial_label_sets/0}
  ]

  @default_save_dir (
                      project_root = File.cwd!()
                      save_dir = Path.join([project_root, "files", "agent_save_files"])
                      # Ensure the directory exists
                      File.mkdir_p!(save_dir)
                      save_dir
                    )

  ### General functions for agents

  def start_agents(agent_name \\ :all) do
    # First ensure save directory exists
    File.mkdir_p!(@default_save_dir)

    agents_to_start =
      if agent_name == :all do
        @agents
      else
        Enum.filter(@agents, fn {name, _} -> name == agent_name end)
      end

    Enum.map(agents_to_start, fn {name, initial_state} ->
      case start_agent(name, initial_state) do
        {:ok, pid} ->
          {:ok, pid}

        {:error, {:already_started, pid}} ->
          {:ok, pid}

        {:error, reason} ->
          IO.puts("Failed to start agent #{name}: #{inspect(reason)}")
          {:error, reason}
      end
    end)
  end

  defp start_agent(name, initial_state) when is_function(initial_state, 0) do
    # Handle function-based initial states
    Agent.start_link(initial_state, name: name)
  end

  defp start_agent(name, initial_state) do
    # Handle regular initial states
    Agent.start_link(fn -> initial_state end, name: name)
  end

  def print_available_agent_names do
    Enum.each(@agents, fn {name, _initial_state} ->
      IO.puts(inspect(name))
    end)
  end

  def ensure_running(agent_name \\ :all) do
    if agent_name == :all do
      Enum.each(@agents, fn {name, _initial_state} ->
        ensure_running(name)
      end)

      :ok
    else
      case :erlang.whereis(agent_name) do
        :undefined ->
          start_agents(agent_name)

        pid when is_pid(pid) ->
          if Process.alive?(pid), do: {:ok, pid}, else: start_agents(agent_name)
      end
    end
  end

  def get_status(agent_name \\ :all) do
    agents_to_check =
      if agent_name == :all do
        @agents
      else
        Enum.filter(@agents, fn {name, _} -> name == agent_name end)
      end

    Enum.map(agents_to_check, fn {name, _initial_state} ->
      case :erlang.whereis(name) do
        :undefined ->
          {name, :not_running}

        pid when is_pid(pid) ->
          {name, pid}
      end
    end)
  end

  def shutdown_agents(agent_name \\ :all) do
    Enum.each(@agents, fn {name, _state} ->
      if agent_name == :all or agent_name == name do
        shutdown_agent(name)
      end
    end)
  end

  defp shutdown_agent(name) do
    case :erlang.whereis(name) do
      :undefined ->
        IO.puts("Shutdown: Agent #{name} is not running.")

      pid when is_pid(pid) ->
        Agent.stop(pid)
        IO.puts("Shut down agent #{name}.")
    end
  end

  def reset_agents(agent_name \\ :all) do
    Enum.each(@agents, fn {name, _initial_state} ->
      if agent_name == :all or agent_name == name do
        shutdown_agent(name)
        start_agents(name)
        IO.puts("Restarted agent #{name}.")
      end
    end)
  end

  def get_initial_labels do
    labels = MainPii.get_all_supported_entities_erlport()

    Enum.map(labels, fn label ->
      %{
        "recognizer_name" => label,
        "active" => false
      }
    end)

    active_labels = get_active_labels_in_label_sets()

    Enum.map(labels, fn label ->
      %{
        "recognizer_name" => label,
        "active" => label in active_labels
      }
    end)
  end

  def get_initial_recognizers do
    MainPii.get_all_recognizers_erlport()
  end

  def get_initial_label_sets do
    # Try to load from file first, if that fails use default label sets
    case load_state_from_file(:label_set_agent) do
      {:ok, state} ->
        state
        # _ -> @label_sets
    end
  end

  def get_initial_mode do
    @mode
  end

  def reset_agents_to_savefiles(directory \\ @default_save_dir) do
    IO.inspect(directory)

    Enum.map(@agents, fn {agent_name, _} ->
      case load_from_file(agent_name, directory) do
        {:ok, state} ->
          {:ok, agent_name, state}

        {:error, reason} ->
          {:error, agent_name, reason}
      end
    end)
  end

  def save_to_file(agentname, directory \\ @default_save_dir) do
    # Ensure directory exists
    File.mkdir_p!(directory)
    savefile = Path.expand(Path.join(directory, "#{Atom.to_string(agentname)}_savefile.json"))

    path = normalize_path(savefile)

    case Agent.get(agentname, & &1) do
      nil ->
        {:error, "Agent #{agentname} is not running or does not exist"}

      state ->
        serialized_state = serialize_atoms(state)

        with {:ok, json} <- Jason.encode(serialized_state),
             :ok <- File.write(path, json) do
          {:ok, path}
        else
          {:error, reason} -> {:error, reason}
        end
    end
  end

  defp load_state_from_file(agentname, directory \\ @default_save_dir) do
    savefile = Path.expand(Path.join(directory, "#{Atom.to_string(agentname)}_savefile.json"))
    path = normalize_path(savefile)

    if File.exists?(path) do
      with {:ok, json} <- File.read(path),
           {:ok, raw_state} <- Jason.decode(json),
           state <- deserialize_atoms(raw_state) do
        # IO.inspect(state, label: "Loaded state for #{agentname}")
        {:ok, state}
      else
        error -> error
      end
    else
      {:error, "Save file for #{agentname} does not exist"}
    end
  end

  def load_from_file(agentname, directory \\ @default_save_dir) do
    case load_state_from_file(agentname, directory) do
      {:ok, state} ->
        shutdown_agent(agentname)

        IO.puts("Restarting #{agentname}.")
        Agent.start_link(fn -> state end, name: agentname)
        {:ok, state}

      error ->
        error
    end
  end

  def normalize_path(path) do
    # Convert to unix-style path first for consistency
    unix_path = String.replace(path, "\\", "/")
    # Then normalize based on OS
    savefile =
      if String.contains?(unix_path, ":") do
        # Windows path with drive letter
        String.replace(unix_path, "/", "\\")
      else
        unix_path
      end
  end

  defp serialize_atoms(data) do
    # needed for representation of atoms / elixir types in JSON file
    cond do
      is_map(data) ->
        Enum.into(data, %{}, fn {key, value} ->
          {serialize_atoms(key), serialize_atoms(value)}
        end)

      is_list(data) ->
        Enum.map(data, &serialize_atoms/1)

      is_atom(data) ->
        "@atom:#{Atom.to_string(data)}"

      true ->
        data
    end
  end

  defp deserialize_atoms(data) do
    # convert strings that are tagged as atoms / elixir types back to atoms
    cond do
      is_map(data) ->
        Enum.into(data, %{}, fn {key, value} ->
          {deserialize_atoms(key), deserialize_atoms(value)}
        end)

      is_list(data) ->
        Enum.map(data, &deserialize_atoms/1)

      is_binary(data) and String.starts_with?(data, "@atom:") ->
        String.trim_leading(data, "@atom:") |> String.to_atom()

      true ->
        data
    end
  end

  ### Functions to retrieve data from agents (Getter)

  def get_save_dir do
    @default_save_dir
  end

  def get_labels do
    ensure_running(:labels_agent)

    case :erlang.whereis(:labels_agent) do
      :undefined -> IO.puts("Labels agent is not running.") && {:error, :agent_not_running}
      pid when is_pid(pid) -> Agent.get(pid, fn labels -> labels end)
    end
  end

  def get_labels_names do
    case get_labels() do
      {:error, :agent_not_running} -> []
      labels -> Enum.map(labels, fn label -> label["recognizer_name"] end)
    end
  end

  def get_mode do
    case :erlang.whereis(:operating_mode_agent) do
      :undefined ->
        IO.puts("Operating mode agent is not running.") && {:error, :agent_not_running}

      pid when is_pid(pid) ->
        Agent.get(pid, fn state -> state["mode"] end)
    end
  end

  def get_recognizers do
    case :erlang.whereis(:recognizers_agent) do
      :undefined -> IO.puts("Recognizers agent is not running.") && {:error, :agent_not_running}
      pid when is_pid(pid) -> Agent.get(pid, fn recognizers -> recognizers end)
    end
  end

  def get_label_sets do
    ensure_running(:label_set_agent)

    case :erlang.whereis(:label_set_agent) do
      :undefined ->
        IO.puts("Label set agent is not running.")
        {:error, :agent_not_running}

      pid when is_pid(pid) ->
        Agent.get(pid, fn state -> state end)
    end
  end

  def get_label_sets_names do
    case get_label_sets() do
      {:error, :agent_not_running} -> []
      label_set -> Enum.map(label_set, fn entry -> Map.get(entry, "name") end)
    end
  end

  def get_active_label_sets_names do
    case get_label_sets() do
      {:error, :agent_not_running} ->
        []

      label_set ->
        label_set
        |> Enum.filter(fn entry -> Map.get(entry, "active") == true end)
        |> Enum.map(fn entry -> Map.get(entry, "name") end)
    end
  end

  def get_active_labels_in_label_sets do
    case get_label_sets() do
      {:error, :agent_not_running} ->
        []

      label_set ->
        label_set
        |> Enum.filter(fn entry -> Map.get(entry, "active") == true end)
        |> Enum.flat_map(fn entry -> Map.get(entry, "supported_entities") end)
        |> Enum.uniq()
    end
  end

  def get_active_labels_names do
    case get_labels() do
      {:error, :agent_not_running} ->
        []

      labels ->
        Enum.filter(labels, fn label -> label["active"] == true end)
        |> Enum.map(fn label -> label["recognizer_name"] end)
    end
  end

  ### Functions for manipulating data in agents (Setter)

  def create_new_label(label_name) do
    new_label = %{
      "recognizer_name" => label_name,
      "active" => true
    }

    Agent.update(:labels_agent, fn labels ->
      if Enum.any?(labels, fn l -> l["recognizer_name"] == label_name end) do
        labels
      else
        [new_label | labels]
      end
    end)
  end

  def remove_label(label) do
    Agent.update(:labels_agent, fn labels ->
      Enum.reject(labels, fn existing_label ->
        existing_label["recognizer_name"] == label
      end)
    end)

    remove_label_from_all_label_sets(label)
  end

  def add_label_to_label_set(label_set_name, label) do
    ensure_running(:label_agent)
    ensure_running(:label_set_agent)

    case :erlang.whereis(:labels_agent) do
      :undefined ->
        IO.puts("Label agent is not running.") && {:error, :agent_not_running}

      pid when is_pid(pid) ->
        unless Enum.any?(Agent.get(pid, & &1), fn l -> l["recognizer_name"] == label end) do
          IO.puts("Label #{label} not found")
          {:error, :label_not_found}
        else
          case(:erlang.whereis(:label_set_agent)) do
            :undefined ->
              IO.puts("Label set agent is not running.")
              {:error, :agent_not_running}

            pid2 when is_pid(pid2) ->
              Agent.update(pid2, fn label_sets ->
                case Enum.find(label_sets, fn set -> set["name"] == label_set_name end) do
                  nil ->
                    IO.puts("Label set #{label_set_name} not found")
                    label_sets

                  label_set ->
                    if label in label_set["supported_entities"] do
                      label_sets
                    else
                      Enum.map(label_sets, fn set ->
                        if set["name"] == label_set_name do
                          Map.update!(set, "supported_entities", fn entities ->
                            [label | entities]
                          end)
                        else
                          set
                        end
                      end)
                    end
                end
              end)

              case save_to_file(:label_set_agent) do
                {:ok, _path} -> :ok
                {:error, reason} -> {:error, reason}
              end
          end
        end
    end
  end

  def remove_label_from_all_label_sets(label) do
    label_set_names = get_label_sets_names()

    IO.inspect(label_set_names)
    IO.inspect(label)

    Enum.each(label_set_names, fn label_set_name ->
      remove_label_from_label_set(label_set_name, ~c"#{label}")
    end)

    label_sets = get_label_sets()
    IO.inspect(label_sets)
  end

  def remove_label_from_label_set(label_set_name, label) do
    ensure_running(:label_set_agent)

    case :erlang.whereis(:label_set_agent) do
      :undefined ->
        IO.puts("Label set agent is not running.") && {:error, :agent_not_running}

      pid when is_pid(pid) ->
        Agent.update(pid, fn label_sets ->
          Enum.map(label_sets, fn set ->
            if set["name"] == label_set_name do
              Map.update!(set, "supported_entities", fn entities ->
                Enum.reject(entities, fn entity -> entity == label end)
              end)
            else
              set
            end
          end)
        end)

        case save_to_file(:label_set_agent) do
          {:ok, _path} -> :ok
          {:error, reason} -> {:error, reason}
        end
    end
  end

  def create_new_label_set(label_set_name, supported_entities) do
    new_label_set = %{
      "name" => label_set_name,
      "supported_entities" => supported_entities,
      "active" => true
    }

    Agent.update(:label_set_agent, fn label_sets ->
      if Enum.any?(label_sets, fn set -> set["name"] == label_set_name end) do
        label_sets
      else
        [new_label_set | label_sets]
      end
    end)

    case save_to_file(:label_set_agent) do
      {:ok, _path} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  def remove_label_set(label_set_name) do
    Agent.update(:label_set_agent, fn label_sets ->
      Enum.reject(label_sets, fn set -> set["name"] == label_set_name end)
    end)

    case save_to_file(:label_set_agent) do
      {:ok, _path} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  def set_mode(new_mode) when new_mode in [:anonymize, :pseudonymize] do
    case :erlang.whereis(:operating_mode_agent) do
      :undefined ->
        IO.puts("Operating mode agent is not running.") && {:error, :agent_not_running}

      pid when is_pid(pid) ->
        Agent.update(pid, fn state ->
          Map.put(state, "mode", new_mode)
        end)

        :ok
    end
  end

  def toggle_label_set_active(label_set_name) do
    ensure_running(:label_set_agent)

    case :erlang.whereis(:label_set_agent) do
      :undefined ->
        IO.puts("Label set agent is not running.") && {:error, :agent_not_running}

      pid when is_pid(pid) ->
        # Check if label_set_name exists in state
        unless Enum.any?(Agent.get(pid, & &1), fn set -> set["name"] == label_set_name end) do
          IO.puts("Label set #{label_set_name} not found")
          throw({:error, :label_set_not_found})
        end

        Agent.update(pid, fn state ->
          Enum.map(state, fn entry ->
            if Map.get(entry, "name") == label_set_name do
              Map.put(entry, "active", !Map.get(entry, "active"))
            else
              entry
            end
          end)
        end)

        # Get the updated state after toggling
        state = Agent.get(pid, & &1)
        # Find the toggled label set
        label_set = Enum.find(state, fn set -> set["name"] == label_set_name end)

        if label_set do
          # If activating, simply activate all labels in the set
          if label_set["active"] do
            Enum.each(label_set["supported_entities"], fn entity ->
              Agent.update(:labels_agent, fn labels ->
                Enum.map(labels, fn label ->
                  if label["recognizer_name"] == entity do
                    Map.put(label, "active", true)
                  else
                    label
                  end
                end)
              end)
            end)
          else
            # If deactivating, only deactivate labels that aren't in other active sets
            active_entities = get_active_labels_in_label_sets()

            Enum.each(label_set["supported_entities"], fn entity ->
              Agent.update(:labels_agent, fn labels ->
                Enum.map(labels, fn label ->
                  if label["recognizer_name"] == entity and entity not in active_entities do
                    Map.put(label, "active", false)
                  else
                    label
                  end
                end)
              end)
            end)
          end
        end

        case save_to_file(:label_set_agent) do
          {:ok, _path} -> :ok
          {:error, reason} -> {:error, reason}
        end
    end
  end

  def toggle_label_active(label) do
    ensure_running(:labels_agent)

    case :erlang.whereis(:labels_agent) do
      :undefined ->
        IO.puts("Labels agent is not running.") && {:error, :agent_not_running}

      pid when is_pid(pid) ->
        unless Enum.any?(Agent.get(pid, & &1), fn l -> l["recognizer_name"] == label end) do
          IO.puts("Label #{label} not found")
          throw({:error, :label_not_found})
        end

        Agent.update(pid, fn labels ->
          Enum.map(labels, fn existing_label ->
            if existing_label["recognizer_name"] == label do
              Map.put(existing_label, "active", !Map.get(existing_label, "active"))
            else
              existing_label
            end
          end)
        end)

        :ok
    end
  end

  def remove_custom_label(label) do
    MainPii.remove_label_from_custom_patterns(label)
    reset_agents(:labels_agent)
    remove_label_from_all_label_sets(label)
    reset_agents(:label_set_agent)
    {:ok, "Custom recognizer #{label} removed successfully"}
  end

  def remove_all_custom_labels do
    MainPii.remove_all_labels_from_custom_patterns()
    reset_agents(:labels_agent)
    custom_labels = MainPii.get_custom_recognizer_entities()
    Enum.each(custom_labels, fn label -> remove_label_from_all_label_sets(label) end)
    reset_agents(:label_set_agent)
    {:ok, "All custom recognizers removed successfully"}
  end
end
