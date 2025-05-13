defmodule MainPii do
  alias APIClient
  alias PatternRecognizer
  alias RegexGenerator

  @presidio Path.join([File.cwd!(), "..", "Presidio"])

  def testmodule do
    IO.puts("Hello from MainPii")
  end

  def restart_analyzer_server do
    IO.puts("[MainPii] Restarting analyzer server...")

    case AnalyzerServer.restart() do
      :ok ->
        IO.puts("[MainPii] Analyzer server restarted successfully")
        :ok

      {:error, _reason} = error ->
        case AnalyzerServer.ensure_running() do
          :ok ->
            IO.puts("[MainPii] Analyzer server is now running")
            :ok

          err ->
            IO.puts("[MainPii] Failed to start analyzer server: #{inspect(err)}")
            error
        end
    end
  end

  def reset_env_variables do
    {machine_path, 0} =
      System.cmd("cmd", ["/c", "echo %PATH%"])

    {user_path, 0} =
      System.cmd("cmd", ["/c", "echo %PATH%"])

    combined_path = "#{String.trim(machine_path)};#{String.trim(user_path)}"
    System.put_env("PATH", combined_path)

    {machine_pythonhome, 0} =
      System.cmd("cmd", ["/c", "echo %PYTHONHOME%"])

    System.put_env("PYTHONHOME", machine_pythonhome)

    {machine_pythonpath, 0} =
      System.cmd("cmd", ["/c", "echo %PYTHONPATH%"])

    System.put_env("PYTHONPATH", machine_pythonpath)
  end

  def run_erlport_test do
    {python_path, 0} = System.cmd("python", ["-c", "import sys; print(sys.prefix)"])
    python_home = "C:/SDKs/Python310"
    modules_dir = "#{python_home}/my_modules"

    {:ok, pid} =
      :python.start_link([
        {:python, ~c"python"},
        {:python_path, String.to_charlist(modules_dir)}
      ])

    IO.puts("Python started successfully.")

    try do
      python_module = :test

      result = :python.call(pid, python_module, :print_success, [])
      :python.stop(pid)
      result
    catch
      kind, error ->
        IO.puts("Error: #{inspect(kind)} - #{inspect(error)}")
        :python.stop(pid)
        {:error, error}
    end
  end

  def run_health_check_api do
    case APIClient.health_check() do
      {:ok, response} ->
        IO.inspect(response, label: "Service Health Check Response")

      {:error, reason} ->
        IO.puts("Service is down: #{reason}")
    end
  end

  defp with_python(fun) do
    AnalyzerServer.ensure_running()
    pid = AnalyzerServer.get_analyzer()

    try do
      result = fun.(pid)
      result
    catch
      kind, error ->
        IO.puts("Error: #{inspect(kind)} - #{inspect(error)}")
        {:error, error}
    end
  end

  def get_all_recognizers_erlport() do
    with_python(fn pid ->
      :python.call(pid, :presidio_service, :get_all_recognizers, [])
    end)
  end

  def get_all_supported_entities_erlport() do
    with_python(fn pid ->
      :python.call(pid, :presidio_service, :get_all_supported_entities, [])
    end)
  end

  def analyze_text_erlport(text, active_labels, language) do
    IO.inspect(language, label: "Language")

    with_python(fn pid ->
      :python.call(pid, :presidio_service, :analyze_text, [text, active_labels, language])
    end)
  end

  def analyze_file_erlport() do
    file_path = Path.join([File.cwd!(), "files", "analyze_input.json"])
    IO.inspect(file_path)

    with_python(fn pid ->
      :python.call(pid, :presidio_service, :analyze_file, [file_path])
    end)
  end

  def analyze_file_and_anonymize_erlport() do
    file_path = Path.join([File.cwd!(), "files", "analyze_input.json"])
    IO.inspect(file_path)

    with_python(fn pid ->
      :python.call(pid, :presidio_service, :analyze_file_and_anonymize, [file_path])
    end)
  end

  def anonymize_text_erlport(text, active_labels, language, get_analysis_data) do
    IO.inspect(get_analysis_data, label: "Get Analysis Data in Main PII")

    with_python(fn pid ->
      :python.call(pid, :presidio_service, :anonymize_text, [
        text,
        active_labels,
        language,
        get_analysis_data
      ])
    end)
  end

  def pseudonymize_text_erlport(text, active_labels, language, get_analysis_data) do
    with_python(fn pid ->
      :python.call(pid, :presidio_service, :pseudonymize_text, [
        text,
        active_labels,
        language,
        get_analysis_data
      ])
    end)
  end

  def get_custom_recognizers do
    with_python(fn pid ->
      result = :python.call(pid, :presidio_service, :get_custom_recognizers, [])

      result
    end)
  end

  def get_all_recognizers do
    with_python(fn pid ->
      :python.call(pid, :presidio_service, :get_all_recognizers, [])
    end)
  end

  # deprecated
  def add_pattern_recognizer_with_examples_erlport(
        name,
        examples,
        context_list \\ [],
        language \\ "any"
      ) do
    IO.puts("\n[MainPii] Starting add_pattern_recognizer_with_examples_erlport")
    regex = RegexGenerator.derive_regex(examples)
    IO.puts("[MainPii] Derived regex: #{inspect(regex)}")

    context = Enum.map(context_list, &to_string/1)
    IO.puts("[MainPii] Processed context: #{inspect(context)}")
    IO.puts("[MainPii] Language: #{inspect(language)}")

    with_python(fn pid ->
      IO.puts("[MainPii] Calling Python service...")

      result =
        :python.call(pid, :presidio_service, :add_pattern_recognizer, [
          to_string(name),
          to_string(regex),
          context,
          language
        ])

      IO.puts("[MainPii] Python service returned: #{inspect(result)}")
      result
    end)
  end

  def add_pattern_recognizer_with_regex_erlport(
        name,
        regex,
        context_list \\ [],
        language \\ "any"
      ) do
    context = Enum.map(context_list, &to_string/1)
    IO.puts("[MainPii] Derived regex: #{inspect(regex)}")
    IO.puts("[MainPii] Processed context: #{inspect(context)}")
    IO.puts("[MainPii] Language: #{inspect(language)}")

    with_python(fn pid ->
      :python.call(pid, :presidio_service, :add_pattern_recognizer, [
        to_string(name),
        to_string(regex),
        context,
        language
      ])
    end)
  end

  def add_deny_list_recognizer_erlport(name, deny_list, context_list \\ [], language \\ "de") do
    context = Enum.map(context_list, &to_string/1)
    deny_list_string = Enum.map(deny_list, &to_string/1)
    IO.puts("[MainPii] Processed deny list: #{inspect(deny_list_string)}")
    IO.puts("[MainPii] Processed context: #{inspect(context)}")
    IO.puts("[MainPii] Language: #{inspect(language)}")

    with_python(fn pid ->
      :python.call(pid, :presidio_service, :add_deny_list_recognizer, [
        to_string(name),
        deny_list_string,
        context,
        language
      ])
    end)
  end

  def remove_label_from_custom_patterns(label) do
    custom_patterns = Path.join(@presidio, "custom_recognizers.yaml")
    yaml_content = YamlElixir.read_from_file!(custom_patterns)
    label = to_string(label)

    original_recognizers = Map.get(yaml_content, "recognizers", [])

    updated_recognizers =
      original_recognizers
      |> Enum.reject(fn recognizer ->
        Map.get(recognizer, "supported_entity") == label
      end)

    if length(original_recognizers) == length(updated_recognizers) do
      {:ok, "No such entity found in custom recognizers."}
    else
      updated_yaml = %{"recognizers" => updated_recognizers}
      yaml_string = Ymlr.document!(updated_yaml)
      File.write!(custom_patterns, yaml_string)
      {:ok, "Label #{label} removed successfully."}
    end
  end

  def remove_all_labels_from_custom_patterns() do
    custom_patterns = Path.join(@presidio, "custom_recognizers.yaml")
    yaml_string = Ymlr.document!(%{"recognizers" => []})
    File.write!(custom_patterns, yaml_string)
    {:ok, "All labels removed successfully."}
  end

  def get_custom_recognizer_entities() do
    custom_patterns = Path.join(@presidio, "custom_recognizers.yaml")
    yaml_content = YamlElixir.read_from_file!(custom_patterns)

    yaml_content
    |> Map.get("recognizers", [])
    |> Enum.map(&Map.get(&1, "supported_entity"))
  end

  def get_custom_recognizers_from_registry_erlport() do
    with_python(fn pid ->
      :python.call(pid, :presidio_service, :get_custom_recognizers_from_registry, [])
    end)
  end

  def get_custom_recognizers_erlport() do
    with_python(fn pid ->
      :python.call(pid, :presidio_service, :get_custom_recognizers, [])
    end)
  end

  def remove_recognizer(name) do
    with_python(fn pid ->
      case :python.call(pid, :presidio_service, :remove_recognizer_from_savefile, [name]) do
        true ->
          PIIState.reset_agents(:labels_agent)
          true

        false ->
          false
      end
    end)
  end

  def remove_all_recognizers() do
    custom_recognizers = get_custom_recognizers()

    # Remove each recognizer
    Enum.each(custom_recognizers, fn recognizer ->
      name = Map.get(recognizer, "name")
      remove_recognizer(name)
    end)

    # Reset state after removing all
    PIIState.reset_agents(:labels_agent)
  end
end
