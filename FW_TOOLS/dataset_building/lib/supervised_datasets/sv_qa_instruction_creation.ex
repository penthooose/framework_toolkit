defmodule SupervisedDatasets.QAInstructionCreation do
  @dataset_template """
  {
    "input": "<%= @input %>",
    "output": "<%= @output %>"
  }
  """

  @base_prompt_output """
    [ANTWORTEN]
    <%= @answers %>
    [ENDE ANTWORTEN]
  """

  @base_prompt_input """
    [INST]
    <%= @task %>:

    [KONTEXT]
    <%= @context %>
    [ENDE KONTEXT]

    [FRAGEN]
    <%= @questions %>
    [ENDE FRAGEN]

    [/INST]
  """

  @prompt_context """
    [AUSZUG <%= @context_num %>]
    <%= @sub_context %>
    [ENDE AUSZUG <%= @context_num %>]

  """

  @tasks [
    {:qa_b1,
     "Lese die folgenden Textausschnitte und beantworte anschließend die Fragen auf Basis der gegebenen Informationen."},
    {:qa_b2,
     "Basierend auf den bereitgestellten Textabschnitten, beantworte bitte die nachfolgenden Fragen."},
    {:qa_b3,
     "Analysiere die unten stehenden Textauszüge und beantworte die Fragen ausschließlich mit Informationen aus den Texten."},
    {:qa_b4,
     "Lies die gegebenen Textauszüge sorgfältig durch und beantworte die dazugehörigen Fragen anhand der vorliegenden Informationen."},
    {:qa_b5,
     "Verwende die folgenden Textpassagen, um die gestellten Fragen präzise zu beantworten. Stütze dich dabei nur auf die angegebenen Informationen."},
    {:qa_b6,
     "Nutze die bereitgestellten Textauszüge als Informationsquelle, um die nachfolgenden Fragen zu beantworten."}
  ]

  def create_instruction(instruction_params) do
    task = instruction_params.task
    context = instruction_params.context
    questions = instruction_params.questions
    answers = instruction_params.answers

    task_tuple = Enum.find(@tasks, fn {t, _} -> t == task end)
    task_template = if task_tuple, do: elem(task_tuple, 1), else: elem(Enum.at(@tasks, 0), 1)

    # Format context items with enumeration
    formatted_context = format_context(context)

    # Format questions and answers
    formatted_questions = Enum.join(questions, "\n")
    formatted_answers = Enum.join(answers, "\n")

    # Generate input and output
    input_string =
      EEx.eval_string(@base_prompt_input,
        assigns: %{
          task: task_template,
          context: formatted_context,
          questions: formatted_questions
        }
      )

    output_string =
      EEx.eval_string(@base_prompt_output,
        assigns: %{
          answers: formatted_answers
        }
      )

    # Combine into final dataset format
    instruction_string =
      EEx.eval_string(@dataset_template,
        assigns: %{
          input: input_string,
          output: output_string
        }
      )
      |> format_instruction_string()

    # Return formatted instruction
    instruction_string
  end

  # Helper function to format context items
  defp format_context(context) do
    context
    |> Enum.with_index(1)
    |> Enum.map(fn {text, index} ->
      EEx.eval_string(@prompt_context,
        assigns: %{
          context_num: index,
          sub_context: text
        }
      )
    end)
    |> Enum.join("")
  end

  defp format_instruction_string(instruction_string) do
    try do
      %{
        "input" => extract_input(instruction_string),
        "output" => extract_output(instruction_string)
      }
      |> Jason.encode!()
    rescue
      e ->
        IO.puts("Error in format_instruction_string: #{inspect(e)}")

        Jason.encode!(%{
          "input" => "[INST]Error processing input[/INST]",
          "output" => "Error processing output"
        })
    end
  end

  defp extract_input(instruction_string) do
    case Regex.run(~r/"input":\s*"(.*?)(?=",\s*"output")/s, instruction_string) do
      [_, captured] ->
        captured
        |> cleanup_string()

      _ ->
        "[INST]Error extracting input[/INST]"
    end
  end

  defp extract_output(instruction_string) do
    case Regex.run(~r/"output":\s*"(.*?)(?="(?:\s*\}))/s, instruction_string) do
      [_, captured] ->
        captured
        |> cleanup_string()

      _ ->
        "Error extracting output"
    end
  end

  defp cleanup_string(text) do
    text
    |> String.replace("\\\"", "\"")
    |> String.replace("\\r", "\r")
    |> String.replace("\\n", "\n")
    |> String.replace("\\\\", "\\")
    |> String.replace(~r/\r\n|\r/, "\n")
    |> String.replace(~r/\n\s+\n/, "\n\n")
    |> String.replace(~r/^\s+/, "")
    |> String.replace(~r/\s+$/, "")
  end
end
