defmodule SupervisedDatasets.MixedInstructionCreation do
  @dataset_template """
  {
    "input": "<%= @input %>",
    "output": "<%= @output %>"
  }
  """

  @base_prompt_output """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @chapter_content %>
    [ENDE KAPITEL <%= @chapter_num %>]
  """

  @base_prompt_input """
    [INST]
    <%= @task %>:

    <%= @previous_content %>

    <%= @meta_data %>

    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]

    <%= @current_chapter_summary %>
    [/INST]
  """

  @base_prompt_technical """
    [INST]
    <%= @task %>:

    <%= @previous_content %>

    <%= @meta_data %>

    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]

    [/INST]
  """

  @prompt_previous_content_chapters """
    [VORHERIGE INHALTE]
    <%= @previous_content %>
    [ENDE VORHERIGE INHALTE]
  """

  @prompt_previous_main_chapter_no_content """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @previous_sub_chapters %>
    [ENDE KAPITEL <%= @chapter_num %>]

    <%= @previous_chapter_following %>
  """

  @prompt_previous_main_chapter_with_content """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @chapter_content %>

    <%= @previous_sub_chapters %>
    [ENDE KAPITEL <%= @chapter_num %>]

    <%= @previous_chapter_following %>
  """

  @prompt_previous_main_chapter_no_closing """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @chapter_content %>

    <%= @previous_sub_chapters %>
  """

  @prompt_previous_main_chapter_no_closing_no_content """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @previous_sub_chapters %>
  """

  @prompt_previous_sub_chapter """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @chapter_content %>
    [ENDE KAPITEL <%= @chapter_num %>]

    <%= @previous_chapter_following %>
  """

  @prompt_previous_sub_chapter_closing """
    [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @chapter_content %>
    [ENDE KAPITEL <%= @chapter_num %>]
  """

  @prompt_previous_content_summaries """
    [ZUSAMMENFASSUNG VORHERIGE INHALTE]
    <%= @previous_content %>
    [ENDE ZUSAMMENFASSUNG VORHERIGE INHALTE]
  """

  @prompt_previous_main_summary """
    [ZUSAMMENFASSUNG VON KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @previous_summary_content %>
    [ENDE ZUSAMMENFASSUNG VON KAPITEL <%= @chapter_num %>]
  """

  @prompt_previous_sub_summary_closing """
    [ZUSAMMENFASSUNG VON KAPITEL <%= @chapter_num %>: <%= @chapter_name %>]
    <%= @previous_summary_content %>
    [ENDE ZUSAMMENFASSUNG VON KAPITEL <%= @chapter_num %>]
  """

  @prompt_summary """
    [KAPITELINHALTE]
    <%= @previous_summary_content %>
    [ENDE KAPITELINHALTE]

    <%= @previous_summary_following %>
  """

  @prompt_meta_data """
    [METADATEN]
    <%= @meta_data_objects %>
    [ENDE METADATEN]
  """

  @tasks [
    {:create_chapter_b1,
     "Erstelle einen Kapitelinhalt für [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] eines Gutachtens über ein Medizinprodukt mit den folgenden Informationen"},
    {:create_chapter_b2,
     "Verfasse den Inhalt des Kapitels [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] eines Gutachtens über ein Medizingerät basierend auf den gegebenen Daten"},
    {:create_chapter_b3,
     "Formuliere [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] für ein technisches Gutachten über ein medizintechnisches Gerät anhand der folgenden Informationen"},
    {:create_chapter_b4,
     "Schreibe [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] als Gutachtenabschnitt über ein medizintechnisches Gerät mit den bereitgestellten Daten"},
    {:create_chapter_b5,
     "Erzeuge einen fachlich korrekten Gutachtentext für [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] über ein Medizinprodukt"},
    {:create_chapter_b6,
     "Verfasse den Inhalt von [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] eines Gutachtens über ein Medizinprodukt auf der Grundlage der vorgegebenen Daten"},
    {:create_chapter_b7,
     "Verfasse den Inhalt von [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] eines Gutachtens über ein Medizinprodukt anhand der vorgegebenen Informationen"},
    {:create_technical_data_b1,
     "Erstelle für [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] eine strukturierte Auflistung der technischen Gerätedaten zu einem medizinischen Gerät mit folgenden Spezifikationen"},
    {:create_technical_data_b2,
     "Formatiere für [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] die im Folgenden enthaltenen technischen Daten des begutachteten Medizinprodukts in übersichtlicher Auflistung"},
    {:create_technical_data_b3,
     "Fasse für [KAPITEL <%= @chapter_num %>: <%= @chapter_name %>] die technischen Spezifikationen des zu bewertenden Medizingeräts in einer klar strukturierten Auflistung der Gerätedaten zusammen"}
  ]

  def create_instruction(instruction_params) do
    task = instruction_params.task
    chapter_num = instruction_params.chapter_num
    chapter_name = instruction_params.chapter_name
    meta_data = instruction_params.meta_data
    category = instruction_params.category
    type = instruction_params.type
    summary = instruction_params.summary
    content = instruction_params.content
    previous_content = instruction_params.previous_content

    if type == "technical" do
      task_string = Atom.to_string(task)
      task_last_char = String.at(task_string, -1)
      task_integer = String.to_integer(task_last_char)
      modulo_task = rem(task_integer, 3) + 1
      technical_task = String.to_atom("create_technical_data_b#{modulo_task}")

      create_technical_data_instruction(
        technical_task,
        chapter_num,
        chapter_name,
        meta_data,
        category,
        previous_content
      )
    else
      create_regular_instruction(
        task,
        chapter_num,
        chapter_name,
        meta_data,
        category,
        summary,
        content,
        previous_content
      )
    end
  end

  def create_technical_data_instruction(
        task,
        chapter_num,
        chapter_name,
        meta_data,
        category,
        previous_content
      ) do
    task_tuple = Enum.find(@tasks, fn {t, _} -> t == task end)
    task_template = if task_tuple, do: elem(task_tuple, 1), else: elem(Enum.at(@tasks, 7), 1)

    task_description =
      EEx.eval_string(task_template,
        assigns: %{
          chapter_num: chapter_num,
          chapter_name: chapter_name
        }
      )

    previous_content_string = format_previous_content_for_technical(previous_content)
    meta_data_string = format_meta_data(meta_data)

    input_string =
      EEx.eval_string(@base_prompt_technical,
        assigns: %{
          task: task_description,
          previous_content: previous_content_string,
          meta_data: meta_data_string,
          chapter_num: chapter_num,
          chapter_name: chapter_name
        }
      )

    output_string =
      EEx.eval_string(@base_prompt_output,
        assigns: %{
          chapter_num: chapter_num,
          chapter_name: chapter_name,
          chapter_content: ""
        }
      )

    instruction_string =
      EEx.eval_string(@dataset_template,
        assigns: %{
          input: input_string,
          output: output_string
        }
      )
      |> format_instruction_string()

    {instruction_string, category}
  end

  def create_regular_instruction(
        task,
        chapter_num,
        chapter_name,
        meta_data,
        category,
        summary,
        content,
        previous_content
      ) do
    task_tuple = Enum.find(@tasks, fn {t, _} -> t == task end)
    task_template = if task_tuple, do: elem(task_tuple, 1), else: elem(Enum.at(@tasks, 0), 1)

    task_description =
      EEx.eval_string(task_template,
        assigns: %{
          chapter_num: chapter_num,
          chapter_name: chapter_name
        }
      )

    previous_content_string = format_previous_content(previous_content, category)
    cleaned_content = cleanup_chapter_content(content)

    current_chapter_summary =
      if summary && summary != "" && !contains_restricted_tags?(summary) do
        normalized_summary =
          summary
          |> remove_footnote_tags()
          |> remove_image_lines()
          |> remove_special_tags()
          |> remove_chapter_header(chapter_num, chapter_name)
          |> normalize_newlines()

        EEx.eval_string(@prompt_summary,
          assigns: %{
            previous_summary_content: normalized_summary,
            previous_summary_following: ""
          }
        )
      else
        ""
      end

    meta_data_string = format_meta_data(meta_data)

    input_string =
      EEx.eval_string(@base_prompt_input,
        assigns: %{
          task: task_description,
          previous_content: previous_content_string,
          current_chapter_summary: current_chapter_summary,
          meta_data: meta_data_string,
          chapter_num: chapter_num,
          chapter_name: chapter_name
        }
      )

    output_string =
      EEx.eval_string(@base_prompt_output,
        assigns: %{
          chapter_num: chapter_num,
          chapter_name: chapter_name,
          chapter_content: cleaned_content
        }
      )

    instruction_string =
      EEx.eval_string(@dataset_template,
        assigns: %{
          input: input_string,
          output: output_string
        }
      )
      |> format_instruction_string()

    {instruction_string, category}
  end

  defp cleanup_chapter_content(content) when is_binary(content) do
    if contains_restricted_tags?(content) do
      ""
    else
      content
      |> remove_title_and_leading_newlines()
      |> remove_footnote_tags()
      |> remove_image_lines()
      |> remove_special_tags()
      |> normalize_newlines()
    end
  end

  defp cleanup_chapter_content(nil), do: ""

  defp remove_footnote_tags(content) when is_binary(content) do
    # This pattern will match both standalone footnotes and embedded footnotes
    Regex.replace(~r/\[\^(\d+)\]/, content, "")
  end

  defp remove_footnote_tags(nil), do: ""

  defp remove_image_lines(content) when is_binary(content) do
    content
    |> String.split("\n")
    |> Enum.reject(fn line -> String.starts_with?(String.trim(line), "![") end)
    |> Enum.join("\n")
  end

  defp remove_image_lines(nil), do: ""

  defp remove_special_tags(content) when is_binary(content) do
    tag_regex =
      ~r/(?i)###\s*(FALL ENDE|FALL BEGINN|ZUSAMMENFASSUNG|TEXT|FALL ZUSAMMENFASSUNG|FALL|KERNAUSSAGEN?|STICHWORTE?|SCHLUSSFOLGERUNGE?N?|EMPFEHLUNGE?N?)\s*\n*/

    Regex.replace(tag_regex, content, "")
  end

  defp remove_special_tags(nil), do: ""

  defp remove_chapter_header(content, chapter_num, chapter_name) when is_binary(content) do
    chapter_with_dot =
      if String.contains?(to_string(chapter_num), ".") do
        "#{chapter_num} #{chapter_name}"
      else
        "#{chapter_num}. #{chapter_name}"
      end

    case_insensitive_regex = Regex.compile!("(?i)" <> Regex.escape(chapter_with_dot))
    Regex.replace(case_insensitive_regex, content, "")
  end

  defp remove_chapter_header(nil, _chapter_num, _chapter_name), do: ""

  defp contains_restricted_tags?(content) when is_binary(content) do
    String.contains?(content, "### BEISPIEL") ||
      String.contains?(content, "### ANWEISUNG")
  end

  defp contains_restricted_tags?(nil), do: false

  defp remove_title_and_leading_newlines(content) when is_binary(content) do
    case String.split(content, "\n", parts: 2) do
      [_title, rest] ->
        String.replace_prefix(rest, "\n", "") |> String.trim_leading()

      [only_line] ->
        ""

      [] ->
        ""
    end
  end

  defp remove_title_and_leading_newlines(nil), do: ""

  defp normalize_newlines(content) when is_binary(content) do
    content
    |> String.replace(~r/\n\s*\n/, "\n")
    |> String.replace(~r/\r\n/, "\n")
    |> String.replace(~r/\r/, "\n")
    |> String.replace(~r/\n\s+\n/, "\n\n")
    |> String.replace(~r/^\s+/, "")
    |> String.replace(~r/\s+$/, "")
  end

  defp normalize_newlines(nil), do: ""

  defp format_previous_content(previous_content, category) do
    case category do
      "only_chapters" ->
        if map_size(previous_content) > 0 do
          filtered_content =
            previous_content
            |> Enum.map(fn {idx, chapter} ->
              chapter_type =
                case chapter.type do
                  "main_chapter_no_closing" -> "main_chapter_no_closing"
                  "main_chapter_no_closing_no_content" -> "main_chapter_no_closing_no_content"
                  "main_chapter_no_closing" -> "main_chapter_no_closing"
                  other -> other
                end

              updated_content =
                chapter.content
                |> remove_special_tags()
                |> remove_chapter_header(chapter.chapter_num, chapter.sanitized_filename)

              {idx, chapter |> Map.put(:type, chapter_type) |> Map.put(:content, updated_content)}
            end)
            |> Map.new()

          formatted_content = process_previous_chapters(filtered_content, true, category)

          EEx.eval_string(@prompt_previous_content_chapters,
            assigns: %{previous_content: formatted_content}
          )
        else
          ""
        end

      "only_summaries" ->
        if map_size(previous_content) > 0 do
          modified_content = override_summary_types(previous_content)
          formatted_summaries = process_previous_summaries(modified_content)

          EEx.eval_string(@prompt_previous_content_summaries,
            assigns: %{previous_content: formatted_summaries}
          )
        else
          ""
        end

      _ ->
        ""
    end
  end

  defp format_previous_content_for_technical(previous_content) do
    if map_size(previous_content) > 0 do
      filtered_content =
        previous_content
        |> Enum.map(fn {idx, chapter} ->
          chapter_type =
            case chapter.type do
              "main_chapter_no_closing" -> "main_chapter_no_closing"
              "main_chapter_no_closing_no_content" -> "main_chapter_no_closing_no_content"
              "main_chapter_no_closing" -> "main_chapter_no_closing"
              other -> other
            end

          updated_content =
            chapter.content
            |> remove_footnote_tags()
            |> remove_image_lines()
            |> remove_special_tags()
            |> remove_chapter_header(chapter.chapter_num, chapter.sanitized_filename)

          {idx, chapter |> Map.put(:type, chapter_type) |> Map.put(:content, updated_content)}
        end)
        |> Map.new()

      process_previous_chapters(filtered_content, true, "only_chapters")
    else
      ""
    end
  end

  defp override_summary_types(previous_content) do
    previous_content
    |> Enum.map(fn {idx, item} ->
      {idx, Map.put(item, :type, "sub_chapter")}
    end)
    |> Map.new()
  end

  defp process_previous_chapters(previous_content, cleanup \\ true, category \\ nil) do
    sorted_chapters =
      previous_content
      |> Enum.sort_by(fn {idx, _} -> idx end)

    {result, _state} =
      process_chapters_recursive(
        sorted_chapters,
        0,
        %{open_chapters: [], processed_chapters: []},
        cleanup,
        category
      )

    result
  end

  defp process_chapters_recursive([], _current_idx, _state, _cleanup, _category),
    do: {"", %{open_chapters: [], processed_chapters: []}}

  defp process_chapters_recursive([{idx, chapter} | rest], current_idx, state, cleanup, category) do
    chapter_type = chapter.type
    chapter_num = chapter.chapter_num
    chapter_name = chapter.sanitized_filename

    if chapter_num in state.processed_chapters do
      process_chapters_recursive(rest, current_idx + 1, state, cleanup, category)
    else
      chapter_content =
        if cleanup do
          cleanup_chapter_content(chapter.content)
        else
          normalize_newlines(chapter.content)
        end

      next_item = if length(rest) > 0, do: hd(rest), else: nil
      has_following = next_item != nil

      {formatted_chapter, new_state} =
        format_chapter_by_type(
          chapter_type,
          chapter_num,
          chapter_name,
          chapter_content,
          has_following,
          rest,
          state
        )

      updated_state = %{
        new_state
        | processed_chapters: [chapter_num | new_state.processed_chapters]
      }

      {following_content, final_state} =
        process_chapters_recursive(rest, idx + 1, updated_state, cleanup, category)

      {formatted_chapter <> following_content, final_state}
    end
  end

  defp extract_sub_chapters(chapters, main_chapter_num, state) do
    direct_children =
      chapters
      |> Enum.filter(fn {_idx, chapter} ->
        is_direct_child(chapter.chapter_num, main_chapter_num) &&
          !(chapter.chapter_num in state.processed_chapters)
      end)
      |> Enum.sort_by(fn {_idx, chapter} -> chapter.chapter_num end)

    {sub_content, remaining_chapters, new_state} =
      Enum.reduce(direct_children, {"", [], state}, fn {idx, chapter},
                                                       {content_acc, remaining_acc, state_acc} ->
        updated_state = %{
          state_acc
          | processed_chapters: [chapter.chapter_num | state_acc.processed_chapters]
        }

        {sub_content, newer_state} =
          process_chapter_with_children(chapter, idx, chapters, updated_state)

        {content_acc <> sub_content, remaining_acc, newer_state}
      end)

    remaining =
      chapters
      |> Enum.reject(fn {_idx, chapter} ->
        is_descendant(chapter.chapter_num, main_chapter_num) ||
          chapter.chapter_num in new_state.processed_chapters
      end)

    {sub_content, remaining, new_state}
  end

  defp format_chapter_by_type(
         chapter_type,
         chapter_num,
         chapter_name,
         chapter_content,
         has_following,
         rest,
         state
       ) do
    cleaned_content = cleanup_chapter_content(chapter_content)

    case chapter_type do
      "main_chapter_no_content" ->
        {sub_chapters, remaining_chapters, new_state} =
          extract_sub_chapters(rest, chapter_num, state)

        following_content = if has_following, do: "", else: "\n\n"

        template = @prompt_previous_main_chapter_no_content

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                previous_sub_chapters: sub_chapters || "",
                previous_chapter_following: following_content
              }
            )

          updated_state = %{new_state | open_chapters: [chapter_num | new_state.open_chapters]}
          {formatted, updated_state}
        else
          {"", new_state}
        end

      "main_chapter_no_closing" ->
        {sub_chapters, remaining_chapters, new_state} =
          extract_sub_chapters(rest, chapter_num, state)

        following_content = if has_following, do: "", else: "\n\n"

        template = @prompt_previous_main_chapter_no_closing

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                chapter_content: cleaned_content,
                previous_sub_chapters: sub_chapters || "",
                previous_chapter_following: following_content
              }
            )

          updated_state = %{new_state | open_chapters: [chapter_num | new_state.open_chapters]}
          {formatted, updated_state}
        else
          {"", new_state}
        end

      "main_chapter_opening" <> _ ->
        {sub_chapters, remaining_chapters, new_state} =
          extract_sub_chapters(rest, chapter_num, state)

        following_content = if has_following, do: "", else: "\n\n"

        template = @prompt_previous_main_chapter_with_content

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                chapter_content: cleaned_content,
                previous_sub_chapters: sub_chapters || "",
                previous_chapter_following: following_content
              }
            )

          updated_state = %{new_state | open_chapters: [chapter_num | new_state.open_chapters]}
          {formatted, updated_state}
        else
          {"", new_state}
        end

      "main_chapter" ->
        following_content = if has_following, do: "", else: "\n\n"

        template = @prompt_previous_main_chapter_with_content

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                chapter_content: cleaned_content,
                previous_sub_chapters: "",
                previous_chapter_following: following_content
              }
            )

          {formatted, state}
        else
          {"", state}
        end

      "main_chapter_no_closing" ->
        {sub_chapters, remaining_chapters, new_state} =
          extract_sub_chapters(rest, chapter_num, state)

        template = @prompt_previous_main_chapter_no_closing

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                chapter_content: cleaned_content,
                previous_sub_chapters: sub_chapters || ""
              }
            )

          updated_state = %{new_state | open_chapters: [chapter_num | new_state.open_chapters]}
          {formatted, updated_state}
        else
          {"", new_state}
        end

      "main_chapter_no_closing_no_content" ->
        {sub_chapters, remaining_chapters, new_state} =
          extract_sub_chapters(rest, chapter_num, state)

        template = @prompt_previous_main_chapter_no_closing_no_content

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                previous_sub_chapters: sub_chapters || ""
              }
            )

          updated_state = %{new_state | open_chapters: [chapter_num | new_state.open_chapters]}
          {formatted, updated_state}
        else
          {"", new_state}
        end

      "subchapter_closing" <> _ ->
        template = @prompt_previous_sub_chapter_closing

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                chapter_content: cleaned_content
              }
            )

          updated_state =
            if length(state.open_chapters) > 0 do
              %{state | open_chapters: tl(state.open_chapters)}
            else
              state
            end

          {formatted, updated_state}
        else
          {"", state}
        end

      "sub_chapter" ->
        following_content = if has_following, do: "", else: "\n\n"

        template = @prompt_previous_sub_chapter

        if is_binary(template) do
          formatted =
            EEx.eval_string(
              template,
              assigns: %{
                chapter_num: chapter_num,
                chapter_name: chapter_name,
                chapter_content: cleaned_content,
                previous_chapter_following: following_content
              }
            )

          {formatted, state}
        else
          {"", state}
        end

      _ ->
        {"", state}
    end
  end

  defp process_chapter_with_children(chapter, idx, all_chapters, state) do
    chapter_type = chapter.type
    chapter_num = chapter.chapter_num
    chapter_name = chapter.sanitized_filename
    chapter_content = cleanup_chapter_content(chapter.content)

    children_chapters =
      all_chapters
      |> Enum.filter(fn {_idx, ch} ->
        is_direct_child(ch.chapter_num, chapter_num) &&
          !(ch.chapter_num in state.processed_chapters)
      end)
      |> Enum.sort_by(fn {_idx, ch} -> ch.chapter_num end)

    case chapter_type do
      "main_chapter_opening" <> _ ->
        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, %{
            state
            | processed_chapters: [chapter_num | state.processed_chapters]
          })

        formatted =
          EEx.eval_string(
            @prompt_previous_main_chapter_with_content,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              chapter_content: chapter_content,
              previous_sub_chapters: sub_content,
              previous_chapter_following: ""
            }
          )

        updated_state = %{sub_state | open_chapters: [chapter_num | sub_state.open_chapters]}
        {formatted, updated_state}

      "main_chapter_no_content" ->
        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, %{
            state
            | processed_chapters: [chapter_num | state.processed_chapters]
          })

        formatted =
          EEx.eval_string(
            @prompt_previous_main_chapter_no_content,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              previous_sub_chapters: sub_content,
              previous_chapter_following: ""
            }
          )

        updated_state = %{sub_state | open_chapters: [chapter_num | sub_state.open_chapters]}
        {formatted, updated_state}

      "main_chapter_no_closing" ->
        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, %{
            state
            | processed_chapters: [chapter_num | state.processed_chapters]
          })

        formatted =
          EEx.eval_string(
            @prompt_previous_main_chapter_no_closing,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              chapter_content: chapter_content,
              previous_sub_chapters: sub_content,
              previous_chapter_following: ""
            }
          )

        updated_state = %{sub_state | open_chapters: [chapter_num | sub_state.open_chapters]}
        {formatted, updated_state}

      "main_chapter_no_closing" ->
        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, %{
            state
            | processed_chapters: [chapter_num | state.processed_chapters]
          })

        formatted =
          EEx.eval_string(
            @prompt_previous_main_chapter_no_closing,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              chapter_content: chapter_content,
              previous_sub_chapters: sub_content
            }
          )

        updated_state = %{sub_state | open_chapters: [chapter_num | sub_state.open_chapters]}
        {formatted, updated_state}

      "main_chapter_no_closing_no_content" ->
        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, %{
            state
            | processed_chapters: [chapter_num | state.processed_chapters]
          })

        formatted =
          EEx.eval_string(
            @prompt_previous_main_chapter_no_closing_no_content,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              previous_sub_chapters: sub_content
            }
          )

        updated_state = %{sub_state | open_chapters: [chapter_num | sub_state.open_chapters]}
        {formatted, updated_state}

      "sub_chapter" ->
        formatted =
          EEx.eval_string(
            @prompt_previous_sub_chapter,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              chapter_content: chapter_content,
              previous_chapter_following: ""
            }
          )

        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, %{
            state
            | processed_chapters: [chapter_num | state.processed_chapters]
          })

        {formatted <> sub_content, sub_state}

      "subchapter_closing" <> _ ->
        formatted =
          EEx.eval_string(
            @prompt_previous_sub_chapter_closing,
            assigns: %{
              chapter_num: chapter_num,
              chapter_name: chapter_name,
              chapter_content: chapter_content
            }
          )

        updated_state =
          if length(state.open_chapters) > 0 do
            %{
              state
              | open_chapters: tl(state.open_chapters),
                processed_chapters: [chapter_num | state.processed_chapters]
            }
          else
            %{state | processed_chapters: [chapter_num | state.processed_chapters]}
          end

        {formatted, updated_state}

      _ ->
        {formatted, state} =
          {"", %{state | processed_chapters: [chapter_num | state.processed_chapters]}}

        {sub_content, sub_state} =
          process_children_recursive(children_chapters, all_chapters, state)

        {formatted <> sub_content, sub_state}
    end
  end

  defp process_children_recursive([], _all_chapters, state), do: {"", state}

  defp process_children_recursive(children_chapters, all_chapters, state) do
    Enum.reduce(children_chapters, {"", state}, fn {idx, chapter}, {acc_content, acc_state} ->
      if chapter.chapter_num in acc_state.processed_chapters do
        {acc_content, acc_state}
      else
        {sub_content, new_state} =
          process_chapter_with_children(chapter, idx, all_chapters, acc_state)

        {acc_content <> sub_content, new_state}
      end
    end)
  end

  defp format_previous_content_for_technical(previous_content) do
    if map_size(previous_content) > 0 do
      filtered_content =
        previous_content
        |> Enum.map(fn {idx, chapter} ->
          chapter_type =
            case chapter.type do
              "main_chapter_no_closing" -> "main_chapter_no_closing"
              "main_chapter_no_closing_no_content" -> "main_chapter_no_closing_no_content"
              "main_chapter_no_closing" -> "main_chapter_no_closing"
              other -> other
            end

          {idx, Map.put(chapter, :type, chapter_type)}
        end)
        |> Map.new()

      process_previous_chapters(filtered_content, true, "only_chapters")
    else
      ""
    end
  end

  defp format_meta_data(meta_data) do
    cond do
      is_nil(meta_data) ->
        ""

      map_size(meta_data) > 0 ->
        meta_data_text =
          meta_data
          |> Enum.map(fn {key, value} -> "#{key}: #{value}" end)
          |> Enum.join("\n")

        EEx.eval_string(@prompt_meta_data,
          assigns: %{meta_data_objects: meta_data_text}
        )

      true ->
        ""
    end
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

  defp process_previous_summaries(previous_content) do
    sorted_summaries =
      previous_content
      |> Enum.sort_by(fn {idx, _} -> idx end)

    {result, _} =
      Enum.reduce(sorted_summaries, {"", nil}, fn {idx, summary}, {acc, _} ->
        next_item = Enum.find(sorted_summaries, fn {next_idx, _} -> next_idx > idx end)
        has_following = next_item != nil

        formatted_summary =
          format_summary_by_type(
            summary.type,
            summary.chapter_num,
            summary.sanitized_filename,
            summary.content,
            has_following
          )

        {acc <> formatted_summary, summary}
      end)

    result
  end

  defp format_summary_by_type(type, chapter_num, chapter_name, content, has_following) do
    normalized_content =
      if contains_restricted_tags?(content) do
        ""
      else
        content
        |> remove_footnote_tags()
        |> remove_image_lines()
        |> remove_special_tags()
        |> remove_chapter_header(chapter_num, chapter_name)
        |> normalize_newlines()
      end

    case type do
      "subchapter_closing" <> _ ->
        EEx.eval_string(
          @prompt_previous_sub_summary_closing,
          assigns: %{
            chapter_num: chapter_num,
            chapter_name: chapter_name,
            previous_summary_content: normalized_content
          }
        )

      _ ->
        following_content = if has_following, do: "", else: "\n\n"

        EEx.eval_string(
          @prompt_previous_main_summary,
          assigns: %{
            chapter_num: chapter_num,
            chapter_name: chapter_name,
            previous_summary_content: normalized_content,
            previous_summary_following: following_content
          }
        )
    end
  end

  defp is_direct_child(chapter_num, parent_num) do
    chapter_str = to_string(chapter_num)
    parent_str = to_string(parent_num)

    chapter_parts = String.split(chapter_str, ".")
    parent_parts = String.split(parent_str, ".")

    String.starts_with?(chapter_str, "#{parent_str}.") &&
      length(chapter_parts) == length(parent_parts) + 1
  end

  defp is_descendant(chapter_num, ancestor_num) do
    chapter_str = to_string(chapter_num)
    ancestor_str = to_string(ancestor_num)

    String.starts_with?(chapter_str, "#{ancestor_str}.")
  end

  defp get_parent_chapter_num(chapter_num) do
    chapter_str = to_string(chapter_num)
    parts = String.split(chapter_str, ".")

    if length(parts) > 1 do
      parts
      |> Enum.take(length(parts) - 1)
      |> Enum.join(".")
    else
      nil
    end
  end

  defp get_direct_children(chapters, parent_num) do
    chapters
    |> Enum.filter(fn {_idx, chapter} ->
      is_direct_child(chapter.chapter_num, parent_num)
    end)
    |> Enum.sort_by(fn {_idx, chapter} -> chapter.chapter_num end)
  end

  defp build_chapter_hierarchy(chapters, root_chapter_num) do
    chapters
    |> Enum.reduce(%{}, fn {idx, chapter}, acc ->
      parent_num = get_parent_chapter_num(chapter.chapter_num)

      children = Map.get(acc, parent_num, [])
      Map.put(acc, parent_num, children ++ [{idx, chapter}])
    end)
  end

  defp get_subchapters(chapters, parent_num) do
    chapters
    |> Enum.filter(fn {_idx, chapter} ->
      is_descendant(chapter.chapter_num, parent_num) &&
        !is_direct_child(chapter.chapter_num, parent_num)
    end)
  end
end
