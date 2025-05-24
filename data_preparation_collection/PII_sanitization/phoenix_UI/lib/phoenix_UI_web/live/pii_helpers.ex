defmodule PhoenixUIWeb.PIIHelpers do
  def normalize_input_text(text) when is_binary(text) do
    text
    |> String.replace("\\n", "\n")
    |> String.replace(~r/\r\n/, "\n")
    |> String.replace(~r/\n\n+/, "\n\n")
    |> String.trim()
    |> :unicode.characters_to_binary(:utf8)
  end

  def format_analysis({:error, {:python, :"builtins.ValueError", message, _trace}}) do
    %{
      error: true,
      message: to_string(message)
    }
  end

  def format_analysis(results) when is_list(results) do
    results
    |> Enum.map(fn {text, start, end_pos, type, score} ->
      %{
        text: to_string(text) |> :unicode.characters_to_binary(:utf8),
        start: start,
        end: end_pos,
        type: to_string(type),
        score: score
      }
    end)
    |> merge_overlapping_entities()
    |> Enum.sort_by(& &1.start)
  end

  defp merge_overlapping_entities(entities) do
    # Group entities that overlap or are substrings
    entities
    |> Enum.reduce([], fn entity, acc ->
      # Find any existing group that overlaps with current entity
      overlapping_group =
        Enum.find(acc, fn group ->
          Enum.any?(group, fn existing ->
            ranges_overlap?(existing, entity) || is_substring?(existing, entity)
          end)
        end)

      case overlapping_group do
        nil -> [MapSet.new([entity]) | acc]
        group -> [MapSet.put(group, entity) | acc -- [group]]
      end
    end)
    |> Enum.map(fn group ->
      entities_list = MapSet.to_list(group)
      # Sort by score and get primary entity
      sorted_entities = Enum.sort_by(entities_list, & &1.score, :desc)
      primary = List.first(sorted_entities)

      %{
        start: primary.start,
        end: primary.end,
        entities: sorted_entities,
        # Use text from highest scoring entity
        text: primary.text
      }
    end)
  end

  defp ranges_overlap?(e1, e2) do
    not (e1.end <= e2.start || e2.end <= e1.start)
  end

  defp is_substring?(e1, e2) do
    (e1.start >= e2.start && e1.end <= e2.end) ||
      (e2.start >= e1.start && e2.end <= e1.end)
  end

  def get_text_segments(text, highlights, _socket) when is_binary(text) do
    if Enum.empty?(highlights) do
      [{:normal, text, nil}]
    else
      utf8_offsets = create_utf8_offset_map(text)

      {segments, last_pos} =
        highlights
        |> Enum.sort_by(& &1.start)
        |> Enum.reduce({[], 0}, fn highlight, {acc, pos} ->
          # Get byte positions for both start and end
          start_pos = Map.get(utf8_offsets, highlight.start, highlight.start)
          end_pos = Map.get(utf8_offsets, highlight.end - 0.5, highlight.end)

          before_text =
            if start_pos > pos,
              do: binary_part(text, pos, start_pos - pos),
              else: ""

          # Use original text from input for the highlight
          highlight_text = binary_part(text, start_pos, end_pos - start_pos)

          new_segments =
            acc ++
              [
                if(before_text != "", do: {:normal, before_text, nil}, else: nil),
                {:highlight, highlight_text, highlight}
              ]

          {Enum.reject(new_segments, &is_nil/1), end_pos}
        end)

      remaining = binary_part(text, last_pos, byte_size(text) - last_pos)
      final_segments = segments ++ if(remaining != "", do: [{:normal, remaining, nil}], else: [])
      Enum.reject(final_segments, fn {_, content, _} -> content == "" end)
    end
  end

  def get_text_segments(text, highlights, socket) when is_binary(text) do
    case socket.assigns do
      %{display_text: display_text} when text == display_text ->
        [{:normal, text, nil}]

      _ ->
        [{:normal, "", nil}]
    end
  end

  def get_text_segments(_text, _highlights, _socket), do: [{:normal, "", nil}]

  def protect_text(text, active_labels, language, get_analysis_data) do
    case Phoenix_UI.State.PIIState.get_mode() do
      :anonymize ->
        case MainPii.anonymize_text_erlport(text, active_labels, language, get_analysis_data) do
          {:error, reason} -> {:error, reason}
          result -> {:ok, result}
        end

      :pseudonymize ->
        case MainPii.pseudonymize_text_erlport(text, active_labels, language, get_analysis_data) do
          {:error, reason} -> {:error, reason}
          result -> {:ok, result}
        end
    end
  end

  def protect_text(text, active_labels, language) do
    protect_text(text, active_labels, language, false)
  end

  defp create_char_map(text) do
    text
    |> String.graphemes()
    |> Enum.with_index()
    |> Enum.reduce(%{}, fn {char, idx}, acc ->
      byte_size = byte_size(String.slice(text, 0, idx))
      Map.put(acc, byte_size, idx)
    end)
  end

  defp encode_protection_result(response) when is_map(response) do
    %{
      "anonymized_text" => :unicode.characters_to_binary(response["anonymized_text"], :utf8),
      "single_results" =>
        Enum.map(response["single_results"], fn {text, start, end_pos, type} ->
          {
            :unicode.characters_to_binary(text, :utf8),
            start,
            end_pos,
            :unicode.characters_to_binary(type, :utf8)
          }
        end)
    }
  end

  def process_protection_response(response) when is_map(response) do
    case response do
      %{~c"anonymized_text" => text, ~c"single_results" => results} ->
        %{
          anonymized_text: to_string(text),
          single_results:
            Enum.map(results, fn {text, start, end_pos, type} ->
              {to_string(text), start, end_pos, to_string(type)}
            end)
        }

      %{"anonymized_text" => text, "single_results" => results} ->
        %{
          anonymized_text: to_string(text),
          single_results:
            Enum.map(results, fn {text, start, end_pos, type} ->
              {to_string(text), start, end_pos, to_string(type)}
            end)
        }

      _ ->
        %{
          anonymized_text: "Error processing text",
          single_results: []
        }
    end
  end

  def process_protection_response(response) when is_list(response) do
    # Handle case where response is a charlist
    %{
      anonymized_text: to_string(response),
      single_results: []
    }
  end

  def process_protection_response_with_analysis(response) when is_map(response) do
    # IO.puts("Processing response: #{inspect(response, pretty: true)}")

    # Always return results in the expected format
    %{
      # We'll generate this from replacements
      anonymized_text: "",
      single_results: response["single_results"] || []
    }
  end

  def process_protection_response_with_analysis({:single_results, results})
      when is_list(results) do
    # IO.puts("Processing tuple response: #{inspect(results, pretty: true)}")
    %{anonymized_text: "", single_results: results}
  end

  def process_protection_response_with_analysis(_) do
    %{anonymized_text: "", single_results: []}
  end

  def format_protected_text_with_analysis(%{
        anonymized_text: text,
        single_results: {:single_results, results}
      })
      when is_list(results) do
    do_format_protected_text_with_analysis(text, results)
  end

  def format_protected_text_with_analysis(%{anonymized_text: text, single_results: results})
      when is_list(results) do
    do_format_protected_text_with_analysis(text, results)
  end

  defp do_format_protected_text_with_analysis(text, results) do
    utf8_offsets = create_utf8_offset_map(text)

    sorted_results =
      results
      |> Enum.map(fn result ->
        case result do
          %{} = map ->
            %{
              id: "segment-#{System.unique_integer()}",
              original_text: map["original_text"] || "",
              protected_text: map["protected_text"] || "",
              start: map["start"] || 0,
              end: map["end"] || 0,
              recognizer_name: map["recognizer_name"] || "",
              score: map["score"] || 0.0,
              pattern: map["pattern"] || "",
              validation_result: map["validation_result"] || ""
            }

          _ ->
            %{
              id: "segment-#{System.unique_integer()}",
              original_text: "",
              protected_text: "",
              start: 0,
              end: 0,
              recognizer_name: "",
              score: 0.0,
              pattern: "",
              validation_result: ""
            }
        end
      end)
      |> Enum.sort_by(& &1.start)

    # Process segments and format output
    {segments, _} =
      Enum.reduce(sorted_results, {[], 0}, fn segment, {acc, last_pos} ->
        before_text = binary_part(text, last_pos, segment.start - last_pos)

        new_segments =
          [
            if(before_text != "", do: {:normal, before_text}, else: nil),
            {:protected, segment}
          ]
          |> Enum.reject(&is_nil/1)

        {acc ++ new_segments, segment.end}
      end)

    # Handle remaining text
    last_pos = if length(sorted_results) > 0, do: List.last(sorted_results).end, else: 0
    remaining = binary_part(text, last_pos, byte_size(text) - last_pos)
    final_segments = if(remaining != "", do: segments ++ [{:normal, remaining}], else: segments)

    Enum.reject(final_segments, fn {_, content} -> content == "" || is_nil(content) end)
  end

  def format_protected_text(%{anonymized_text: text, single_results: results})
      when is_list(results) do
    utf8_offsets = create_utf8_offset_map(text)

    sorted_results =
      results
      |> Enum.map(fn {replacement, start, end_pos, type} ->
        # Adjust positions for UTF-8 characters
        adjusted_start = Map.get(utf8_offsets, start, start)
        adjusted_end = Map.get(utf8_offsets, end_pos, end_pos)
        {replacement, adjusted_start, adjusted_end, type}
      end)
      |> Enum.sort_by(fn {_, start, _, _} -> start end)

    {segments, _} =
      {segments, _} =
      Enum.reduce(sorted_results, {[], 0}, fn {_text, start, end_pos, _type}, {acc, last_pos} ->
        before_text = binary_part(text, last_pos, start - last_pos)
        protected_text = binary_part(text, start, end_pos - start)

        new_segments =
          [
            if(before_text != "", do: {:normal, before_text}, else: nil),
            {:protected, protected_text}
          ]
          |> Enum.reject(&is_nil/1)

        {acc ++ new_segments, end_pos}
      end)

    last_result = List.last(sorted_results)
    last_pos = if last_result, do: elem(last_result, 2), else: 0

    remaining = binary_part(text, last_pos, byte_size(text) - last_pos)
    final_segments = if(remaining != "", do: segments ++ [{:normal, remaining}], else: segments)

    Enum.reject(final_segments, fn {_, content} -> content == "" end)
  end

  defp create_utf8_offset_map(text) do
    # Create a mapping of character positions to byte positions
    {pos_map, _total_bytes} =
      text
      |> String.graphemes()
      |> Enum.with_index()
      |> Enum.reduce({%{}, 0}, fn {char, idx}, {map, byte_pos} ->
        char_bytes = byte_size(char)

        new_map =
          map
          # Char pos -> Start byte pos
          |> Map.put(idx, byte_pos)
          # Char pos + 0.5 -> End byte pos
          |> Map.put(idx + 0.5, byte_pos + char_bytes)

        {new_map, byte_pos + char_bytes}
      end)

    pos_map
  end

  defp create_char_length_map(text) do
    text
    |> String.graphemes()
    |> Enum.reduce({%{}, 0}, fn char, {map, pos} ->
      char_length = byte_size(char)
      {Map.put(map, pos, char_length), pos + char_length}
    end)
    |> elem(0)
  end

  defp get_char_position(byte_pos, char_length_map) do
    char_length_map
    |> Map.keys()
    |> Enum.sort()
    |> Enum.find_index(fn pos -> pos >= byte_pos end)
    |> case do
      nil -> byte_pos
      idx -> idx
    end
  end

  # Helper to find the actual position of a replacement in text
  defp find_actual_position(text, replacement, approximate_pos) do
    # Look for the actual word position around the approximate position
    # Characters to look before and after
    search_range = 10
    start_pos = max(0, approximate_pos - search_range)
    end_pos = min(String.length(text), approximate_pos + search_range)

    search_text = String.slice(text, start_pos, end_pos - start_pos)

    case :binary.match(search_text, String.trim(replacement)) do
      {offset, _} -> start_pos + offset
      :nomatch -> approximate_pos
    end
  end

  def get_all_recognizers() do
    case MainPii.get_all_recognizers_erlport() do
      {:error, reason} ->
        {:error, reason}

      result when is_list(result) ->
        # Convert charlists to strings in the map
        formatted_results =
          Enum.map(result, fn recognizer ->
            %{
              recognizer_name: to_string(recognizer["recognizer_name"]),
              supported_entity: to_string(recognizer["supported_entity"])
            }
          end)

        {:ok, formatted_results}
    end
  end
end
