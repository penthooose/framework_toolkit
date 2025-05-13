defmodule PatternRecognizer do
  def run_demo do
    examples = ["ID284$NJ2", "ID300$SOP", "ID299$L8S", "ID400%"]

    RegexGenerator.derive_regex(examples)
    |> IO.inspect()
  end
end

defmodule RegexGenerator do
  def derive_regex(examples) do
    examples
    |> RegexGenerator.generate_preregex()
    |> RegexGenerator.alignRegex()
    |> RegexGenerator.remove_zero_counts()
    |> RegexGenerator.convert_optionals()
    |> RegexGenerator.remove_labels()
    |> RegexGenerator.add_regex_word_labels()
    |> Enum.join("")

    # |> IO.inspect()
  end

  def generate_presentable_regex(examples) do
    examples
    |> RegexGenerator.generate_preregex()
    |> RegexGenerator.alignRegex()
    |> RegexGenerator.remove_zero_counts()
    |> RegexGenerator.convert_optionals()
    |> RegexGenerator.remove_labels()
    # |> RegexGenerator.simplify_presentation()
    |> IO.inspect()
  end

  def validate_regex(pattern) when is_binary(pattern) do
    case Regex.compile(pattern) do
      {:ok, _regex} -> {:ok, "Valid regex pattern"}
      {:error, error} -> {:error, "Invalid regex pattern"}
    end
  end

  def formate_regex(pattern) when is_binary(pattern) do
    pattern
    |> RegexGenerator.remove_unwanted_regex_parts()
    |> RegexGenerator.ensure_word_regex_parts()
  end

  def check_regex_fitting(regex, examples) do
    IO.inspect(regex, label: "fitting regex")
    IO.inspect(examples, label: "transferred examples")

    case Regex.compile(regex) do
      {:ok, compiled_regex} ->
        examples
        |> Enum.all?(fn example ->
          Regex.match?(compiled_regex, example)
        end)

      {:error, _} ->
        false
    end
  end

  def simplify_presentation(regex) do
    regex
    |> Enum.map(fn char -> "<" <> char <> ">" end)
  end

  def generate_preregex(examples) do
    # Determine the length of the longest example
    max_length = Enum.map(examples, &String.length/1) |> Enum.max()

    # Pad shorter examples with a placeholder
    padded_examples =
      examples
      |> Enum.map(&String.pad_trailing(&1, max_length, "?"))

    # Transpose into columns
    char_columns = padded_examples |> Enum.map(&String.graphemes/1) |> transpose()

    # Analyze each column to generate regex rules
    {pattern, optional_started} =
      char_columns
      |> Enum.map_reduce(false, fn column, optional_started ->
        analyze_column(column, optional_started)
      end)

    # Close any open optional group with {:end}
    final_pattern =
      if optional_started do
        pattern ++ ["{:end}"]
      else
        pattern
      end

    # Return the list of individual patterns (not joined into a single string)
    final_pattern
    |> List.flatten()
  end

  defp transpose(list) do
    list
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  defp analyze_column(column, optional_started) do
    unique_chars = Enum.uniq(column)

    cond do
      Enum.all?(unique_chars, &(&1 == "?")) ->
        # Entire column is optional, do nothing but return the state
        {"", optional_started}

      "?" in unique_chars ->
        # Mixed column, start or continue an optional group
        rule = infer_rule(Enum.reject(unique_chars, &(&1 == "?")))

        # IO.inspect(rule, label: "RULE")

        if optional_started do
          {rule, true}
        else
          # Add "{:start}" as a string and insert it as a separate element in the list
          {["{:start}"] ++ [rule], true}
        end

      true ->
        # Fixed column, close any open optional group
        rule = infer_rule(unique_chars)

        # IO.inspect(rule, label: "RULE")

        if optional_started do
          {rule <> "{:end}", false}
        else
          {rule, false}
        end
    end
  end

  defp infer_rule(characters) do
    cond do
      # Fixed character
      length(characters) == 1 ->
        escape_special(hd(characters)) <> "{0}"

      # Digits
      Enum.all?(characters, &(&1 =~ ~r/\d/)) ->
        "\\d{0}"

      # # Special symbols that need to be escaped
      # Enum.all?(characters, &(&1 =~ ~r/[\$\^\*\(\)\-\+\[\]\{\}\|\/\.\?\\]/)) ->
      #   "[\\$\\^\\*\\(\\)\\-\\+\\[\\]\\{\\}\\|\\/\\?\\\]{0}"

      # # Special symbols that don't need to be escaped
      # Enum.all?(characters, &(&1 =~ ~r/[%&=<>_!@#;:'",`~§]/)) ->
      #   "[%&=<>_!@#;:'\",`~§]{0}"

      # All special characters
      Enum.all?(characters, &(&1 =~ ~r/[\$\^\*\(\)\-\+\[\]\{\}\|\/\.\?\\%&=<>_!@#;:'",`~§]/)) ->
        "[\\$\\^\\*\\(\\)\\-\\+\\[\\]\\{\\}\\|\\/\\.\\?\\\\%&=<>_!@#;:'\",`~§]{0}"

      # Uppercase letters
      Enum.all?(characters, &(&1 =~ ~r/[A-Z]/)) ->
        "[A-Z]{0}"

      # Letters
      Enum.all?(characters, &(&1 =~ ~r/[A-Za-z]/)) ->
        "[A-Za-z]{0}"

      Enum.all?(characters, &(&1 =~ ~r/[A-Z0-9]/)) ->
        "[A-Z0-9]{0}"

      Enum.all?(characters, &(&1 =~ ~r/[a-z0-9]/)) ->
        "[a-z0-9]{0}"

      # Alphanumeric
      Enum.all?(characters, &(&1 =~ ~r/[A-Za-z0-9]/)) ->
        "[A-Za-z0-9]{0}"

      # Any character
      true ->
        # Check if any character is a special character
        escaped_chars = Enum.map(characters, &escape_special(&1))

        # If there is at least one non-special character, return "."
        if Enum.any?(characters, fn char -> char != escape_special(char) end) do
          ".{0}"
        else
          # Check if all characters are special but not the same
          if Enum.uniq(escaped_chars) |> length() > 1 do
            ".{0}"
          else
            # If all characters are the same special character
            Enum.join(escaped_chars)
          end
        end
    end
  end

  defp escape_special(character) do
    # Escaping special regex characters
    case character do
      "\\" -> "\\\\"
      "/" -> "\\/"
      "." -> "\\."
      "^" -> "\\^"
      "$" -> "\\$"
      "|" -> "\\|"
      "?" -> "\\?"
      "*" -> "\\*"
      "+" -> "\\+"
      "(" -> "\\("
      ")" -> "\\)"
      "[" -> "\\["
      "]" -> "\\]"
      "{" -> "\\{"
      "}" -> "\\}"
      # If it's not a special character, return any possible other char as "."
      _ -> character
    end
  end

  def alignRegex(list) do
    list
    # Remove nil entries
    |> Enum.reject(&is_nil/1)
    |> Enum.reduce({[], nil, 0}, fn entry, {acc, prev, count} ->
      case entry do
        ^prev ->
          # Consecutive duplicate found, increase count
          {acc, prev, count + 1}

        _ ->
          # If we have a count > 1 for the previous item, update the count in the regex
          new_acc =
            if count > 1, do: acc ++ [increment_count(prev, count)], else: acc ++ List.wrap(prev)

          # Reset count for new unique item
          {new_acc, entry, 1}
      end
    end)
    # Ensure the last item is processed
    |> finalize_last_entry()
  end

  # Helper function to increment the number inside the curly brackets
  defp increment_count(entry, count) do
    Regex.replace(~r/\{(\d+)\}/, entry, fn _, num_str ->
      new_count = String.to_integer(num_str) + count
      "{#{new_count}}"
    end)
  end

  # Finalize the last element in case it's part of a group
  defp finalize_last_entry({acc, prev, count}) do
    # Process the last item
    if count > 1 do
      acc ++ [increment_count(prev, count)]
    else
      acc ++ List.wrap(prev)
    end
  end

  def remove_zero_counts(list) do
    Enum.map(list, fn entry ->
      # Replace "{0}" at the end of the string with an empty string
      Regex.replace(~r/\{0\}$/, entry, "")
    end)
  end

  def convert_optionals(list) do
    # Helper function to process the list, using recursion
    do_convert(list, false)
  end

  # Base case: empty list
  defp do_convert([], _inside), do: []

  defp do_convert(["{:start}" | tail], _inside) do
    # Start the processing
    ["{:start}" | do_convert(tail, true)]
  end

  defp do_convert(["{:end}" | tail], true) do
    # End the processing
    ["{:end}" | do_convert(tail, false)]
  end

  defp do_convert([head | tail], true) do
    # Add round brackets and "?" to elements inside {:start} and {:end}
    ["(#{head})?" | do_convert(tail, true)]
  end

  defp do_convert([head | tail], false) do
    # Leave elements outside {:start} and {:end} unchanged
    [head | do_convert(tail, false)]
  end

  def remove_labels(list) do
    Enum.reject(list, fn entry -> entry in ["{:start}", "{:end}"] end)
  end

  def add_regex_word_labels(list) do
    # Add "^" at the beginning and "$" at the end of the list
    list
    |> List.insert_at(0, "\\b(")
    |> List.insert_at(-1, ")\\b")
  end

  def add_regex_word_labels(string) do
    "\\b(" <> string <> ")\\b"
  end

  def remove_unwanted_regex_parts(string) do
    # remove ^ and $ from the string
    Regex.replace(~r/^\^/, string, "")
    Regex.replace(~r/\$$/, string, "")
  end

  def ensure_word_regex_parts(string) do
    # ensure that string starts and ends with \b if it doesn't already
    if String.starts_with?(string, "\\b") && String.ends_with?(string, "\\b") do
      string
    else
      "\\b(" <> string <> ")\\b"
    end
  end
end
