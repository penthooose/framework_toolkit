defmodule DataPrepare.DPExtractMdbData do
  @moduledoc """
  This module is responsible for extracting data from a Microsoft Access database (MDB) file.
  It uses the `mdb` library to read the MDB file and extract data from specified tables.
  """

  @mdb_file_path "N:/Thesis/data_prepare/mdb/DB/gutachten_db.mdb"
  @mdb_records_dir "N:/Thesis/data_prepare/mdb/mdb_data"
  # Adjust this path to where you installed MDB Tools
  @wsl_mdb_tools_path "/mnt/n/Thesis/data_prepare/mdb/DB/gutachten_db.mdb"
  @encodings ["cp1252", "iso8859-1", "iso8859-15", "cp850"]

  defp run_wsl_command(command) do
    # Using bash -l to load the full login environment
    System.cmd("wsl", ["bash", "-l", "-c", command])
  end

  def test_wsl do
    case run_wsl_command("which mdb-tables") do
      {path, 0} ->
        IO.puts("MDB Tools found at: #{String.trim(path)}")
        :ok

      {error, code} ->
        IO.puts("MDB Tools not found. Exit code: #{code}, Error: #{error}")
        :error
    end
  end

  # Build full path to an MDB tool executable
  defp mdb_tool_path(tool_name) do
    Path.join(@wsl_mdb_tools_path, tool_name)
  end

  @doc """
  Main function to extract all table data from the MDB file and save it to the output directory.
  """
  def extract_all_data do
    File.mkdir_p!(@mdb_records_dir)

    tables = list_tables()

    Enum.each(tables, fn table ->
      IO.puts("Extracting data from table: #{table}")

      # Skip schema retrieval which is failing
      # columns = get_table_columns(table)
      columns = []

      # Try with export instead
      records = get_table_records_direct(table)

      # Only save if we got records
      if length(records) > 0 do
        save_table_data(table, Map.keys(hd(records)), records)
      else
        IO.puts("No records found for table: #{table}")
      end
    end)

    IO.puts("Data extraction completed. Files saved to #{@mdb_records_dir}")
  end

  @doc """
  Lists all tables in the MDB file.
  """
  def list_tables do
    try do
      {result, 0} = run_wsl_command("mdb-tables -1 \"#{@wsl_mdb_tools_path}\"")

      result
      |> String.trim()
      |> String.split("\n")
      |> Enum.filter(&(String.trim(&1) != ""))
    rescue
      e ->
        IO.puts("Error executing WSL command: #{inspect(e)}")
        []
    end
  end

  # New direct approach, skipping schema retrieval
  def get_table_records_direct(table) do
    try do
      # Get headers first using the correct delimiter syntax
      {header_result, _} =
        run_wsl_command("mdb-export -d \"|\" \"#{@wsl_mdb_tools_path}\" \"#{table}\" | head -n 1")

      headers =
        header_result
        |> String.trim()
        |> String.split("|")

      # Use CSV output with header for full data
      {result, 0} = run_wsl_command("mdb-export -d \"|\" \"#{@wsl_mdb_tools_path}\" \"#{table}\"")

      # Parse data with pipe delimiter
      rows =
        result
        |> String.split("\n", trim: true)
        |> Enum.map(fn line ->
          values =
            line
            |> String.split("|")
            |> Enum.map(&safe_decode_value/1)

          # Create a record with the headers (ensuring we have matching pairs)
          Enum.zip(Enum.take(headers, length(values)), values)
          |> Enum.into(%{})
          |> sanitize_map_values()
        end)
        # Filter out empty maps and records with only binary data
        |> Enum.filter(fn map ->
          map != %{} && !only_binary_data?(map)
        end)

      rows
    rescue
      e ->
        IO.puts("Error getting records: #{inspect(e)}")
        []
    end
  end

  # Sanitize all values in a map to ensure they're safe for JSON
  defp sanitize_map_values(map) do
    Enum.map(map, fn {key, value} ->
      {key, sanitize_for_json(value)}
    end)
    |> Enum.into(%{})
  end

  # Check if a record contains only binary data
  defp only_binary_data?(map) when map_size(map) == 1 do
    # Special case: if there's only one field and it has binary data
    {_key, value} = Enum.at(map, 0)
    is_binary_data?(value)
  end

  defp only_binary_data?(map) when map_size(map) > 0 do
    # Check if all values in the map might be binary data
    map
    |> Map.values()
    |> Enum.all?(&is_binary_data?/1)
  end

  defp only_binary_data?(_), do: false

  # Comprehensive check if a value is likely binary data
  defp is_binary_data?(value) when is_binary(value) do
    cond do
      # Already marked as binary data
      String.starts_with?(value, "[BINARY DATA]") -> true
      # Empty strings aren't binary data
      String.length(value) == 0 -> false
      # Common binary data indicators
      has_null_bytes?(value) -> true
      has_control_chars?(value) -> true
      !String.valid?(value) -> true
      # Check for high concentration of non-printable characters
      binary_char_ratio(value) > 0.3 -> true
      # Otherwise assume it's valid text
      true -> false
    end
  end

  defp is_binary_data?(_), do: false

  # Calculate ratio of likely binary characters to total length
  defp binary_char_ratio(str) when byte_size(str) > 0 do
    binary_chars =
      str
      |> :binary.bin_to_list()
      |> Enum.count(fn byte ->
        byte < 32 or (byte > 126 and byte != 10 and byte != 13 and byte != 9)
      end)

    binary_chars / byte_size(str)
  end

  defp binary_char_ratio(_), do: 0.0

  # Check for control characters which often indicate binary data
  defp has_control_chars?(str) do
    String.codepoints(str)
    |> Enum.any?(fn char ->
      codepoint = String.to_integer(Base.encode16(char), 16)

      (codepoint < 32 and codepoint not in [9, 10, 13]) or
        (codepoint >= 0x80 and codepoint <= 0x9F)
    end)
  rescue
    _ -> true
  end

  # Check if a string contains null bytes (common in binary data)
  defp has_null_bytes?(str) do
    String.contains?(str, <<0>>)
  end

  # Safely convert values, handling potential binary data
  defp safe_decode_value(value) do
    trimmed_value = String.trim(value)

    cond do
      # Empty value
      trimmed_value == "" ->
        ""

      # Check for common binary data patterns
      has_null_bytes?(trimmed_value) ->
        "[BINARY DATA]"

      # Try to clean up the value
      true ->
        cleaned_value = String.replace(trimmed_value, ~r/^"(.*)"$/, "\\1")
        filter_binary_data(cleaned_value)
    end
  end

  # Filter out non-UTF8 data
  defp filter_binary_data(str) do
    cond do
      # Short circuit for empty strings
      String.length(str) == 0 ->
        ""

      # Check for common binary data indicators
      String.contains?(str, <<1::size(8)>>) ->
        "[BINARY DATA]"

      # More comprehensive binary detection
      is_binary_data?(str) ->
        "[BINARY DATA]"

      # Try to verify it's valid UTF-8
      true ->
        case :unicode.characters_to_binary(str, :utf8) do
          {:error, _, _} ->
            "[BINARY DATA]"

          {:incomplete, _, _} ->
            "[INCOMPLETE DATA]"

          binary when is_binary(binary) ->
            if String.valid?(binary), do: binary, else: "[BINARY DATA]"

          _ ->
            "[UNKNOWN DATA]"
        end
    end
  rescue
    _ -> "[INVALID DATA]"
  end

  # Make sure all data is JSON encodable
  defp sanitize_for_json(value) when is_binary(value) do
    # Remove any invalid UTF-8 sequences
    cond do
      String.valid?(value) and not is_binary_data?(value) ->
        value

      true ->
        "[BINARY DATA]"
    end
  end

  defp sanitize_for_json(value), do: to_string(value)

  @doc """
  Saves the extracted table data to a file.
  """
  def save_table_data(table, columns, records) do
    file_path = Path.join(@mdb_records_dir, "#{table}.json")

    # Filter out non-encodable data
    safe_records =
      Enum.map(records, fn record ->
        Enum.map(record, fn {key, value} ->
          {key, sanitize_for_json(value)}
        end)
        |> Enum.into(%{})
      end)
      |> Enum.filter(fn map ->
        map != %{} && !only_binary_data?(map) &&
          not Enum.any?(
            Map.values(map),
            &(is_binary(&1) && String.contains?(&1, "[BINARY DATA]"))
          )
      end)

    data = %{
      table: table,
      columns: columns,
      records: safe_records
    }

    json_content = Jason.encode!(data, pretty: true)
    File.write!(file_path, json_content)

    # Also save as JSONL
    jsonl_path = Path.join(@mdb_records_dir, "#{table}.jsonl")

    jsonl_content =
      Enum.map_join(safe_records, "\n", fn record ->
        # Format records before encoding to JSON
        formatted_record =
          cond do
            # Handle special case for malformed records
            is_map(record) && map_size(record) == 1 &&
                (Enum.at(Map.keys(record), 0) == "Gutachten-Nr" ||
                   Enum.at(Map.keys(record), 0) == "Schaden-Nr" ||
                   Enum.at(Map.keys(record), 0) == "Versicherungsnehmer") ->
              nil

            # Filter out continuation rows (records with only these three fields)
            is_map(record) && map_size(record) == 3 &&
              Map.has_key?(record, "Gutachten-Nr") &&
              Map.has_key?(record, "Schaden-Nr") &&
              Map.has_key?(record, "Versicherungsnehmer") &&
                (String.contains?(Map.get(record, "Gutachten-Nr", ""), "\"") ||
                   String.match?(Map.get(record, "Schaden-Nr", ""), ~r/\d{2}\/\d{2}\/\d{2}/) ||
                   Map.get(record, "Versicherungsnehmer") in ["0", "1", "2"]) ->
              nil

            # Normal record
            is_map(record) && map_size(record) > 0 ->
              # Format values before final sanitization pass
              Enum.map(record, fn {k, v} ->
                formatted_value = format_field_value(k, v)
                {k, sanitize_final_json_value(formatted_value)}
              end)
              |> Enum.into(%{})

            # Empty or invalid record
            true ->
              nil
          end

        # Only encode valid records
        if formatted_record, do: Jason.encode!(formatted_record), else: nil
      end)
      |> String.split("\n")
      # Remove nil entries
      |> Enum.filter(&(&1 != "nil" && &1 != ""))
      |> Enum.join("\n")

    File.write!(jsonl_path, jsonl_content)
  end

  # Format field values based on field name
  defp format_field_value(field_name, value) when is_binary(value) do
    case field_name do
      # Format currency fields to 2 decimal places
      field
      when field in [
             "Akutschaden",
             "Anschaffungspreis brutto",
             "Gutachtenumsatz netto",
             "Sonstiges",
             "Verschleißschaden",
             "Wiederbeschaffungswert brutto"
           ] ->
        format_currency_value(value)

      # Format date fields with proper date format
      field
      when field in ["Anschaffungsdatum", "Auftragsdatum", "Schadentag", "Stichtag", "Datum"] ->
        format_date_value(value)

      # Return other values unchanged
      _ ->
        value
    end
  end

  defp format_field_value(_field_name, value), do: value

  # Format currency values to have exactly 2 decimal places
  defp format_currency_value(value) do
    case Float.parse(value) do
      {float_val, _} ->
        # Format to have exactly two decimal places
        :io_lib.format("~.2f", [float_val]) |> to_string()

      _ ->
        value
    end
  end

  # Format date values to have proper date format (DD.MM.YYYY)
  defp format_date_value(value) do
    # Clean the value by removing spaces
    cleaned_value = String.trim(value)

    cond do
      # Handle empty dates
      cleaned_value == "" ->
        ""

      # Handle dates that already have formatting
      String.contains?(cleaned_value, ".") ->
        cleaned_value

      # Handle typical 8-digit format like "14052003" for 14.05.2003
      String.length(cleaned_value) == 8 ->
        <<day::binary-size(2), month::binary-size(2), year::binary-size(4)>> = cleaned_value
        "#{day}.#{month}.#{year}"

      # Handle typical 6-digit format like "140503" for 14.05.2003
      String.length(cleaned_value) == 6 ->
        <<day::binary-size(2), month::binary-size(2), year::binary-size(2)>> = cleaned_value
        "#{day}.#{month}.20#{year}"

      # Handle special formats like "  092005" (spaces for missing day)
      String.match?(cleaned_value, ~r/^\s*\d{6}$/) ->
        cleaned = String.replace(cleaned_value, ~r/\s+/, "0")
        <<day::binary-size(2), month::binary-size(2), year::binary-size(2)>> = cleaned
        "#{day}.#{month}.20#{year}"

      # Match MM/DD/YY or MM/DD/YYYY followed by 00:00:00
      String.match?(cleaned_value, ~r/^(\d{2})\/(\d{2})\/(\d{2,4}) 00:00:00$/) ->
        regex = ~r/^(\d{2})\/(\d{2})\/(\d{2,4}) 00:00:00$/

        case Regex.run(regex, cleaned_value) do
          [_, mm, dd, yy] ->
            yyyy =
              case String.length(yy) do
                2 ->
                  y = String.to_integer(yy)
                  if y < 30, do: "20#{yy}", else: "19#{yy}"

                4 ->
                  yy

                _ ->
                  yy
              end

            "#{dd}.#{mm}.#{yyyy}"

          _ ->
            cleaned_value
        end

      # Return unchanged if can't parse
      true ->
        cleaned_value
    end
  end

  # Final sanitization pass to catch any remaining binary data
  defp sanitize_final_json_value(value) when is_binary(value) do
    cond do
      String.starts_with?(value, "[BINARY DATA]") -> "[BINARY DATA]"
      not String.valid?(value) -> "[BINARY DATA]"
      is_binary_data?(value) -> "[BINARY DATA]"
      true -> value
    end
  end

  defp sanitize_final_json_value(value), do: value

  defp decode_field("[BINARY DATA]", raw_binary) when is_binary(raw_binary) do
    Enum.find_value(@encodings, fn enc ->
      case :iconv.convert(enc, "utf-8", raw_binary) do
        {:ok, decoded} -> decoded
        _ -> nil
      end
    end) || "BINARY"
  end

  defp decode_field(value, _raw_binary), do: value

  defp convert_us_date_with_time(str) when is_binary(str) do
    # Match MM/DD/YY or MM/DD/YYYY followed by 00:00:00
    regex = ~r/^(\d{2})\/(\d{2})\/(\d{2,4}) 00:00:00$/

    case Regex.run(regex, str) do
      [_, mm, dd, yy] ->
        yyyy =
          case String.length(yy) do
            2 ->
              y = String.to_integer(yy)
              if y < 30, do: "20#{yy}", else: "19#{yy}"

            4 ->
              yy

            _ ->
              yy
          end

        "#{dd}.#{mm}.#{yyyy}"

      _ ->
        str
    end
  end

  defp convert_us_date_with_time(str), do: str

  # Function specifically for processing JSONL files
  def process_jsonl_file(file_path) do
    # Read the JSONL file
    file_path
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(fn line ->
      case Jason.decode(line) do
        {:ok, json_entry} ->
          process_json_entry(json_entry)

        _ ->
          nil
      end
    end)
    |> Enum.filter(& &1)
  end

  # Process a single JSON entry, converting dates properly
  def process_json_entry(entry) when is_map(entry) do
    entry
    |> Enum.map(fn {k, v} ->
      # Apply date conversion to all string values
      new_value = if is_binary(v), do: convert_us_date_with_time(v), else: v
      {k, new_value}
    end)
    |> Enum.into(%{})
  end

  def process_json_entry(_), do: nil

  # This function is for processing MDB data entries
  defp process_entry(entry) when is_map(entry) do
    entry
    |> Enum.map(fn
      {k, "[BINARY DATA]"} = pair ->
        # Try to get the raw binary for this field if available
        raw_binary = Map.get(entry, :"#{k}_raw")
        {k, decode_field("[BINARY DATA]", raw_binary)}

      {k, v} when is_binary(v) ->
        # Apply date conversion to all string values
        {k, convert_us_date_with_time(v)}

      pair ->
        pair
    end)
    |> Enum.into(%{})
  end

  defp process_entry(_), do: nil

  def extract_entries(raw_entries) do
    raw_entries
    |> Enum.map(&process_entry/1)
    |> Enum.filter(fn
      nil -> false
      map when is_map(map) -> map != %{} and not malformed_entry?(map)
      _ -> false
    end)
  end

  # New function to extract and format entries from JSONL files
  def extract_json_entries(json_entries) do
    json_entries
    |> Enum.map(&process_json_entry/1)
    |> Enum.filter(fn
      nil -> false
      map when is_map(map) -> map != %{} and not malformed_entry?(map)
      _ -> false
    end)
  end

  defp malformed_entry?(map) do
    # Add more sophisticated checks if needed
    Enum.any?(Map.values(map), fn v ->
      is_binary(v) and String.match?(v, ~r/^[^"]*Schadenstag \d:/)
    end)
  end
end
