defmodule SE.SymbolicExtractor do
  # define custom rules with exception possible transformations on the captured text
  # Format: {key, regex_pattern, transformations}
  rules = [
    {:Aktenzeichen, ~r/Aktenzeichen\s*:\s*([^\n]+)/, [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Aktenzeichen_RA, ~r/Aktenzeichen\s+RA\s*:\s*([^\n]+)/, [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Aktenzeichen_StA, ~r/Aktenzeichen der StA\s*:\s*([^\n]+)/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Aktenzeichen_Polizei, ~r/Aktenzeichen Polizei\s*:\s*([^\n]+)/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Anlagen_Nr, ~r/Anlagen-Nr\.\s*:\s*([^\n]+)/, nil},
    {:Anlagentyp, ~r/Anlagentyp\s*:\s*([^\n]+)/, nil},
    {:Anschaffung, ~r/Anschaffung\s*:\s*([^\n]+)/, nil},
    {:Anschaffungsdatum,
     ~r/(?:\*\*)?Anschaffungsdatum(?:\/Baujahr)?(?:\s*\*\*)?\s*:(?:\s*\*\*)?\s*([^\n*]+)/, nil},
    {:Anschaffungsjahr, ~r/Anschaffungsjahr\s*:\s*([^\n]+)/, nil},
    {:Anschaffungskosten, ~r/Anschaffungskosten\s*:\s*([^\n]+)/, nil},
    {:Anschaffungspreis,
     ~r/(?:\*\*)?Anschaffungspreise?(?:\s*\*\*)?\s*:(?:\s*\*\*)?\s*([^\n*]+)(?:\s+€\s+(?:brutto|netto))?/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Anschaffungszeitraum, ~r/Anschaffungszeitraum\s*:\s*([^\n]+)/, nil},
    {:Anschrift,
     ~r/\A(.*?\b\d{5}\s+[A-Z][a-zäöüß\-]+(?:\s+[A-Z]?[a-zäöüß\-]+)*)(?:\s+GA\s+\d+\/\d+)?(?=\s*\n+(?:Nr\.?|GA|Vom|\d{2}\.\d{2}\.\d{4})|\s*\z)/s,
     [{~r/\n+/, "\n"}]},
    {:Artikelnummer, ~r/(?:Art\.|Artikel)\s*Nr\.?\s*:\s*([^\n]+)/, nil},
    {:Auftragsbeschreibung, ~r/Auftragsbeschreibung\s*:\s*([^\n]+)/, nil},
    {:Auftragsdatum, ~r/Auftragsdatum\s*:\s*([^\n]+)/, nil},
    {:Auftraggeber, ~r/Auftraggeber\s*:\s*([^\n]+)/, nil},
    {:Auftraggeber_Adresse,
     ~r/Auftraggeber\s*:\s*[^\n]+\n+([^:\n]+(?:\n+[^:\n]+)*?(?:\n+\d{5}\s+[^\n]+))/s,
     [{~r/\n+/, "\n"}]},
    {:Auftraggeber_Email,
     ~r/Auftrag erteilt durch\s*:\s*[^\n]+(?:\n+.+\d{5}\s+[^\n]+)(?:\n+.+\d{5}\s+[^\n]+)(?=\n)/s,
     [{~r/\n+/, "\n"}]},
    {:Auftraggeber_Firma, ~r/Auftrag erteilt durch\s*:\s*[^\n]+\n+([^:\n]+?)(?=\n+\w+\s*:)/s,
     [{~r/^\s+|\s+$/, ""}]},
    {:Auftraggeber_Telefax,
     ~r/Auftrag erteilt durch\s*:\s*[^\n]+(?:\n+.+\d{5}\s+[^\n]+)(?:\n+.+\d{5}\s+[^\n]+)(?=\n)/s,
     [{~r/\n+/, "\n"}]},
    {:Auftraggeber_Telefon,
     ~r/Auftrag erteilt durch\s*:\s*[^\n]+(?:\n+.+\d{5}\s+[^\n]+)(?:\n+.+\d{5}\s+[^\n]+)(?=\n)/s,
     [{~r/\n+/, "\n"}]},
    {:Auftraggeber_Website,
     ~r/Auftrag erteilt durch\s*:\s*[^\n]+(?:\n+.+\d{5}\s+[^\n]+)(?:\n+.+\d{5}\s+[^\n]+)(?=\n)/s,
     [{~r/\n+/, "\n"}]},
    {:Auftragerteiler, ~r/Auftrag erteilt durch\s*:\s*([^,\n]+?)(?:,|\s*(?=\n))/, nil},
    {:Auftragerteiler_Titel, ~r/Auftrag erteilt durch\s*:.*?\n+\s*(Beauftragter\s+für\s+[^\n]+)/,
     nil},
    {:Auftragsart, ~r/Auftragsart\s*:\s*([^\n]+)/, nil},
    {:Auftragsnummer, ~r/Auftragsnummer\s*:\s*([^\n]+)/, nil},
    {:Austauschpreis, ~r/Austauschpreis(?:\s+\d{4})?\s*:\s*([^\n]+)/, nil},
    {:Bestellnummer, ~r/Bestellnummer\s*:\s*([^\n]+)/, nil},
    {:Betreiberadresse, ~r/Betreiberadresse\s*:\s*([^\n]+(?:\n+[^:\n]+)*?(?:\n+\d{5}\s+[^\n]+))/s,
     [{~r/\n+/, "\n"}]},
    {:Baujahr, ~r/(?:\*\*)?Baujahr(?:\s*\*\*)?\s*:(?:\s*\*\*)?\s*([^\n*]+)/, nil},
    {:GA_Datum_Erstellung,
     ~r/(?:.*?GA.*?\n*\s*(?:Vom\s+)?(\d{2}\.\d{2}\.\d{4}))|(?:GA\s+\d+\/\d+\s+(\d{2}\.\d{2}\.\d{4}))|(?:GA\s+\d+\/\d+\s*\n*\s*(\d{2}\.\d{2}\.\d{4}))/s,
     nil},
    {:Dokumentation, ~r/(?:Zugrundeliegende\s+)?Dokumentation\s*:\s*([^\n]+)/, nil},
    {:Equipment_Nr, ~r/Equipment(?:\/KS-|\s*)Nr\.\s*:\s*([^\n]+)/, nil},
    {:Folgemodell,
     ~r/(?:Für\s+)?(?:\(?(?:Folgemodell|Folgetyp)(?:\s*\*\*)?\s*:?\s*([^)\n]*(?:,[^)\n]*)?)(?:\)?|\s*$)|(?:Folgetyp\s*:\s*([A-Za-z0-9\-\s]+))\))/,
     nil},
    {:Folgemodell_2,
     ~r/\(([A-Za-z0-9\-]+)\s+(?:Folgemodell|Folgetyp)\)|(?:\(Folgetyp:\s*([A-Za-z0-9\-\s]+)\))/,
     nil},
    {:GA_Nr, ~r/(?:Nr\.: )?(GA \d+\/\d+)/, nil},
    {:GA_Bezeichnung, ~r/G\s*U\s*T\s*A\s*C\s*H\s*T\s*E\s*N/, [{~r/^.*$/, "Gutachten"}]},
    {:GA_Bezeichnung_2, ~r/K\s*U\s*R\s*Z(?:\s*-+)?/, [{~r/^.*$/, "Kurz-Gutachten"}]},
    {:Gehört_zu, ~r/gehört\s+zu\s*:\s*([^\n]+)/, nil},
    {:Geräteart, ~r/Geräteart\s*:\s*([^\n]+)/, nil},
    {:Gerätebeschreibung, ~r/^(.{80,})$|^(Der untersuchungsgegenständliche.+)$/, nil},
    {:Gerätetyp, ~r/Gerätetyp\s*:\s*([^\n]+)/, nil},
    {:Gutachtenart, ~r/Gutachtenart\s*:\s*([^\n]+)/, nil},
    {:Hersteller,
     ~r/(?:\*\*)?Hersteller(?:\/Vertrieb)?(?:\s*\*\*)?\s*:(?:\s*\*\*)?\s*([^\n]+(?:\\[^\n]*)*)/,
     [{~r/\\(?!\s*\n|\s*$)/, "\n"}]},
    {:Hersteller_Adresse,
     ~r/Hersteller\s*:\s*[^\n\\\r]+\\?(?:\r?\n|\\\s*)((?:.+\\?\s*\n?)+?(?:\d{5}\s+[^\n]+))/s,
     [
       # Clean up backslashes that might appear at line ends
       {~r/\\(\s*\n|\s*$)/, "\n"},
       # Replace multiple newlines with single newlines
       {~r/\n+/, "\n"}
     ]},
    {:Installationsdatum, ~r/Installationsdatum\s*:\s*([^\n]+)/, nil},
    {:Leistung, ~r/Leistung\s*:\s*([^\n]+)/, nil},
    {:Letzte_Wartung, ~r/Letzte\s+(?:Anlagen)?[Ww]artung\s*:\s*([^\n]+)/, nil},
    {:Makler, ~r/Makler\s*(?:\(ggf\.\s*Schadennummer\))?\s*:\s*([^\n]+)/, nil},
    {:Makler_Nr, ~r/Makler\s+Nr\.\s*:\s*([^\n]+)/, nil},
    {:Makler_Zeichen, ~r/Makler(?:\s*-+)?(?:zeichen|(?:\s+Zeichen))\s*:\s*(.*?)(?=\n)/s,
     [
       {~r/^\s*\\?--\s*$/, "unbekannt"},
       {~r/\\(\$)/, "$"}
     ]},
    {:Medizinprodukteart, ~r/Medizinprodukteart\s*:\s*([^\n]+)/, nil},
    {:Medizinproduktetypen, ~r/Medizinproduktetyp(?:en)?\s*:\s*((?:(?:-|\\\-)[^\n]+\n+)+)/s,
     [
       # Remove empty lines
       {~r/\n+/, "\n"},
       # Remove "-" and "\-" prefixes and trim
       {~r/(?:^|\n)(?:-|\\\-)?\s*/, "\n"},
       # Remove leading and trailing newlines
       {~r/^\n|\n$/, ""},
       # Replace newlines with commas
       {~r/\n/, ", "}
     ]},
    {:Modell_Nr, ~r/(?:Modell\s+Nr\.?|Modell)\s*:\s*([^\n]+)/, nil},
    {:Neupreis,
     ~r/(?:\*\*)?(?:Listen-)?Neupreis(?:\s+[^:\n]*)?(?:\s*\*\*)?(?:\s*)?(?:\[(?:\^|fn)\d+\])?(?:,?\s*(?:\((?:Liste(?:\s+\d{4})?|Liste\s+international|Liste\s+ca\.|\d{4}|brutto|ca\.)\)|\s*Liste)|\s*\(\d{4}\)|\s*\(Liste\s+\d{4}\)|\s*\(Liste\s+international\)|\s*Liste\s+\(\d{4}\)|\s*\(brutto\)|(?:\s+aktuell)|(?:\s+ca\.\s+\d{4}))?\s*(?:,?\s*ca\.?)?\s*(?:\d{1,2}\/\d{4})?\s*(?:\*\*)?\s*:(?:\s*\*\*)?\s*([^\n*]+)(?:\s+(?:\([^)\n]+\)|\(historisch\)))?/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Neupreis_2,
     ~r/(?:\*\*)?Neupreis(?:\s*\*\*)?.*?:(?:\s*\*\*)?\s*[^\n*]+\n+\s*((?:\d[\d\.]*[\d,]+\s*[€$]\s*(?:brutto|netto))(?:\n+\s*(?:\d[\d\.]*[\d,]+\s*[€$]\s*(?:brutto|netto)))?)/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Neupreis_3, ~r/Neupreis\s+ca\.\s*\(Liste\s+\d{4}\)\s*:\s*([^\n]+)/, nil},
    {:Lieferantennummer, ~r/Lieferantennummer\s*:\s*([^\n]+)/, nil},
    {:Listenpreis,
     ~r/(?:\*\*)?Listenpreis(?:\s*\*\*)?(?:\s*\((?:NEU(?:\s+\d{4})?|Liste(?:\s+\d{4})?|\d{4})\)|\s+(?:aktuell|[Nn]eu(?:\s+\(?(?:\d{4})\)?)?)|(?:\s+\d+\.\)(?:\s*--\s*\d+\.\))?)?|\s+\d{4})?\s*(?:,?\s*ca\.?)?\s*(?:\*\*)?\s*:(?:\s*\*\*)?\s*([^\n*]+)/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Listenpreis_Folgetyp,
     ~r/(?:\*\*)?Neupreis\s+Folgetyp(?:\s*\*\*)?(?:\s*\((?:National|International)?\))?(?:\s*,?\s*ca\.?)?\s*(?:\*\*)?\s*:(?:\s*\*\*)?\s*([^\n*]+)/,
     nil},
    {:LOT, ~r/LOT\s*(?:\(Charge\))?\s*:\s*([^\n]+)/, nil},
    {:Paketpreis,
     ~r/((?:\d[\d\.]*[\d,]+)\s*[€$](?:\s*[a-zäöüß]+)?\s*(?:brutto\s*)?\(Paketpreis\))/, nil},
    {:Polizei_Aktenzeichen, ~r/Polizei\s+Aktenzeichen\s*:\s*([^\n]+)/, nil},
    {:Produktart, ~r/(?:\*\*)?Produktart(?:\s*\*\*)?\s*:(?:\s*\*\*)?\s*([^\n\(]+)(?!\s*\n\s*\()/,
     nil},
    {:Produktart_Kategorie,
     ~r/(?:\*\*)?Produktart(?:\s*\*\*)?\s*:(?:\s*\*\*)?\s*[^\n*]+\n+\s*\(([^\)]+)\)(?!\s*:)/,
     [{~r/^\s+|\s+$/, ""}]},
    {:Produkt_Nr, ~r/(?:Produkt\s+Nr\.|P\/N)\s*:?\s*([^\n]+)/, nil},
    {:Produkttyp,
     ~r/(?:\*\*)?Produkttype?n?(?:\s*\*\*)?(?:\s*\([^\)]*\))?\s*:(?:\s*\*\*)?\s*([^\n*]+)/, nil},
    {:Projektnummer, ~r/Projektnummer\s*:\s*([^\n]+)/, nil},
    {:Referenznummer, ~r/Referenznummer\s*:\s*([^\n]+)/, nil},
    {:Ref, ~r/(?:REF|ref|Ref)\s*(?:Nr\.?|No)?\s*:\s*([^\n]+)/, nil},
    {:Sache, ~r/Sache(?:\/Objekt)?\s*:\s*([^\n]+)/, nil},
    {:Sachverständiger, ~r/Sachverständiger\s*:\s*([^,\n]+)/, nil},
    {:Sachverständiger_2,
     ~r/Sachverständige\s*:\s*(?:[^,\n]+\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?=\s*$|\n)/, nil},
    {:Sachverständiger_Titel,
     ~r/Sachverständiger\s*:(?:.*?,)?\s*(.+?(?:\n+Sachverständiger für .*?)?)(?=\n\n|\n+[^S]|\z)/s,
     [{~r/\n+/, "\n"}]},
    {:Sachverständiger_Titel_2,
     ~r/Sachverständige\s*:\s*([^,\n]+?)(?=\s+[A-Z][a-z]+\s+[A-Z][a-z]+)/, nil},
    {:Schadennummer, ~r/Schadennummern?\s*:\s*([^\n]*)/, nil},
    {:Schaden_ID_Versicherungsnehmer, ~r/Schaden-ID\s*\n+\s*Versicherungsnehmer\s*:\s*([^\n]+)/,
     nil},
    {:Schadennummer_Makler, ~r/Schadennummer(?:\s+|\s*\()Makler(?:\))?\s*:\s*([^\n]*)/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:Schadennummer_Makler_VNr,
     ~r/Schadennummer\s+Makler\s*(?:\/|\s*\/\s*)\s*VN\s*:\s*([^\n]+?)(?=\n)/, nil},
    {:Schadennummer_Versicherer, ~r/Schadennummer Versicherer\s*:\s*([^\n]*)/, nil},
    {:Schadenobjekt, ~r/Schadenobjekt\s*:\s*([^\n]+)/, nil},
    {:Schadensache, ~r/Schadensache\s*:\s*([^\n]+)(?:\n+\s*\(([^\)]+)\))?/,
     [{~r/\n+\s*\(([^\)]+)\)/, "\n$1"}]},
    {:Schadentag, ~r/Schadentag\s*:\s*([^\n]+)/, nil},
    {:Schadentag_Meldung, ~r/Schadentag\/Schadenmeldung\s*:\s*([^\n]+)/, nil},
    {:Schadentag_Zeitraum, ~r/Schadentag\/-zeitraum\s*:\s*([^\n]+)/, nil},
    {:Serie_Nr,
     ~r/(?:\*\*)?Serie(?:\s*\*\*)?\s*Nr\.?\s*(?:\*\*)?\s*:?(?:\s*\*\*)?\s*([^\n*]+)|Serie\s+Nr\.\s+([^\n]+)|Serie\s+Nr\.\s*:\s*([^\n]+)|Serie\s+Nr\.\s*([^\n]+)/,
     [{~r/^\s*\\?--\s*$/, "unbekannt"}]},
    {:SK_Nr, ~r/SK\s+(?:Nummer|Nr\.?)\s*:?\s*([^\n]+)/, nil},
    {:Software, ~r/Software\s*:\s*([^\n]+)/, nil},
    {:Sondenneupreis, ~r/Sondenneupreis\s*\(Liste\)\s*:\s*([^\n]+)/, nil},
    {:Standort, ~r/Standort\s*:\s*([^\n]+)/, nil},
    {:Standort_Adresse, ~r/Standort\s*:\s*([^\n]*(?:\n+[^:\n]+)*?(?:\n+\d{5}\s+[^\n]+))/s,
     [{~r/\n+/, " "}, {~r/\s{2,}/, " "}, {~r/,\s+/, ", "}]},
    {:System_Nr, ~r/System\s+Nr\.\s*:\s*([^\n]+)/, nil},
    {:Technical_ID, ~r/Technical\s+ID\s*:\s*([^\n]+)/, nil},
    {:Störmeldung, ~r/Störmeldung\s*:\s*([^\n]+)/, nil},
    {:Untersuchungsort,
     ~r/Untersuchungsort\s*:\s*([^\n]+(?:\n+[^:\n]+)*?(?:\n+\d{4,5}\s+[^\n]+))/s,
     [{~r/\n+/, "\n"}]},
    {:Versicherungsnehmer, ~r/Versicherungsnehmer\s*:\s*([^\n]+?)(?=\n)/, nil},
    {:Versicherungsnehmer_Adresse,
     ~r/Versicherungsnehmer\s*:[^\n]+\n+((?:.*?\n+)*?.*?\d{4,5}\s+[A-Z][a-zäöüß\-]+(?:\s+[A-Z]?[a-zäöüß\-]+)*)(?=\n|$)/s,
     [{~r/\n+/, "\n"}]},
    {:Versicherungsscheinnummer, ~r/Versicherungsscheinnummer\s*:\s*([\w\s\-\/.]+?)(?=\n)/, nil},
    {:Vertragsnummer, ~r/Vertragsnummer(?:\s+Versicherer)?\s*:\s*([\w\s\-\/.]+?)(?=\n)/, nil},
    {:Verwendbarkeit, ~r/Verwendbarkeit\s*:\s*([^\n]+)/, nil},
    {:Verwendungshäufigkeit, ~r/Verwendungshäufigkeit\s*:\s*([^\n]+)/,
     [{~r/^\s*\\?--\s*$|^\s*\.\.\s*$/, "unbekannt"}]},
    {:Wiederbeschaffungswert, ~r/Wiederbeschaffungswert\s*:\s*([^\n]+)/, nil},
    {:Zeichen, ~r/Zeichen\s*:\s*([^\n]+)/, nil},
    {:Zubehör, ~r/Zubehör\s*:\s*([^\n]+)/, nil},
    {:Zulassung, ~r/Zulassung\s*:\s*([^\n]+)/, nil}
  ]

  exception_rules = [
    ~r/(?:\#{0,3})\s*G\s*U\s*T\s*A\s*C\s*H\s*T\s*E\s*N\s*/i,
    ~r/(?:#\s*)?\d+(?:\.\s*)?(?:Technische\s+)?Gerätedaten/i,
    ~r/Makler(?:\s*-+)?(?:zeichen|(?:\s+Zeichen)).*$/i,
    # Add this new rule to catch standalone "Makler -" lines
    ~r/^Makler\s*-+$/i,
    # Add rule to catch standalone "Vom DD.MM.YYYY" dates
    ~r/^Vom\s.*$/,
    # Add rule to catch standalone "### " lines
    ~r/###\s*/,
    ~r/##/,
    # Add rule to catch "2. Technische Daten" lines
    ~r/2\.\s+Technische\s+Daten/,
    ~r/2\.\s*Technische\s*Daten\s*/,
    ~r/2\.\s*Untersuchungsgegenständliche\s*Geräte/
    # ~r/^Für\s/
  ]

  def extract(content) do
    # Apply each rule and extract information with a generalized approach
    Enum.reduce(get_rules(), %{}, fn
      {key, pattern, transformations}, acc ->
        case Regex.run(pattern, content) do
          [_, capture] ->
            # Only add if key doesn't already exist in the accumulator
            if Map.has_key?(acc, Atom.to_string(key)) do
              acc
            else
              transformed_value = apply_transformations(capture, transformations)
              trimmed_value = String.trim(transformed_value)

              if trimmed_value != "",
                do: Map.put(acc, Atom.to_string(key), trimmed_value),
                else: acc
            end

          [capture] ->
            # Only add if key doesn't already exist in the accumulator
            if Map.has_key?(acc, Atom.to_string(key)) do
              acc
            else
              transformed_value = apply_transformations(capture, transformations)
              trimmed_value = String.trim(transformed_value)

              if trimmed_value != "",
                do: Map.put(acc, Atom.to_string(key), trimmed_value),
                else: acc
            end

          _ ->
            acc
        end
    end)
  end

  def detect_uncaptured_content(content, extracted_info) do
    # Extract content only up to the first chapter
    content_before_first_chapter =
      case Regex.run(~r/\A(.*?)(?=# 1\. |\z)/s, content) do
        [_, capture] -> capture
        [capture] -> capture
        nil -> content
      end

    # Start with the preprocessed content
    remaining_content = content_before_first_chapter

    # Track which parts of the content were matched by our rules
    matched_sections = []

    # Add matched content to our tracking list
    matched_sections =
      Enum.reduce(get_rules(), matched_sections, fn {key, pattern, _}, acc ->
        case Regex.run(pattern, content) do
          [full_match | _] ->
            [{full_match, Atom.to_string(key)} | acc]

          _ ->
            acc
        end
      end)

    # Remove all matched content from the document
    remaining_content =
      Enum.reduce(matched_sections, remaining_content, fn {match, _}, acc ->
        String.replace(acc, match, "", global: false)
      end)

    # Apply exception rules to further filter out specific patterns
    remaining_content =
      Enum.reduce(get_exception_rules(), remaining_content, fn pattern, acc ->
        Regex.replace(pattern, acc, "", global: true)
      end)

    # Clean up the remaining content
    remaining_content =
      remaining_content
      # Replace multiple newlines with double newlines
      |> String.replace(~r/\n{3,}/, "\n\n")
      # Trim whitespace
      |> String.replace(~r/^\s+|\s+$/, "")
      |> String.trim()

    # Split by newlines to check each line separately
    lines = String.split(remaining_content, ~r/\n+/)

    # Filter only lines that are not empty and don't appear in any extracted value
    unmatched_lines =
      Enum.filter(lines, fn line ->
        trimmed = String.trim(line)
        not_empty = trimmed != ""
        not_in_extracted = !content_in_extracted_values?(trimmed, extracted_info)
        not_makler_dash = not Regex.match?(~r/^Makler\s*-+$/i, trimmed)
        not_vom_date = not Regex.match?(~r/^Vom\s+\d{2}\.\d{2}\.\d{4}$/, trimmed)
        not_empty && not_in_extracted && not_makler_dash && not_vom_date
      end)

    # Return non-empty content or nil
    case Enum.join(unmatched_lines, "\n") do
      "" -> nil
      content -> content
    end
  end

  # Helper function to check if a line appears in any extracted value
  defp content_in_extracted_values?(line, extracted_info) do
    Enum.any?(extracted_info, fn {_, value} ->
      String.contains?(value, line)
    end)
  end

  # Helper function to apply transformations to captured text
  defp apply_transformations(text, nil), do: text

  defp apply_transformations(text, transformations) do
    # First remove any trailing asterisks that might have been captured
    text = Regex.replace(~r/\s*\*+\s*$/, text, "")

    result =
      Enum.reduce(transformations, text, fn {pattern, replacement}, acc ->
        Regex.replace(pattern, acc, replacement)
      end)

    # For things that are all uppercase, convert to lowercase and capitalize
    if result =~ ~r/^[A-Z\s]+$/ do
      result |> String.downcase() |> String.capitalize()
    else
      result
    end
  end

  # Make rules accessible as a function
  def get_rules, do: unquote(Macro.escape(rules))

  # Make exception rules accessible as a function
  def get_exception_rules, do: unquote(Macro.escape(exception_rules))

  # Function to get the keys of the rules
  def get_rules_keys do
    Enum.map(get_rules(), fn {key, _, _} -> key end)
  end
end
