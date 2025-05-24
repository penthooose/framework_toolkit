defmodule APIClient do
  # Update this with the correct base URL for your FastAPI application
  @base_url "http://localhost:50000"

  # Function to check if the service is running
  def health_check() do
    case HTTPoison.get("#{@base_url}/") do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        {:ok, Jason.decode!(body)}

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        {:error, "Error #{status_code}: #{body}"}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end

  # Function to analyze text
  def analyze(text) do
    body = %{"text" => text}
    headers = [{"Content-Type", "application/json"}]

    case HTTPoison.post("#{@base_url}/analyze/", Jason.encode!(body), headers) do
      {:ok, %HTTPoison.Response{status_code: 200, body: response_body}} ->
        # Use Jason.decode without ! to handle potential errors
        case Jason.decode(response_body) do
          {:ok, decoded_body} ->
            {:ok, decoded_body}

          {:error, decoding_error} ->
            {:error, "Failed to decode response: #{decoding_error}"}
        end

      {:ok, %HTTPoison.Response{status_code: status_code, body: response_body}} ->
        {:error, "Error #{status_code}: #{response_body}"}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end

  # Function to add a custom recognizer
  def add_custom_recognizer_regex(pattern_name, regex) do
    body = %{"pattern_name" => pattern_name, "regex" => regex}
    headers = [{"Content-Type", "application/json"}]

    case HTTPoison.post("#{@base_url}/add_custom_recognizer/", Jason.encode!(body), headers) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        {:ok, Jason.decode!(body)}

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        {:error, "Error #{status_code}: #{body}"}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end

  def add_custom_recognizer_examples(pattern_name, examples) do
    body = %{"pattern_name" => pattern_name, "examples" => examples}
    headers = [{"Content-Type", "application/json"}]

    case HTTPoison.post("#{@base_url}/add_custom_recognizer/", Jason.encode!(body), headers) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        {:ok, Jason.decode!(body)}

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        {:error, "Error #{status_code}: #{body}"}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end
end
