defmodule Phoenix_UIWeb.UploadController do
  use Phoenix_UIWeb, :controller

  @input_dir "files/conversion_input"
  @output_dir "files/conversion_output"

  def create(conn, %{"file" => upload}) do
    # Ensure directories exist
    File.mkdir_p!(@input_dir)
    File.mkdir_p!(@output_dir)

    # Create a unique filename
    filename = "#{System.unique_integer([:positive])}_#{upload.filename}"
    file_path = Path.join(@input_dir, filename)

    # Copy the uploaded file
    File.cp!(upload.path, file_path)

    # Create corresponding output path
    output_path = Path.join(@output_dir, filename)

    json(conn, %{
      path: file_path,
      output_path: output_path
    })
  end
end
