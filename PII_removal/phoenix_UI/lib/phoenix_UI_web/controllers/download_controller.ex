defmodule Phoenix_UIWeb.DownloadController do
  use Phoenix_UIWeb, :controller

  @output_dir "files/conversion_output"

  def download(conn, %{"filename" => filename}) do
    file_path = Path.join(@output_dir, filename)
    IO.puts("Download requested for: #{file_path}")
    IO.puts("File exists: #{File.exists?(file_path)}")
    IO.puts("Directory contents: #{inspect(File.ls!(@output_dir))}")

    if File.exists?(file_path) do
      conn
      |> put_resp_header("content-disposition", ~s(attachment; filename="#{filename}"))
      |> put_resp_content_type("application/octet-stream")
      |> send_file(200, file_path)
    else
      conn
      |> put_status(:not_found)
      |> text(
        "File not found: #{file_path} (Directory contents: #{inspect(File.ls!(@output_dir))})"
      )
    end
  end

  def download_zip(conn, %{"dirname" => dirname}) do
    dir_path = Path.join(@output_dir, dirname)

    if File.dir?(dir_path) do
      {:ok, zip_path} = create_zip(dir_path)

      conn
      |> put_resp_header("content-disposition", ~s(attachment; filename="#{dirname}.zip"))
      |> put_resp_content_type("application/zip")
      |> send_file(200, zip_path)
    else
      conn
      |> put_status(:not_found)
      |> text("Directory not found: #{dir_path}")
    end
  end

  defp create_zip(dir_path) do
    zip_path = Path.join(System.tmp_dir!(), "#{Path.basename(dir_path)}.zip")
    File.rm(zip_path)

    {:ok, _} =
      :zip.create(
        String.to_charlist(zip_path),
        get_files_to_zip(dir_path) |> Enum.map(&String.to_charlist/1),
        cwd: String.to_charlist(Path.dirname(dir_path))
      )

    {:ok, zip_path}
  end

  defp get_files_to_zip(dir_path) do
    Path.wildcard(Path.join(dir_path, "**/*"))
    |> Enum.filter(&File.regular?/1)
    |> Enum.map(&Path.relative_to(&1, Path.dirname(dir_path)))
  end
end
