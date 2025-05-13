defmodule Phoenix_UIWeb.PageControllerTest do
  use Phoenix_UIWeb.ConnCase

  test "GET /", %{conn: conn} do
    assert get(conn, "/").status == 200
  end
end
