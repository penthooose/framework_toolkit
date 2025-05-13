defmodule Phoenix_UI.Repo do
  use Ecto.Repo,
    otp_app: :phoenix_UI,
    adapter: Ecto.Adapters.Postgres
end
