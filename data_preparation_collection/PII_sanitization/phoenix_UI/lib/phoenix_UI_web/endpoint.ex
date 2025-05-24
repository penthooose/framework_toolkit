defmodule Phoenix_UIWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :phoenix_UI

  # The session will be stored in the cookie and signed,
  # this means its contents can be read but not tampered with.
  # Set :encryption_salt if you would also like to encrypt it.
  @session_options [
    store: :cookie,
    key: "_phoenix_UI_key",
    signing_salt: "FlffrNX9",
    same_site: "Lax"
  ]

  socket("/live", Phoenix.LiveView.Socket,
    websocket: [
      connect_info: [session: @session_options],
      timeout: 60_000,
      transport_options: [
        max_connections: 16384
      ]
    ],
    longpoll: [connect_info: [session: @session_options]]
  )

  # Serve at "/" the static files from "priv/static" directory.
  #
  # You should set gzip to true if you are running phx.digest
  # when deploying your static files in production.
  plug(Plug.Static,
    at: "/downloads",
    from: {:phoenix_UI, "priv/static/downloads"},
    gzip: false
  )

  plug(Plug.Static,
    at: "/",
    from: :phoenix_UI,
    gzip: false,
    only: Phoenix_UIWeb.static_paths(),
    # Add this line
    only_matching: ["downloads"]
  )

  # Code reloading can be explicitly enabled under the
  # :code_reloader configuration of your endpoint.
  if code_reloading? do
    socket("/phoenix/live_reload/socket", Phoenix.LiveReloader.Socket)
    plug(Phoenix.LiveReloader)
    plug(Phoenix.CodeReloader)
    plug(Phoenix.Ecto.CheckRepoStatus, otp_app: :phoenix_UI)
  end

  plug(Phoenix.LiveDashboard.RequestLogger,
    param_key: "request_logger",
    cookie_key: "request_logger"
  )

  plug(Plug.RequestId)
  plug(Plug.Telemetry, event_prefix: [:phoenix, :endpoint])

  plug(Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library(),
    # 3000MB limit
    length: 30_000_000_000,
    multipart: [
      # Same 300MB limit for multipart
      length: 30_000_000_000,
      max_files: 200
    ]
  )

  plug(Plug.MethodOverride)
  plug(Plug.Head)
  plug(Plug.Session, @session_options)
  plug(Phoenix_UIWeb.Router)
end
