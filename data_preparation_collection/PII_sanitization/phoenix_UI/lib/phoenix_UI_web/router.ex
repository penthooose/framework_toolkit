defmodule Phoenix_UIWeb.Router do
  use Phoenix_UIWeb, :router

  pipeline :browser do
    plug(:accepts, ["html"])
    plug(:fetch_session)
    plug(:fetch_live_flash)
    plug(:put_root_layout, html: {Phoenix_UIWeb.Layouts, :root})
    plug(:protect_from_forgery)
    plug(:put_secure_browser_headers)
  end

  pipeline :api do
    plug(:accepts, ["json"])
  end

  scope "/", Phoenix_UIWeb do
    pipe_through(:browser)
    live("/", OverviewLive)
    live("/pii", PIILive)
    live("/pii/config", PIIConfigurator)
    live("/pii/industrial", PIIIndustrial)
    live("/tools", Tools)
    get("/download/:filename", DownloadController, :download)
    get("/download-zip/:dirname", DownloadController, :download_zip)
    # get "/", PageController, :home
  end

  # Add this new scope for API routes
  scope "/api", Phoenix_UIWeb do
    pipe_through(:api)
    post("/upload", UploadController, :create)
  end

  # Enable LiveDashboard and Swoosh mailbox preview in development
  if Application.compile_env(:phoenix_UI, :dev_routes) do
    # If you want to use the LiveDashboard in production, you should put
    # it behind authentication and allow only admins to access it.
    # If your application does not have an admins-only section yet,
    # you can use Plug.BasicAuth to set up some basic authentication
    # as long as you are also using SSL (which you should anyway).
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through(:browser)

      live_dashboard("/dashboard", metrics: Phoenix_UIWeb.Telemetry)
      forward("/mailbox", Plug.Swoosh.MailboxPreview)
    end
  end
end
