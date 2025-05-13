# This file is responsible for configuring your application
# and its dependencies with the aid of the Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.

# General application configuration
import Config

config :phoenix_UI,
  ecto_repos: [Phoenix_UI.Repo],
  generators: [timestamp_type: :utc_datetime]

config :phoenix_UI, Phoenix_UI.Repo,
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  database: "phoenix_ui_dev",
  stacktrace: true,
  show_sensitive_data_on_connection_error: true,
  pool_size: 10

# Add these configurations
config :phoenix_UI,
  upload_directory: Path.expand("files"),
  input_directory: Path.expand("files/conversion_input"),
  output_directory: Path.expand("files/conversion_output")

# Configures the endpoint
config :phoenix_UI, Phoenix_UIWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [html: Phoenix_UIWeb.ErrorHTML, json: Phoenix_UIWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: Phoenix_UI.PubSub,
  live_view: [signing_salt: "/GCrF8L5"],
  http: [ip: {127, 0, 0, 1}, port: 4000]

# Configures the mailer
#
# By default it uses the "Local" adapter which stores the emails
# locally. You can see the emails in your browser, at "/dev/mailbox".
#
# For production it's recommended to configure a different adapter
# at the `config/runtime.exs`.
config :phoenix_UI, Phoenix_UI.Mailer, adapter: Swoosh.Adapters.Local

# Configure esbuild (the version is required)
config :esbuild,
  version: "0.17.11",
  phoenix_UI: [
    args:
      ~w(js/app.js --bundle --target=es2017 --outdir=../priv/static/assets --external:/fonts/* --external:/images/*),
    cd: Path.expand("../assets", __DIR__),
    env: %{"NODE_PATH" => Path.expand("../deps", __DIR__)}
  ]

# Configure esbuild
config :esbuild,
  version: "0.17.11",
  app: [
    args: ~w(js/app.js --bundle --target=es2016 --outdir=../priv/static/assets),
    cd: Path.expand("../assets", __DIR__),
    env: %{"NODE_PATH" => Path.expand("../deps", __DIR__)}
  ]

# Configure tailwind (the version is required)
config :tailwind,
  version: "3.4.3",
  phoenix_UI: [
    args: ~w(
      --config=tailwind.config.js
      --input=css/app.css
      --output=../priv/static/assets/app.css
    ),
    cd: Path.expand("../assets", __DIR__)
  ]

# Configures Elixir's Logger
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]

# Suppress Bandit and ThousandIsland error logs
config :logger, level: :info

config :logger, :bandit, level: :warn

config :logger, :thousand_island, level: :warn

# Use Jason for JSON parsing in Phoenix
config :phoenix, :json_library, Jason

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"
