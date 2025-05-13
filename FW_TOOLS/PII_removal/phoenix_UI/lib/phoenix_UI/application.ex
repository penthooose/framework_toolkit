defmodule Phoenix_UI.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      Phoenix_UIWeb.Telemetry,
      Phoenix_UI.Repo,
      {DNSCluster, query: Application.get_env(:phoenix_UI, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: Phoenix_UI.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: Phoenix_UI.Finch},
      # Start a worker by calling: Phoenix_UI.Worker.start_link(arg)
      # {Phoenix_UI.Worker, arg},

      # Add PIIState initialization
      {Task,
       fn ->
         Phoenix_UI.State.PIIState.start_agents()
         IO.puts("PII State Agents initialized")
       end},

      # Start to serve requests, typically the last entry
      Phoenix_UIWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: Phoenix_UI.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    Phoenix_UIWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
