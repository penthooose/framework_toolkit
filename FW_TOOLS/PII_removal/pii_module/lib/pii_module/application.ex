defmodule PiiModule.Application do
  use Application

  def start(_type, _args) do
    children = [
      {AnalyzerServer, []}
    ]

    opts = [strategy: :one_for_one, name: PiiModule.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
