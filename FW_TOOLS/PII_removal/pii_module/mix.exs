defmodule PiiModule.MixProject do
  use Mix.Project

  def project do
    [
      app: :pii_module,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {PiiModule.Application, []}
    ]
  end

  defp deps do
    [
      {:erlport, "~> 0.11"},
      {:jason, "~> 1.4"},
      {:httpoison, "~> 2.2"},
      {:cowlib, "~> 2.13.0", override: true},
      {:yaml_elixir, "~> 2.11.0"},
      {:ymlr, "~> 5.0"}
    ]
  end
end
