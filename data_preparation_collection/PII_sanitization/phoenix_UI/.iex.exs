defmodule IExHelpers do
  def restart do
    IEx.Helpers.recompile()
    Phoenix_UIWeb.ServerHelpers.restart()
  end
end

import IExHelpers
