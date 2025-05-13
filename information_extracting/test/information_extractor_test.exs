defmodule InformationExtractorTest do
  use ExUnit.Case
  doctest InformationExtractor

  test "greets the world" do
    assert InformationExtractor.hello() == :world
  end
end
