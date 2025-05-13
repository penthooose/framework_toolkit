defmodule ModelConverterTest do
  use ExUnit.Case
  doctest ModelConverter

  test "greets the world" do
    assert ModelConverter.hello() == :world
  end
end
