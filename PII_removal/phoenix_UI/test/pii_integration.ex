defmodule PhoenixUI.PiiIntegrationTest do
  use ExUnit.Case

  # Replace PiiModule with your actual module name
  test "can access PII module functions" do
    # Try to use a function from your PII module
    result = MainPii.testmodule()
    assert result != nil
  end
end
