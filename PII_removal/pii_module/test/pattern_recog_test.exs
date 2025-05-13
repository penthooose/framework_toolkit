defmodule PatternRecogTest do
  use ExUnit.Case
  doctest PatternRecognizer

  test "derive regex from examples" do
    examples = ["123-45-6789", "987-65-4321"]
    assert RegexGenerator.derive_regex(examples) == "\\b(\\d{3}-\\d5-\\d{4})\\b"
  end

  test "generate regex from string" do
    string = ["Fabi ist dum"]
    assert RegexGenerator.derive_regex(string) == "\\b(Fabi ist dum)\\b"
  end

  test "check if examples fit regex" do
    examples = ["123-45-6789", "987-65-4321"]
    regex = "\\b(\\d{3}-\\d5-\\d{4})\\b"
    assert RegexGenerator.check_regex_fitting(regex, examples) == true
  end

  test "check if examples fit regex 2" do
    examples = ["123-45-6789", "987-65-432s"]
    regex = "\\b(\\d{3}-\\d5-\\d{4})\\b"
    assert RegexGenerator.check_regex_fitting(regex, examples) == false
  end

  test "validate regex on invalid regex" do
    assert RegexGenerator.validate_regex("\\b(\\d{0}))-/d5-\\d{0})\\b") ==
             {:error, "Invalid regex pattern"}
  end

  test "validate regex on valid regex" do
    assert RegexGenerator.validate_regex("\\b(\\d{1}-\\d5-\\d{2})\\b") ==
             {:ok, "Valid regex pattern"}
  end

  test "validate regex fails with unmatched parenthesis" do
    assert RegexGenerator.validate_regex("\\b(\\d{3}") ==
             {:error, "Invalid regex pattern"}
  end

  test "validate regex fails with unmatched bracket" do
    assert RegexGenerator.validate_regex("\\b[A-Z\\b") ==
             {:error, "Invalid regex pattern"}
  end

  test "validate regex fails with invalid escape" do
    assert RegexGenerator.validate_regex("\\b\\k\\b") ==
             {:error, "Invalid regex pattern"}
  end
end
