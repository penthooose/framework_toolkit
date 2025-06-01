defmodule ModelToolsController do
  alias ModelConverter
  alias ModelQuantizer

  def convert_and_integrate_model(file_param_entry \\ "3") do
    ModelConverter.convert_safetensors_to_gguf(file_param_entry)
  end

  def integrate_model_in_ollama(file_param_entry \\ "2") do
    ModelConverter.integrate_gguf_in_ollama(file_param_entry)
  end

  def quantize_model(file_param_entry \\ "2") do
    ModelQuantizer.quantize_model(file_param_entry)
  end
end
