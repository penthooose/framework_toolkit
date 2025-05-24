defmodule Phoenix_UIWeb.Components.CustomRecognizerForm do
  use Phoenix.Component

  def custom_recognizer_form(assigns) do
    ~H"""
    <div class="custom-recognizer mt-8 p-4 bg-gray-50 rounded-lg">
      <div class="flex items-center justify-between cursor-pointer" phx-click="toggle_custom_recognizer">
        <h3 class="text-lg font-semibold">Add Custom Recognizer</h3>
        <span class="text-xl"><%= if @custom_recognizer_expanded, do: "▼", else: "▶" %></span>
      </div>

      <%= if @custom_recognizer_expanded do %>
        <.recognizer_form {assigns} />
      <% end %>
    </div>
    """
  end

  def recognizer_form(assigns) do
    ~H"""
    <%= if @custom_recognizer_expanded do %>
        <div class="mt-4">
          <!-- Recognizer Type Selector -->
          <div class="mb-4">
            <form phx-change="recognizer_type_changed">
              <select name="type" class="w-full p-2 border rounded">
                <option value="example_based" selected={@recognizer_type == "example_based"}>Example-Based Pattern</option>
                <option value="regex_based" selected={@recognizer_type == "regex_based"}>Direct Regex Pattern</option>
                <option value="deny_list" selected={@recognizer_type == "deny_list"}>Deny List</option>
              </select>
            </form>
          </div>

          <%= case @recognizer_type do %>
            <% "example_based" -> %>
              <form phx-submit="generate_regex" class="space-y-4">
                <div>
                  <input type="text"
                        name="name"
                        value={@recognizer_name || ""}
                        placeholder="Example-based pattern name"
                        required
                        class="w-full p-2 border rounded" />
                </div>
                <div>
                  <input type="text"
                        name="examples"
                        value={Enum.join(@current_examples, ", ") || ""}
                        placeholder="Examples (comma separated)"
                        required
                        class="w-full p-2 border rounded" />
                  <p class="text-sm text-gray-600 mt-1">Enter at least 2 examples, separated by commas</p>
                </div>

                <div>
                  <input type="text"
                        name="context"
                        value={Enum.join(@recognizer_context, ", ") || ""}
                        placeholder="Context (optional)"
                        class="w-full p-2 border rounded" />
                  <p class="text-sm text-gray-600 mt-1">Optional context for the recognizer</p>
                </div>
                <div>
                  <select name="language" class="w-full p-2 border rounded">
                    <option value="any" selected={@recognizer_language == "any"}>Any Language</option>
                    <option value="de" selected={@recognizer_language == "de"}>German</option>
                    <option value="en" selected={@recognizer_language == "en"}>English</option>
                    <option value="fr" selected={@recognizer_language == "fr"}>French</option>
                    <option value="es" selected={@recognizer_language == "es"}>Spanish</option>
                  </select>
                </div>
                <button type="submit" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                  Generate Regex
                </button>
              </form>

              <%= if @generated_regex do %>
                <form phx-submit="add_regex" class="mt-4 space-y-4">
                  <input type="hidden" name="name" value={@recognizer_name} />
                  <input type="hidden" name="context" value={Enum.join(@recognizer_context, ",")} />
                  <input type="hidden" name="language" value={@recognizer_language} />
                  <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Generated/Modified Regex:</label>
                    <input
                      type="text"
                      name="regex"
                      value={@current_regex_input || @generated_regex}
                      phx-change="update_regex_input"
                      class="w-full p-2 border rounded"
                      id="regex-input" />
                    <div class="mt-2 flex gap-2">
                      <button
                        type="button"
                        phx-click="restore_regex"
                        class="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-sm">
                        Restore Original
                      </button>
                      <button
                        type="button"
                        phx-click="validate_modified_regex"
                        phx-value-regex={@current_regex_input}
                        class="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600">
                        Validate
                      </button>
                    </div>
                  </div>
                  <button
                    type="submit"
                    class={"px-4 py-2 rounded #{if @recognizer_valid, do: 'bg-green-500 text-white hover:bg-green-600', else: 'bg-slate-100 text-slate-400 border border-slate-300 cursor-not-allowed'}"}
                    disabled={!@recognizer_valid}>
                    Add Recognizer
                  </button>
                </form>
              <% end %>

            <% "regex_based" -> %>
              <form phx-submit="add_custom_regex" class="space-y-4">
                <div>
                  <input type="text" name="name" placeholder="Regex pattern name" value={@recognizer_name || ""} required class="w-full p-2 border rounded" />
                </div>
                <div>
                  <input type="text" name="regex" placeholder="Regular expression pattern" value={@current_regex_input || ""} required class="w-full p-2 border rounded" />
                </div>
                <div>
                  <input type="text" name="context" placeholder="Context (optional)" value={Enum.join(@recognizer_context, ",") || ""} class="w-full p-2 border rounded" />
                </div>
                <div>
                  <select name="language" class="w-full p-2 border rounded">
                    <option value="any" selected={@recognizer_language == "any"}>Any Language</option>
                    <option value="de" selected={@recognizer_language == "de"}>German</option>
                    <option value="en" selected={@recognizer_language == "en"}>English</option>
                    <option value="fr" selected={@recognizer_language == "fr"}>French</option>
                    <option value="es" selected={@recognizer_language == "es"}>Spanish</option>
                  </select>
                </div>
                <button type="submit" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                  Add Regex Pattern
                </button>
              </form>

            <% "deny_list" -> %>
              <form phx-submit="add_recognizer_deny_list" class="space-y-4">
                <div>
                  <input type="text" name="name" placeholder="Deny list name" value={@recognizer_name || ""} required class="w-full p-2 border rounded" />
                </div>
                <div>
                  <textarea name="deny_list" placeholder="Enter one or multiple words (one word per line)" required class="w-full p-2 border rounded" rows="4"><%= Enum.join(@current_examples, "\n") || "" %></textarea>
                </div>
                <div>
                  <input type="text" name="context" placeholder="Context (optional)" value={Enum.join(@recognizer_context, ",") || ""} class="w-full p-2 border rounded" />
                </div>
                <div>
                  <select name="language" class="w-full p-2 border rounded">
                    <option value="any" selected={@recognizer_language == "any"}>Any Language</option>
                    <option value="de" selected={@recognizer_language == "de"}>German</option>
                    <option value="en" selected={@recognizer_language == "en"}>English</option>
                    <option value="fr" selected={@recognizer_language == "fr"}>French</option>
                    <option value="es" selected={@recognizer_language == "es"}>Spanish</option>
                  </select>
                </div>
                <button type="submit" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                  Add Deny List
                </button>
              </form>
          <% end %>

          <%= if @loading_add_recognizer do %>
            <div class="mt-4">
              <div class="spinner"></div>
            </div>
          <% end %>
        </div>
      <% end %>
    """
  end
end
