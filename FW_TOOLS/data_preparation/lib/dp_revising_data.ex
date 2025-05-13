defmodule DP.RevisingData do
  alias DP.PrepareFollowingChapters

  @path_statistics_fragmented_summary "N:/Thesis/data_prepare/statistics/revise_summary_creation.json"
  @input_revised_summaries "N:/Thesis/data_prepare/information_extraction/sub_symbolic_revise"
  @building_supervised_2 "N:/Thesis/data_prepare/datasets_building/supervised/multiple_chapters_format2"

  def revise_data(building_dir \\ @building_supervised_2) do
    integrate_revised_summaries(building_dir)
    PrepareFollowingChapters.prepare_following_chapters_supervised(building_dir)
  end

  def integrate_revised_summaries(
        building_dir \\ @building_supervised_2,
        revised_summaries \\ @input_revised_summaries
      ) do
    # Get all parent directories in the input path
    parent_dirs = File.ls!(revised_summaries)

    # Process each parent directory
    Enum.each(parent_dirs, fn dir_name ->
      # Path to source summary file
      source_summary_path =
        Path.join([revised_summaries, dir_name, "extracted_summary.json"])

      # Path to target directory and its summary file
      target_dir = Path.join(building_dir, dir_name)
      target_summary_path = Path.join([target_dir, "extracted_summary.json"])

      if File.exists?(source_summary_path) && File.exists?(target_summary_path) do
        # Read source summary file
        source_content =
          source_summary_path
          |> File.read!()
          |> Jason.decode!()

        # Read target summary file
        target_content =
          target_summary_path
          |> File.read!()
          |> Jason.decode!()

        # Update target with source summaries
        updated_target = update_summaries(target_content, source_content)

        # Write the updated content back to the target file
        File.write!(target_summary_path, Jason.encode!(updated_target, pretty: true))

        IO.puts("Updated summaries for directory: #{dir_name}")
      else
        IO.puts("Skipping directory: #{dir_name} - required files not found")
      end
    end)

    IO.puts("Summary integration complete")
  end

  # Helper function to update summaries in target content with those from source
  defp update_summaries(target_content, source_content) do
    Enum.reduce(source_content, target_content, fn {filename, source_data}, acc ->
      case Map.get(acc, filename) do
        nil ->
          # If the file doesn't exist in target, we skip it
          acc

        target_data ->
          # Update the summary in the target data
          updated_data = Map.put(target_data, "summary", source_data["summary"])
          Map.put(acc, filename, updated_data)
      end
    end)
  end
end
