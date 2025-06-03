import re, os
from collections import OrderedDict

INPUT = "logging_phase3"


def extract_training_metrics(log_file_path, output_file_path=None):
    """
    Extract training metrics from log file, keeping only unique validation results,
    first training progress line, and final test metrics.

    Args:
        log_file_path (str): Path to the input log file
        output_file_path (str): Path to save filtered results (optional)

    Returns:
        list: Filtered log lines
    """

    # Patterns to match different types of lines
    patterns = {
        "eval_metrics": re.compile(
            r".*'eval_loss'.*'eval_runtime'.*'eval_samples_per_second'.*"
        ),
        "training_progress": re.compile(
            r".*Training progress:.*'loss':.*'grad_norm':.*"
        ),
        "final_test": re.compile(r".*Final test metrics:.*'eval_loss'.*"),
    }

    filtered_lines = []
    seen_eval_lines = set()
    seen_all_lines = set()  # Track all lines to prevent duplicates
    first_training_progress_found = False

    try:
        with open(log_file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                # Check for final test metrics
                if patterns["final_test"].search(line):
                    if line not in seen_all_lines:
                        seen_all_lines.add(line)
                        filtered_lines.append(line)
                    continue

                # Check for evaluation metrics
                if patterns["eval_metrics"].search(line):
                    # Extract the metrics dictionary part to check for duplicates
                    metrics_match = re.search(r"(\{.*\})", line)
                    if metrics_match:
                        metrics_dict = metrics_match.group(1)
                        if (
                            metrics_dict not in seen_eval_lines
                            and line not in seen_all_lines
                        ):
                            seen_eval_lines.add(metrics_dict)
                            seen_all_lines.add(line)
                            filtered_lines.append(line)
                    continue

                # Check for first training progress line
                if (
                    patterns["training_progress"].search(line)
                    and not first_training_progress_found
                ):
                    # Make sure it's not an eval line
                    if "'eval_loss'" not in line and line not in seen_all_lines:
                        seen_all_lines.add(line)
                        filtered_lines.append(line)
                        first_training_progress_found = True
                    continue

        # Sort lines by timestamp to maintain chronological order
        filtered_lines.sort(key=lambda x: extract_timestamp(x))

        # Remove any remaining duplicates that might have been introduced during sorting
        unique_filtered_lines = []
        seen_after_sort = set()
        for line in filtered_lines:
            if line not in seen_after_sort:
                seen_after_sort.add(line)
                unique_filtered_lines.append(line)

        filtered_lines = unique_filtered_lines

        # Print results
        print(f"Extracted {len(filtered_lines)} unique training metric lines:")
        print("-" * 80)

        for line in filtered_lines:
            print(line)

        # Save to file if output path provided
        # if output_file_path:
        #     with open(output_file_path, "w", encoding="utf-8") as output_file:
        #         for line in filtered_lines:
        #             output_file.write(line + "\n")
        #     print(f"\nResults saved to: {output_file_path}")

        return filtered_lines

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def extract_timestamp(log_line):
    """Extract timestamp from log line for sorting"""
    timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", log_line)
    if timestamp_match:
        return timestamp_match.group(1)
    return ""


def extract_metrics_only(log_file_path, output_file_path=None):
    """
    Alternative function that extracts only the metrics dictionaries
    """
    filtered_lines = extract_training_metrics(log_file_path)

    metrics_only = []
    for line in filtered_lines:
        # Extract just the metrics dictionary
        metrics_match = re.search(r"(\{.*\})", line)
        if metrics_match:
            metrics_dict = metrics_match.group(1)
            # Also extract epoch info for context
            timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            timestamp = timestamp_match.group(1) if timestamp_match else ""

            # Determine line type
            if "Final test metrics" in line:
                line_type = "FINAL TEST VALIDATION"
            elif "'eval_loss'" in metrics_dict and "'eval_runtime'" in metrics_dict:
                line_type = "VALIDATION"
            else:
                line_type = "TRAINING"

            metrics_only.append(f"{timestamp} - {line_type}: {metrics_dict}")

    print("\n" + "=" * 80)
    print("METRICS ONLY:")
    print("=" * 80)
    for metric_line in metrics_only:
        print(metric_line)

    if output_file_path:
        metrics_output_path = output_file_path.replace(".txt", "_metrics_only.txt")
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            for metric_line in metrics_only:
                f.write(metric_line + "\n")
        print(f"\nMetrics-only results saved to: {metrics_output_path}")

    return metrics_only


# Example usage
if __name__ == "__main__":
    # Update this path to your log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

    input_folder = "./" + INPUT
    logging_dir = os.path.join(script_dir, input_folder)
    output_folder = "./filtered_" + INPUT
    output_dir = os.path.join(script_dir, output_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Logging directory: {logging_dir}")
    print(f"Output directory: {output_dir}")

    # Find all logging files that start with "logging_"
    if os.path.exists(logging_dir):
        logging_files = [
            f
            for f in os.listdir(logging_dir)
            if f.startswith("logging_") and f.endswith(".txt")
        ]

        if not logging_files:
            print("No logging files found that start with 'logging_'")
        else:
            print(f"Found {len(logging_files)} logging files: {logging_files}")

            for logging_file in logging_files:
                print(f"\n{'='*80}")
                print(f"Processing: {logging_file}")
                print(f"{'='*80}")

                logging_path = os.path.join(logging_dir, logging_file)

                # Create output filename by replacing "logging_" with "filtered_logging_"
                output_filename = logging_file.replace("logging_", "filtered_logging_")
                output_file_path = os.path.join(output_dir, output_filename)

                # Extract full lines
                filtered_lines = extract_training_metrics(
                    logging_path, output_file_path
                )

                # Extract only metrics dictionaries
                metrics_only = extract_metrics_only(logging_path, output_file_path)

                print(f"\nSummary for {logging_file}:")
                print(f"- Total filtered lines: {len(filtered_lines)}")
                print(
                    f"- Unique validation checkpoints: {len([l for l in filtered_lines if 'eval_loss' in l and 'eval_runtime' in l])}"
                )
                print(
                    f"- Training progress lines: {len([l for l in filtered_lines if 'Training progress' in l and 'eval_loss' not in l])}"
                )
                print(
                    f"- Final test lines: {len([l for l in filtered_lines if 'Final test metrics' in l])}"
                )
    else:
        print(f"Logging directory not found: {logging_dir}")
