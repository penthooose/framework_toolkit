import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

path_logging = "logging_phase3"
output_path = path_logging + "_curves"
description = {
    # "logging_phase1": {1: "Initial Test"},
    "logging_phase2": {
        1: "Evaluation Performance: Stage A",
        2: "Evaluation Performance: Stage B",
        3: "Evaluation Performance: Stage C",
        4: "Evaluation Performance: Stage D",
        5: "Evaluation Performance: Stage E",
    },
    "logging_phase3": {
        1: "Evaluation Performance: V1-A",
        2: "Evaluation Performance: V1-B",
        3: "Evaluation Performance: V2-A",
        4: "Evaluation Performance: V2-B",
        5: "Evaluation Performance: V3-C1",
        6: "Evaluation Performance: V3-C2",
    },
}
cut_entries = 6


def find_logging_files(logging_dir="./logging"):
    """Find and sort logging files by their numeric suffix."""
    # Ensure the logging directory exists
    if not os.path.exists(logging_dir):
        print(f"Error: Directory {logging_dir} not found")
        return []

    # Find all .txt files in the directory
    log_files = glob.glob(os.path.join(logging_dir, "*.txt"))

    # Extract numbers from filenames and sort
    def extract_number(filename):
        match = re.search(r"(\d+)", os.path.basename(filename))
        if match:
            return int(match.group(1))
        return float("inf")  # Files without numbers go last

    # Sort files by their numeric part
    sorted_files = sorted(log_files, key=extract_number)

    if not sorted_files:
        print(f"No log files found in {logging_dir}")
    else:
        print(
            f"Found {len(sorted_files)} log files: {[os.path.basename(f) for f in sorted_files]}"
        )

    return sorted_files


def extract_data_from_log(log_file):
    """Extract training metrics from log file."""
    data = {
        "train_epochs": [],
        "train_losses": [],
        "eval_epochs": [],
        "eval_losses": [],
        "lr_epochs": [],
        "learning_rates": [],
    }

    try:
        with open(log_file, "r") as file:
            content = file.read()

            # Extract training data
            train_pattern = re.compile(r"{'loss': ([\d\.]+).*?'epoch': ([\d\.]+)}")
            for match in train_pattern.finditer(content):
                data["train_losses"].append(float(match.group(1)))
                data["train_epochs"].append(float(match.group(2)))

            # Extract evaluation data
            eval_pattern = re.compile(r"{'eval_loss': ([\d\.]+).*?'epoch': ([\d\.]+)}")
            temp_eval_data = set()  # Use a set to store unique (epoch, loss) tuples
            for match in eval_pattern.finditer(content):
                epoch = float(match.group(2))
                loss = float(match.group(1))
                temp_eval_data.add((epoch, loss))

            # Convert back to lists and sort by epoch
            sorted_eval_data = sorted(temp_eval_data)
            data["eval_epochs"] = [item[0] for item in sorted_eval_data]
            data["eval_losses"] = [item[1] for item in sorted_eval_data]

            # Extract learning rate data
            lr_pattern = re.compile(
                r"'learning_rate': ([\d\.eE\-\+]+).*?'epoch': ([\d\.]+)"
            )
            for match in lr_pattern.finditer(content):
                data["learning_rates"].append(float(match.group(1)))
                data["lr_epochs"].append(float(match.group(2)))

        print(
            f"Extracted from {os.path.basename(log_file)}: {len(data['train_losses'])} train points, "
            f"{len(data['eval_losses'])} eval points, {len(data['learning_rates'])} lr points"
        )

    except Exception as e:
        print(f"Error extracting data from {log_file}: {e}")

    return data


def get_loss_info(train_losses, eval_losses):
    """Generate loss information string for plot legends."""
    loss_info = ""

    if train_losses:
        first_loss = train_losses[0]
        last_loss = train_losses[-1]
        train_reduction = 100 * (1 - last_loss / first_loss)
        loss_info += f" - Train: {last_loss:.4f} (-{train_reduction:.1f}%)"

    if eval_losses:
        if loss_info:
            loss_info += ""
        first_eval = eval_losses[0]
        last_eval = eval_losses[-1]
        eval_reduction = 100 * (1 - last_eval / first_eval)
        loss_info += f"\n - Eval: {last_eval:.4f} (-{eval_reduction:.1f}%)\n"

    return loss_info


def plot_loss_curves(ax, data, stage_name):
    """Plot loss curves on the given axis."""
    train_epochs = data["train_epochs"]
    train_losses = data["train_losses"]
    eval_epochs = data["eval_epochs"]
    eval_losses = data["eval_losses"]

    # Debug information
    print(f"Processing evaluation points: {len(eval_losses)} points found")

    # Set axis labels and title
    ax.set_xlabel("Epoch", fontsize=12, labelpad=10)
    ax.set_ylabel("Loss", fontsize=12, labelpad=10)
    ax.set_title(f"Loss Curves", fontsize=13, pad=10)

    # Plot training loss with points and moving average
    if train_losses:
        ax.plot(
            train_epochs, train_losses, "bo", alpha=0.3, markersize=2, label="Train"
        )

        # Add moving average for training loss
        window_size = min(10, max(3, len(train_losses) // 10))
        if len(train_losses) > window_size:
            moving_avg = np.convolve(
                train_losses, np.ones(window_size) / window_size, mode="valid"
            )
            ma_epochs = train_epochs[window_size - 1 :]
            ax.plot(
                ma_epochs,
                moving_avg,
                "b-",
                linewidth=2,
                label=f"Train MA({window_size})",
            )

    # Plot evaluation loss with annotations
    if eval_losses:
        ax.plot(eval_epochs, eval_losses, "ro", markersize=6, label="Eval")

        # Debug the extracted data
        print(f"Evaluation points data: {list(zip(eval_epochs, eval_losses))[:5]}...")

        # Handle different cases based on actual number of points, not filtered
        total_points = len(eval_losses)
        print(f"Number of eval points to annotate: {total_points}")

        if total_points <= 5:
            # For 5 or fewer points, use simple logic
            # Skip the point right before the last one if we have exactly 5 points
            if total_points == 5:
                indices_to_annotate = [0, 1, 2, 4]
            else:
                indices_to_annotate = list(range(total_points))
        else:
            # For more than 5 points, use the complex logic
            last_idx = total_points - 1

            # Define indices that should NEVER be annotated (to avoid overlap with last point)
            excluded_indices = set([last_idx - 1, last_idx - 2, last_idx - 3])

            # Define target number of annotations (including first and last)
            target_annotations = min(6, len(eval_losses) // 3)

            # Create initial set with first and last point
            indices_to_annotate = [0, last_idx]

            if last_idx > 4:
                # Use a more deterministic approach to select evenly spaced points
                # This ensures better distribution and prevents adjacency
                if target_annotations > 2:
                    # Calculate how many points to add between first and last
                    middle_points = target_annotations - 2

                    # Divide the range evenly
                    step = last_idx / (middle_points + 1)

                    for i in range(1, middle_points + 1):
                        # Calculate the position
                        pos = int(i * step)
                        # Make sure it's not in the excluded range (points near the end)
                        if pos not in excluded_indices and pos > 0 and pos < last_idx:
                            indices_to_annotate.append(pos)

                # Always try to add a point near 80% of the way through if not already covered
                percentage_pos = int(last_idx * 0.8)
                if (
                    percentage_pos not in indices_to_annotate
                    and percentage_pos not in excluded_indices
                ):
                    indices_to_annotate.append(percentage_pos)

                # Sort indices to maintain consistency
                indices_to_annotate = sorted(indices_to_annotate)

                # Ensure no consecutive points (final check)
                final_indices = [indices_to_annotate[0]]  # Always include first point
                for idx in indices_to_annotate[1:]:
                    # Only add point if it's at least 2 positions away from the last added point
                    if idx - final_indices[-1] >= 2:
                        final_indices.append(idx)

                indices_to_annotate = final_indices

        # For few points, annotate all except potentially overlapping ones
        if len(eval_losses) <= 3:
            indices_to_annotate = [
                i
                for i in range(len(eval_losses))
                if i == 0 or i == len(eval_losses) - 1 or i < len(eval_losses) - 4
            ]

        # Create annotations
        for i in indices_to_annotate:
            if i < len(eval_losses):
                ax.annotate(
                    f"{eval_losses[i]:.4f}",
                    (eval_epochs[i], eval_losses[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    loss_info = get_loss_info(train_losses, eval_losses)

    # Create legend with adjusted line spacing
    legend = ax.legend(title=f"Final Results:")

    # Access the legend's title and set line spacing property
    title = legend.get_title()
    title.set_text(f"Final Results:\n{loss_info}")

    # Set line spacing - increase the value for more spacing
    plt.setp(legend.get_texts(), linespacing=1.7)
    plt.setp(title, linespacing=1.7)


def plot_learning_rate(ax, data):
    """Plot learning rate on the given axis."""
    lr_epochs = data["lr_epochs"]
    learning_rates = data["learning_rates"]

    # Set axis labels and title
    ax.set_xlabel("Epoch", fontsize=12, labelpad=10)
    ax.set_ylabel("Learning Rate", fontsize=12, labelpad=10)
    ax.set_title("Learning Rate Curve", fontsize=13, pad=10)

    # Plot learning rate with points and moving average
    if learning_rates:
        ax.plot(lr_epochs, learning_rates, "go", alpha=0.5, markersize=4, label="LR")

        # Add moving average for learning rate
        window_size = min(10, max(3, len(learning_rates) // 10))
        if len(learning_rates) > window_size:
            lr_ma = np.convolve(
                learning_rates, np.ones(window_size) / window_size, mode="valid"
            )
            ma_epochs = lr_epochs[window_size - 1 :]
            ax.plot(ma_epochs, lr_ma, "g-", linewidth=2, label=f"LR MA({window_size})")

        # Log scale for learning rate if all values are positive
        if learning_rates and min(learning_rates) > 0:
            ax.set_yscale("log")

        # Add final learning rate to legend
        if learning_rates:
            lr_info = f"Final: {learning_rates[-1]:.2e}"
            ax.legend(title=lr_info)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)


def get_stage_name_from_file(log_file, logging_dir, description_dict):
    """Extract stage name from log file using description dictionary."""
    # Get the logging directory name to match with description keys
    logging_dir_name = os.path.basename(logging_dir.rstrip(os.sep))

    # Get the appropriate description mapping for this logging directory
    if logging_dir_name in description_dict:
        phase_descriptions = description_dict[logging_dir_name]
    else:
        phase_descriptions = {}

    # Extract number from filename
    match = re.search(r"(\d+)", os.path.basename(log_file))
    if match:
        file_number = int(match.group(1))
        if file_number in phase_descriptions:
            return phase_descriptions[file_number]

    # Fallback to filename if no description found
    return f"Stage: {os.path.basename(log_file)}"


def create_multi_stage_loss_curve(
    log_files, output_path="fine_tuning_stages.png", stage_names=None, logging_dir=None
):
    """Generate loss and learning rate curves for multiple fine-tuning stages."""

    if not isinstance(log_files, list):
        log_files = [log_files]

    if stage_names is None:
        stage_names = [
            get_stage_name_from_file(f, logging_dir, description) for f in log_files
        ]

    # Create a figure with grid of subplots (stages × metrics) - add extra height for title
    fig, axes = plt.subplots(len(log_files), 2, figsize=(15, 5 * len(log_files) + 1))

    # Only set main title for single stage with valid description
    if len(log_files) == 1:
        # Only create main title if we have a valid description (not fallback)
        if stage_names[0] != f"Stage: {os.path.basename(log_files[0])}":
            main_title = f"Fine-Tuning Metrics: {stage_names[0]}"
            fig.suptitle(main_title, fontsize=16, y=0.95)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(
        top=0.85, wspace=0.3
    )  # Reduce top to 0.85 to create more space above plots

    # Track min/max values for consistent y-axis scaling
    loss_min, loss_max = float("inf"), float("-inf")

    # First pass to collect data and determine scales
    all_data = []
    for stage_idx, log_file in enumerate(log_files):
        data = extract_data_from_log(log_file)
        all_data.append(data)
        if data["train_losses"]:
            loss_min = min(loss_min, min(data["train_losses"]))
            loss_max = max(loss_max, max(data["train_losses"]))
        if data["eval_losses"]:
            loss_min = min(loss_min, min(data["eval_losses"]))
            loss_max = max(loss_max, max(data["eval_losses"]))

    # Handle case if no data was found
    if loss_min == float("inf") or loss_max == float("-inf"):
        print("No valid loss data found in log files")
        return False

    # Second pass to create plots with consistent scales
    for stage_idx, data in enumerate(all_data):
        # Get the right axis objects
        if len(log_files) == 1:
            ax1, ax2 = axes  # When only one stage
        else:
            ax1, ax2 = axes[stage_idx]

        # For multiple stages, add stage title above each pair of plots
        if len(log_files) > 1:
            # Add stage title above the pair of plots - adjusted positioning
            title_y_position = 1 - (stage_idx / len(log_files)) - 0.02
            fig.text(
                0.5,
                title_y_position,
                stage_names[stage_idx],
                fontsize=14,
                ha="center",
                weight="bold",
            )

        # Plot loss curves - use simplified title for individual plots
        simplified_name = "Loss Curves"
        plot_loss_curves(ax1, data, simplified_name)

        # Plot learning rate
        plot_learning_rate(ax2, data)

        # Set consistent y-axis for loss plots
        ax1.set_ylim([max(0, loss_min * 0.9), loss_max * 1.1])

    # Adjust layout with more top space
    plt.tight_layout()
    if len(log_files) == 1:
        fig.subplots_adjust(top=0.85, wspace=0.4, hspace=0.4)
    else:
        fig.subplots_adjust(top=0.95, wspace=0.4, hspace=0.5)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()
    return True


if __name__ == "__main__":
    # Change working directory to script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

    # Set default paths relative to script directory
    default_logging_dir = os.path.join(script_dir, path_logging)
    default_output_path = os.path.join(
        script_dir, output_path, "fine_tuning_stages.png"
    )

    parser = argparse.ArgumentParser(
        description="Generate multi-stage loss curves from training logs"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=default_logging_dir,
        help=f"Directory containing logging files (default: {path_logging})",
    )
    parser.add_argument(
        "--output", type=str, default=default_output_path, help="Output image path"
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", help="Custom names for each fine-tuning stage"
    )

    args = parser.parse_args()

    # Find and sort log files in the specified directory
    log_files = find_logging_files(args.dir)

    if log_files:
        # Process files in batches based on cut_entries
        total_files = len(log_files)
        batch_count = 0

        for i in range(0, total_files, cut_entries):
            batch_files = log_files[i : i + cut_entries]
            batch_count += 1

            # Generate output filename for this batch
            if total_files <= cut_entries:
                # Single batch - use original filename
                batch_output_path = args.output
            else:
                # Multiple batches - add batch number to filename
                base_name = os.path.splitext(args.output)[0]
                extension = os.path.splitext(args.output)[1]
                batch_output_path = f"{base_name}_batch_{batch_count}{extension}"

            print(f"\nProcessing batch {batch_count}: {len(batch_files)} files")
            print(f"Files: {[os.path.basename(f) for f in batch_files]}")

            # Create the visualization for this batch
            create_multi_stage_loss_curve(
                batch_files, batch_output_path, None, args.dir
            )
    else:
        print("No log files found to process.")
