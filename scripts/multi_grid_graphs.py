import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


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
            for match in eval_pattern.finditer(content):
                data["eval_losses"].append(float(match.group(1)))
                data["eval_epochs"].append(float(match.group(2)))

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
        loss_info += f"Train: {last_loss:.4f} (-{train_reduction:.1f}%)"

    if eval_losses:
        if loss_info:
            loss_info += ", "
        first_eval = eval_losses[0]
        last_eval = eval_losses[-1]
        eval_reduction = 100 * (1 - last_eval / first_eval)
        loss_info += f"Eval: {last_eval:.4f} (-{eval_reduction:.1f}%)"

    return loss_info


def plot_loss_curves(ax, data, stage_name):
    """Plot loss curves on the given axis."""
    train_epochs = data["train_epochs"]
    train_losses = data["train_losses"]
    eval_epochs = data["eval_epochs"]
    eval_losses = data["eval_losses"]

    # Set axis labels and title
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{stage_name}: Loss Curves", fontsize=13)

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

        # Add annotations for eval loss values
        for i, val in enumerate(eval_losses):
            ax.annotate(
                f"{val:.4f}",
                (eval_epochs[i], eval_losses[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    loss_info = get_loss_info(train_losses, eval_losses)
    ax.legend(title=f"Final: {loss_info}")


def plot_learning_rate(ax, data):
    """Plot learning rate on the given axis."""
    lr_epochs = data["lr_epochs"]
    learning_rates = data["learning_rates"]

    # Set axis labels and title
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=13)

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


def create_multi_stage_loss_curve(
    log_files, output_path="fine_tuning_stages.png", stage_names=None
):
    """Generate loss and learning rate curves for multiple fine-tuning stages."""

    if not isinstance(log_files, list):
        log_files = [log_files]

    if stage_names is None:
        stage_names = [
            f"Stage {i+1}: {os.path.basename(f)}" for i, f in enumerate(log_files)
        ]

    # Create a figure with grid of subplots (stages × metrics)
    fig, axes = plt.subplots(len(log_files), 2, figsize=(15, 5 * len(log_files)))
    fig.suptitle("Fine-Tuning Stages Metrics", fontsize=16)

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

        # Plot loss curves
        plot_loss_curves(ax1, data, stage_names[stage_idx])

        # Plot learning rate
        plot_learning_rate(ax2, data)

        # Set consistent y-axis for loss plots
        ax1.set_ylim([max(0, loss_min * 0.9), loss_max * 1.1])

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(
        top=0.95, wspace=0.3
    )  # Make room for suptitle and add space between plots

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

    parser = argparse.ArgumentParser(
        description="Generate multi-stage loss curves from training logs"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./logging",
        help="Directory containing logging files (default: ./logging)",
    )
    parser.add_argument(
        "--output", type=str, default="fine_tuning_stages.png", help="Output image path"
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", help="Custom names for each fine-tuning stage"
    )

    args = parser.parse_args()

    # Find and sort log files in the specified directory
    log_files = find_logging_files(args.dir)

    if log_files:
        # Create the visualization
        create_multi_stage_loss_curve(log_files, args.output, args.stages)
    else:
        print("No log files found to process.")
