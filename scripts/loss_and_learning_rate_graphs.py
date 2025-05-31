import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

# vertical or horizontal layout for the two diagrams
diagrams_vertical = True
diagrams_merged = False
high_reability = True


def create_loss_curve(log_file="logging.txt", output_path="fine_tuning_loss_curve.png"):
    """Generate side-by-side loss and learning rate curves from training logs."""

    # Lists to store extracted data
    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []
    lr_epochs = []
    learning_rates = []

    # Regular expressions to extract the data
    train_pattern = re.compile(r"{'loss': ([\d\.]+).*?'epoch': ([\d\.]+)}")
    eval_pattern = re.compile(r"{'eval_loss': ([\d\.]+).*?'epoch': ([\d\.]+)}")
    lr_pattern = re.compile(r"'learning_rate': ([\d\.eE\-\+]+).*?'epoch': ([\d\.]+)")

    try:
        # Read and parse the log file
        with open(log_file, "r") as file:
            content = file.read()

            # Extract training data
            for match in train_pattern.finditer(content):
                loss = float(match.group(1))
                epoch = float(match.group(2))
                train_epochs.append(epoch)
                train_losses.append(loss)

            # Extract evaluation data
            for match in eval_pattern.finditer(content):
                loss = float(match.group(1))
                epoch = float(match.group(2))
                eval_epochs.append(epoch)
                eval_losses.append(loss)

            # Extract learning rate data
            for match in lr_pattern.finditer(content):
                lr = float(match.group(1))
                epoch = float(match.group(2))
                lr_epochs.append(epoch)
                learning_rates.append(lr)

        # Check if we found any data
        if not train_losses and not eval_losses and not learning_rates:
            print(f"No data found in {log_file}")
            return False

        # Set font sizes based on high_reability setting
        if high_reability:
            title_size = 24
            subtitle_size = 20
            label_size = 18
            legend_size = 16
            tick_size = 14
            annotation_size = 14
            suptitle_size = 26
            font_weight = "bold"
        else:
            title_size = 14
            subtitle_size = 12
            label_size = 12
            legend_size = 10
            tick_size = 10
            annotation_size = 8
            suptitle_size = 16
            font_weight = "normal"

        # Create figure based on layout preferences
        if diagrams_merged:
            # Single plot with dual y-axes
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(
                "Fine-Tuning Metrics (Combined)",
                fontsize=suptitle_size,
                fontweight=font_weight,
            )

            # LOSS CURVES on primary y-axis
            ax1.set_xlabel("Epoch", fontsize=label_size, fontweight=font_weight)
            ax1.set_ylabel(
                "Loss", fontsize=label_size, color="b", fontweight=font_weight
            )
            ax1.set_title(
                "Training Loss, Evaluation Loss & Learning Rate",
                fontsize=title_size,
                fontweight=font_weight,
            )
            ax1.tick_params(axis="both", labelsize=tick_size)

            # Plot training loss
            if train_losses:
                ax1.plot(
                    train_epochs,
                    train_losses,
                    "bo",
                    alpha=0.3,
                    markersize=2,
                    label="Training Loss",
                )

                # Add moving average for training loss
                window_size = min(10, max(3, len(train_losses) // 10))
                if len(train_losses) > window_size:
                    moving_avg = np.convolve(
                        train_losses, np.ones(window_size) / window_size, mode="valid"
                    )
                    moving_avg_epochs = train_epochs[window_size - 1 :]
                    ax1.plot(
                        moving_avg_epochs,
                        moving_avg,
                        "b-",
                        label=f"Training Loss (Moving Avg {window_size})",
                        linewidth=2,
                    )

            # Plot evaluation loss
            if eval_losses:
                ax1.plot(
                    eval_epochs,
                    eval_losses,
                    "ro",
                    markersize=6,
                    label="Evaluation Loss",
                )

            ax1.tick_params(axis="y", labelcolor="b", labelsize=tick_size)
            ax1.grid(True, linestyle="--", alpha=0.7)

            # LEARNING RATE on secondary y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel(
                "Learning Rate", fontsize=label_size, color="g", fontweight=font_weight
            )
            ax2.tick_params(axis="y", labelcolor="g", labelsize=tick_size)

            if learning_rates:
                ax2.plot(
                    lr_epochs,
                    learning_rates,
                    "go",
                    alpha=0.5,
                    markersize=4,
                    label="Learning Rate",
                )

                # Add moving average for learning rate
                window_size = min(10, max(3, len(learning_rates) // 10))
                if len(learning_rates) > window_size:
                    lr_moving_avg = np.convolve(
                        learning_rates, np.ones(window_size) / window_size, mode="valid"
                    )
                    lr_moving_avg_epochs = lr_epochs[window_size - 1 :]
                    ax2.plot(
                        lr_moving_avg_epochs,
                        lr_moving_avg,
                        "g-",
                        label=f"Learning Rate (Moving Avg {window_size})",
                        linewidth=2,
                    )

                # Log scale for learning rate if all values are positive
                if min(learning_rates) > 0:
                    ax2.set_yscale("log")

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper right",
                fontsize=legend_size,
            )

        else:
            # Separate subplots
            if diagrams_vertical:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            else:  # horizontal
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(
                "Fine-Tuning Metrics", fontsize=suptitle_size, fontweight=font_weight
            )

            # FIRST SUBPLOT - LOSS CURVES
            ax1.set_xlabel("Epoch", fontsize=label_size, fontweight=font_weight)
            ax1.set_ylabel("Loss", fontsize=label_size, fontweight=font_weight)
            ax1.set_title(
                "Training and Evaluation Loss",
                fontsize=title_size,
                fontweight=font_weight,
            )
            ax1.tick_params(axis="both", labelsize=tick_size)

            # Plot training loss points as actual values
            if train_losses:
                ax1.plot(
                    train_epochs,
                    train_losses,
                    "bo",
                    alpha=0.3,
                    markersize=2,
                    label="Training Loss",
                )

                # Add moving average for training loss
                window_size = min(10, max(3, len(train_losses) // 10))
                if len(train_losses) > window_size:
                    moving_avg = np.convolve(
                        train_losses, np.ones(window_size) / window_size, mode="valid"
                    )
                    moving_avg_epochs = train_epochs[window_size - 1 :]
                    ax1.plot(
                        moving_avg_epochs,
                        moving_avg,
                        "b-",
                        label=f"Training Loss (Moving Avg {window_size})",
                        linewidth=2,
                    )

            # Plot evaluation loss
            if eval_losses:
                ax1.plot(
                    eval_epochs,
                    eval_losses,
                    "ro",
                    markersize=6,
                    label="Evaluation Loss",
                )

                # Add annotations for eval loss values
                for i, txt in enumerate(eval_losses):
                    ax1.annotate(
                        f"{txt:.4f}",
                        (eval_epochs[i], eval_losses[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=annotation_size,
                        fontweight=font_weight,
                    )

            # Calculate loss reductions for context
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

            # Add grid and legend to loss plot
            ax1.grid(True, linestyle="--", alpha=0.7)
            ax1.legend(
                title=f"Final Losses: {loss_info}",
                fontsize=legend_size,
                title_fontsize=legend_size,
            )

            # SECOND SUBPLOT - LEARNING RATE
            ax2.set_xlabel("Epoch", fontsize=label_size, fontweight=font_weight)
            ax2.set_ylabel("Learning Rate", fontsize=label_size, fontweight=font_weight)
            ax2.set_title(
                "Learning Rate Schedule", fontsize=title_size, fontweight=font_weight
            )
            ax2.tick_params(axis="both", labelsize=tick_size)

            # Plot learning rate
            if learning_rates:
                ax2.plot(
                    lr_epochs,
                    learning_rates,
                    "go",
                    alpha=0.5,
                    markersize=4,
                    label="Learning Rate",
                )

                # Add moving average for learning rate
                window_size = min(10, max(3, len(learning_rates) // 10))
                if len(learning_rates) > window_size:
                    lr_moving_avg = np.convolve(
                        learning_rates, np.ones(window_size) / window_size, mode="valid"
                    )
                    lr_moving_avg_epochs = lr_epochs[window_size - 1 :]
                    ax2.plot(
                        lr_moving_avg_epochs,
                        lr_moving_avg,
                        "g-",
                        label=f"Learning Rate (Moving Avg {window_size})",
                        linewidth=2,
                    )

                # Log scale for learning rate if all values are positive
                if min(learning_rates) > 0:
                    ax2.set_yscale("log")

                # Add final learning rate to legend
                lr_info = f"Final LR: {learning_rates[-1]:.2e}"
                ax2.legend(
                    title=lr_info, fontsize=legend_size, title_fontsize=legend_size
                )

            # Add grid to learning rate plot
            ax2.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout and save
        plt.tight_layout()
        if not diagrams_merged:
            if diagrams_vertical:
                fig.subplots_adjust(hspace=0.42)
            else:  # horizontal
                fig.subplots_adjust(wspace=0.4)

        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

        # Display the plot
        plt.show()
        return True

    except FileNotFoundError:
        print(f"Error: Log file {log_file} not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Change working directory to script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

    parser = argparse.ArgumentParser(
        description="Generate loss and learning rate curves from training logs"
    )
    parser.add_argument(
        "--log", type=str, default="./logging.txt", help="Path to the log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fine_tuning_curves.png",
        help="Output image path",
    )

    args = parser.parse_args()
    create_loss_curve(args.log, args.output)
