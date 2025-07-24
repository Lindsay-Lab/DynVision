"""Plot classifier unit responses with focus on top active units."""

import argparse
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils import load_df


def get_label_sequence_info(
    df: pd.DataFrame, label_set: str
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract label sequence information including changepoints and label sequence.

    Args:
        df: DataFrame containing classifier responses
        label_set: String identifier for the specific label sequence

    Returns:
        Tuple containing:
        - Array of timepoints where labels change
        - List of labels in sequence order
    """
    # Filter DataFrame for the specific label set
    label_set_df = df[df.label_set == label_set]
    sample_index = label_set_df.sample_index.unique()[0]
    class_index = label_set_df.class_index.unique()[0]
    label_set_df = label_set_df[
        (label_set_df.sample_index == sample_index)
        & (label_set_df.class_index == class_index)
    ]
    label_indices = label_set_df.label_index.values

    # Find points where label changes
    changepoints = np.where(np.diff(label_indices) != 0)[0]

    # Get the sequence of labels at change points (including the first label)
    sequence = [label_indices[0]]  # Start with first label
    for point in changepoints:
        sequence.append(label_indices[point + 1])  # Add label after each change

    return changepoints, sequence


def get_top_units(df: pd.DataFrame, n_units: int = 10) -> List[int]:
    """
    Identify the top N most active classifier units based on mean response.

    Args:
        df: DataFrame containing classifier responses
        n_units: Number of top units to select

    Returns:
        List of class indices for top N units
    """
    # Calculate mean response per class across all samples and timepoints
    mean_responses = df.groupby("class_index")["response"].mean()
    top_units = mean_responses.nlargest(n_units).index.tolist()
    return top_units


def calculate_accuracy(df: pd.DataFrame) -> float:
    """
    Calculate classification accuracy for valid labels.

    Args:
        df: DataFrame containing classifier responses

    Returns:
        Classification accuracy as float
    """
    valid_samples = df[df.label_index > -1]
    accuracy = (valid_samples.guess_index == valid_samples.label_index).mean()
    return accuracy


def plot_classifier_responses(
    df: pd.DataFrame, n_units: int = 10, figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot response traces for top N most active classifier units.

    Args:
        df: DataFrame containing classifier responses
        n_units: Number of top units to plot
        figsize: Figure dimensions as (width, height)

    Returns:
        Matplotlib figure and axes objects
    """
    # Set Seaborn style
    sns.set_context("talk")
    sns.set_style("ticks")

    # Get unique label set and top units
    label_set = df.label_set.unique()[0]
    top_units = get_top_units(df, n_units)

    # Filter data for top units
    plot_df = df[df.label_set == label_set]
    plot_df = plot_df[plot_df.class_index.isin(top_units)].copy()

    # Create figure and prepare colors
    fig, ax = plt.subplots(figsize=figsize)

    # Create a color palette with exactly n_units distinct colors
    colors = sns.color_palette("husl", n_colors=n_units)
    color_dict = dict(zip(top_units, colors))

    # Plot response traces with explicit hue order
    sns.lineplot(
        data=plot_df,
        x="times_index",
        y="response",
        hue="class_index",
        hue_order=top_units,  # Ensure specific order of units
        errorbar=("ci", 75),
        palette=color_dict,  # Use our explicit color mapping
        ax=ax,
        marker=".",
        legend="full",  # Ensure full legend is shown
    )

    # Get label sequence information
    changepoints, label_sequence = get_label_sequence_info(plot_df, label_set)

    # Add vertical lines at changepoints
    for changepoint in changepoints:
        ax.axvline(
            changepoint,
            color="0.6",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="_nolegend_",
        )

    # Customize plot
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Classifier Response")
    # Create informative title
    label_seq = "→".join(map(str, label_sequence))
    ax.set_title(f"Top {n_units} Most Active Units\n" f"Label Sequence: {label_seq}")

    # Add accuracy text
    accuracy = calculate_accuracy(plot_df)
    ax.text(
        0.05,
        0.95,
        f"Accuracy: {accuracy:.2f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Customize legend
    ax.legend(
        title="Class Index", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False
    )

    # Final styling
    sns.despine()
    plt.tight_layout()

    return fig, ax


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Plot classifier unit responses.")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--n_units", type=int, default=10, help="Number of top units to plot"
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load and process data
    df = load_df(args.input)
    data_identifier = args.input.stem

    # Generate plots for each unique label sequence
    for label_set in df.label_set.unique():
        label_df = df[df.label_set == label_set]
        fig, _ = plot_classifier_responses(label_df, n_units=args.n_units)

        # Save plot
        output_path = args.output / f"{data_identifier}_label{label_set}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
