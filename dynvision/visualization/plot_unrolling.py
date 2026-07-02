import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_color,
    get_display_name,
    order_layers,
    tensor_to_numpy,
    layer_response_avg,
)

# Global styling parameters (matching plot_responses.py)
FORMATTING = {
    "fontsize_title": 20,
    "fontsize_axis": 19,
    "fontsize_tick": 16,
    "fontsize_legend": 19,
    "fontsize_label": 18,
    "linewidth_main": 3,
    "linewidth_indicator": 3,
    "alpha_line": 0.8,
    "alpha_indicator": 0.6,
    "layer_circle_colors": {
        "V1": "#ff69b4ff",
        "V2": "#dda0ddff",
        "V4": "#da70d6ff",
        "IT": "#ba55d3ff",
    },
}
FIGURE_HEIGHT_PER_SUBPLOT = 2.5
SUBPLOT_SPACING = 0.2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--engineering_time_data",
    type=Path,
    required=True,
    help="Path to engineering time response .pt file",
)
parser.add_argument(
    "--biological_time_data",
    type=Path,
    required=True,
    help="Path to biological time response .pt file",
)
parser.add_argument(
    "--t_feedforward",
    type=int,
    required=True,
    help="Feedforward delay per layer in timesteps (already converted from ms)",
)
parser.add_argument(
    "--dt",
    type=float,
    default=1.0,
    help="Temporal resolution in ms per timestep (for axis labels)",
)
parser.add_argument(
    "--idle-timesteps",
    type=int,
    default=0,
    help="Number of idle timesteps before recorded data (shifts time axis)",
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output plot file"
)
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")


def load_and_process_responses(response_file):
    """Load response tensor file and calculate average responses for each layer.

    Handles both old (unit-wise) and new (layer-wise pre-averaged) response formats.

    Args:
        response_file: Path to .pt file containing layer responses

    Returns:
        dict: Dictionary mapping layer_name -> averaged response array (samples x time)
    """
    print(f"Loading responses from: {response_file}")
    responses = torch.load(response_file, map_location=torch.device("cpu"))

    # Extract metadata if present (new format indicator)
    metadata = responses.pop("_metadata", None)
    if metadata is not None:
        response_resolution = metadata.get("response_resolution", "unit")
        print(f"Response format: {response_resolution} (metadata: {metadata})")
    else:
        # Auto-detect: if non-classifier keys end with _response_avg/_response_std, it's layer-wise
        non_classifier_keys = [k for k in responses if k != "classifier"]
        has_avg_keys = any(k.endswith("_response_avg") for k in non_classifier_keys)
        response_resolution = "layer" if has_avg_keys else "unit"
        if response_resolution == "layer":
            print("Auto-detected layer-wise response format (no metadata)")

    # Process each layer to get average responses
    processed_responses = {}

    if response_resolution == "layer":
        # New format: tensors are pre-averaged (already 2D or 3D, not spatial)
        for key, tensor in responses.items():
            if key.endswith("_response_avg"):
                # Extract layer name from key (e.g., "layer0_response_avg" -> "layer0")
                layer_name = key.replace("_response_avg", "")
                # Already averaged, just convert to numpy
                processed_responses[layer_name] = tensor_to_numpy(tensor)
            elif key == "classifier":
                # Handle classifier directly (no _response_avg suffix for classifier)
                processed_responses["classifier"] = tensor_to_numpy(tensor)
            # Skip std tensors and metadata keys
        print(f"Loaded {len(processed_responses)} layers from layer-wise format")
        print(f"Layer names: {list(processed_responses.keys())}")
    else:
        # Old format: tensors have spatial dimensions, need averaging
        for layer_name, tensor in responses.items():
            if isinstance(tensor, torch.Tensor):
                # Calculate layer response average across spatial/feature dimensions
                avg_response = layer_response_avg(tensor)  # Shape: (samples, time)
                processed_responses[layer_name] = tensor_to_numpy(avg_response)
        print(f"Loaded {len(processed_responses)} layers from unit-wise format")

    return processed_responses


def pad_responses_to_match(engineering_responses, biological_responses):
    """Pad response data to ensure both datasets have the same time dimensions.

    Args:
        engineering_responses: Dict of layer_name -> response arrays
        biological_responses: Dict of layer_name -> response arrays

    Returns:
        Tuple of (padded_engineering_responses, padded_biological_responses)
    """
    print("Padding responses to match time dimensions...")

    padded_eng = {}
    padded_bio = {}

    # Get common layers
    common_layers = set(engineering_responses.keys()) & set(
        biological_responses.keys()
    )

    for layer_name in common_layers:
        eng_data = engineering_responses[layer_name]
        bio_data = biological_responses[layer_name]

        # Get maximum timesteps for this layer
        eng_timesteps = eng_data.shape[1]
        bio_timesteps = bio_data.shape[1]
        max_timesteps = max(eng_timesteps, bio_timesteps)

        # Pad engineering data if needed
        if eng_timesteps < max_timesteps:
            pad_len = max_timesteps - eng_timesteps
            padded_eng_data = np.zeros((eng_data.shape[0], max_timesteps))
            padded_eng_data[:, pad_len:] = eng_data
            padded_eng[layer_name] = padded_eng_data
            print(
                f"  Padded {layer_name} engineering data: {eng_timesteps} -> {max_timesteps}"
            )
        else:
            padded_eng[layer_name] = eng_data

        # Pad biological data if needed
        if bio_timesteps < max_timesteps:
            pad_len = max_timesteps - bio_timesteps
            padded_bio_data = np.zeros((bio_data.shape[0], max_timesteps))
            padded_bio_data[:, pad_len:] = bio_data
            padded_bio[layer_name] = padded_bio_data
            print(
                f"  Padded {layer_name} biological data: {bio_timesteps} -> {max_timesteps}"
            )
        else:
            padded_bio[layer_name] = bio_data

    return padded_eng, padded_bio


def create_time_shifted_trace(response_data, shift_amount):
    """Create time-shifted version of response data.

    Args:
        response_data: Array of shape (samples, time)
        shift_amount: Number of time steps to shift forward

    Returns:
        Array with time-shifted data, padded with zeros at the beginning
    """
    if shift_amount <= 0:
        return response_data

    # Pad at the beginning with zeros
    n_samples, n_time = response_data.shape
    shifted_data = np.zeros((n_samples, n_time))

    if shift_amount < n_time:
        shifted_data[:, shift_amount:] = response_data[:, :-shift_amount]

    return shifted_data


def _add_layer_circle(x, y, layer_name, ax=None, config=None):
    """Add a circular layer indicator label to the axes.

    Replicates the style from plot_responses.py for visual consistency.
    """
    if ax is None:
        ax = plt.gca()

    pad = 0.5 if layer_name == "IT" else 0.4

    color = get_color(layer_name, config) if config else None
    if color is None:
        color = FORMATTING["layer_circle_colors"].get(layer_name, "#808080ff")

    ax.text(
        x,
        y,
        layer_name,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle=f"circle,pad={pad}",
            facecolor=color,
            edgecolor="#353535ff",
            linewidth=2,
            alpha=0.8,
        ),
        fontsize=FORMATTING["fontsize_label"],
        fontweight="bold",
    )


def plot_unrolling(
    engineering_responses,
    biological_responses,
    t_feedforward,
    dt=1.0,
    time_offset=0.0,
    config=None,
    output_path=None,
):
    """Create unrolling plot comparing biological vs engineering time responses.

    Args:
        engineering_responses: Dict of layer_name -> response arrays from engineering time
        biological_responses: Dict of layer_name -> response arrays from biological time
        t_feedforward: Time shift amount per layer in timesteps
        dt: Temporal resolution in ms per timestep (for axis labels)
        time_offset: Time offset in ms to shift the time axis (e.g., from idle timesteps)
        config: Configuration dictionary for styling
        output_path: Path to save the plot

    Returns:
        matplotlib Figure object
    """

    # Pad responses to ensure matching time dimensions
    engineering_responses, biological_responses = pad_responses_to_match(
        engineering_responses, biological_responses
    )

    # Get layer names and order them
    layer_names = list(engineering_responses.keys())
    ordered_layers = order_layers(layer_names, config)

    print(f"Plotting layers in order: {ordered_layers}")

    n_layers = len(ordered_layers)
    fmt = FORMATTING

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(
        n_layers, 1, figsize=(10, FIGURE_HEIGHT_PER_SUBPLOT * n_layers), sharex=True
    )

    # Ensure axes is always a list
    if n_layers == 1:
        axes = [axes]

    # Define colors and markers for the three traces
    bio_color = "#1b780eff"  # Green for biological
    eng_color = "#d3a60cff"  # Orange/yellow for engineering
    shift_color = "#b3860cff"  # Darker shade of engineering color

    # Plot each layer
    for i, layer in enumerate(ordered_layers):
        ax = axes[i]

        if layer not in engineering_responses or layer not in biological_responses:
            print(f"Warning: {layer} not found in response data")
            continue

        eng_data = engineering_responses[layer]
        bio_data = biological_responses[layer]

        # Calculate time shift for this layer (bottom-up, V2 gets 1x shift, V4 gets 2x, etc.)
        # Find layer index from bottom (V1=0, V2=1, V4=2, etc.)
        layer_hierarchy = ["V1", "V2", "V4", "IT", "classifier"]
        if layer in layer_hierarchy:
            layer_idx = layer_hierarchy.index(layer)
        else:
            layer_idx = i  # fallback to subplot index

        shift_amount = layer_idx * t_feedforward

        # Create time-shifted engineering data
        shifted_eng_data = create_time_shifted_trace(eng_data, shift_amount)

        # Calculate mean across samples for plotting
        eng_mean = np.mean(eng_data, axis=0)
        bio_mean = np.mean(bio_data, axis=0)
        shifted_eng_mean = np.mean(shifted_eng_data, axis=0)

        # Create time axis in ms
        time_steps = np.arange(len(eng_mean)) * dt + time_offset

        # Plot the three traces with different markers
        ax.plot(
            time_steps,
            bio_mean,
            color=bio_color,
            linewidth=fmt["linewidth_main"],
            alpha=fmt["alpha_line"],
            label="Biological time",
            linestyle="-",
        )

        ax.plot(
            time_steps,
            eng_mean,
            color=eng_color,
            linewidth=fmt["linewidth_main"],
            alpha=fmt["alpha_line"],
            label="Engineering time",
            linestyle="-",
        )

        ax.plot(
            time_steps,
            shifted_eng_mean,
            color=shift_color,
            linewidth=fmt["linewidth_main"] + 1,
            alpha=fmt["alpha_line"],
            label="Engineering time shifted",
            linestyle="none",
            marker="o",
            markersize=8,
            markevery=3,
        )

        # Remove y-axis text label (replaced by layer circle)
        ax.set_ylabel("")

        if i == n_layers - 1:
            ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
            ax.tick_params(labelsize=fmt["fontsize_tick"])
        else:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("")

        ax.tick_params(axis="y", labelsize=fmt["fontsize_tick"])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(time_steps), max(time_steps))

        # Remove top and right spines
        sns.despine(ax=ax, left=True, bottom=True)

    # Add label indicator to the bottom subplot
    ax_bottom = axes[-1]
    y_min, y_max = ax_bottom.get_ylim()
    stimulus_duration = 20  # labels present for first 20 timesteps
    label_indicator = np.full(len(time_steps), y_min)
    label_indicator[:stimulus_duration] = y_min + 0.1 * (y_max - y_min)
    ax_bottom.plot(
        time_steps,
        label_indicator,
        color="dimgray",
        linewidth=fmt["linewidth_indicator"],
        drawstyle="steps-mid",
        alpha=fmt["alpha_indicator"],
    )

    # Add legend to the bottom subplot with white background, no border
    axes[-1].legend(
        loc="center left",
        fontsize=fmt["fontsize_legend"],
        frameon=False,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )

    # Add layer circles after y-limits are finalized (drawn after legend so they appear on top)
    for i, layer in enumerate(ordered_layers):
        display_name = get_display_name(layer, config)
        _add_layer_circle(
            x=0.95,
            y=0.5,
            layer_name=display_name,
            ax=axes[i],
            config=config,
        )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=SUBPLOT_SPACING)

    # Add shared y-label centered vertically
    fig.text(
        0.01,
        0.5,
        "Average Response",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=fmt["fontsize_axis"],
    )

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    # Compute time offset from idle timesteps
    time_offset = args.idle_timesteps * args.dt

    # Load configuration from command line arguments
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
    )

    # Load and process response data
    engineering_responses = load_and_process_responses(args.engineering_time_data)
    biological_responses = load_and_process_responses(args.biological_time_data)

    # Check if data is empty
    if len(engineering_responses) == 0 or len(biological_responses) == 0:
        print("Warning: No data found in the input files")
        # Create an empty figure
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        # Generate the unrolling plot
        fig = plot_unrolling(
            engineering_responses,
            biological_responses,
            args.t_feedforward,
            dt=args.dt,
            time_offset=time_offset,
            config=config,
            output_path=args.output,
        )

    # Save the plot
    save_plot(args.output)

    print(f"Unrolling plot saved to: {args.output}")
