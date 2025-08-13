import argparse
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dynvision.utils import replace_param_in_string
from dynvision.utils.visualization_utils import save_plot

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True, help="Path to pt files")
parser.add_argument("--output", type=Path, required=True, help="Path to directory")
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
parser.add_argument(
    "--measures", nargs="+", type=str, required=True, help="Measure to plot"
)
parser.add_argument("--category", type=str, required=True, help="Category to plot")


def plot_adaption(
    df,
    data_arg_key="contrast",
    measures=["power", "peak_height", "peak_time"],
    layer_name="layer1",
    label_target=None,
    category="rctype",
):
    if label_target is None:
        plot_df = df
    else:
        plot_df = df[(df.label_set.str.contains(str(label_target)))]

    key_values = plot_df[data_arg_key].unique()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 10))
    ax_left.set_axis_off()
    ax_right.set_axis_off()

    for i, key_value in enumerate(key_values):
        ax = fig.add_subplot(len(key_values), 2, 2 * i + 1, sharey=ax_left)
        sns.lineplot(
            data=plot_df[plot_df[data_arg_key] == key_value],
            x="times_index",
            y=f"{layer_name}_power",
            ax=ax,
            marker=".",
            palette="tab10",
            hue=category,
        )
        if not i:
            ax.set_title(data_arg_key)
        ax.text(
            0.95,
            0.95,
            key_value,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        if i == len(key_values) // 2:
            ax.set_ylabel("power")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        # ax.set_yticklabels([])
        if i < len(key_values) - 1:
            ax.set_xticklabels([])

        ax.legend().remove()
        ax.xaxis.set_tick_params(which="both", labelbottom=True)
        sns.despine(ax=ax, left=True, bottom=True)

    ax.set_xlabel("times index")

    for i, measure in enumerate(measures):

        ax = fig.add_subplot(len(measures), 2, 2 * i + 2)
        sns.lineplot(
            data=plot_df,
            x=data_arg_key,
            y=f"{layer_name}_{measure}",
            ax=ax,
            marker=".",
            palette="tab10",
            hue=category,
        )
        ax.set_xlabel("")
        ax.set_ylabel(f"{measure.replace('_', ' ')}")
        sns.despine(ax=ax, left=True)
        if i:
            ax.legend().remove()
        else:
            ax.legend(frameon=False, title=category)

    ax.set_xlabel(data_arg_key.replace("_", " "))
    fig.suptitle(layer_name)
    return ax


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    print(f"Loading processed data from: {args.data}")
    df = pd.read_csv(args.data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sns.set_context("talk")

    for layer_name in tqdm(layer_names):
        plot_adaption(
            df,
            data_arg_key=args.parameter,
            measures=args.measures,
            layer_name=layer_name,
            category=args.category,
        )
        save_plot(args.output.parent / f"{layer_name}.png")

    args.output.touch(exist_ok=True)
