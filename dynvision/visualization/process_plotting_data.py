import argparse
import re
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from dynvision.utils import replace_param_in_string
from dynvision.utils.visualization_utils import (
    save_plot,
    load_responses,
    load_responses_in_batches,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--responses", nargs="+", type=Path, required=True, help="Path to pt files"
)
parser.add_argument(
    "--test_outputs", nargs="+", type=Path, required=True, help="Path to csv files"
)
parser.add_argument("--output", type=Path, required=True, help="Path to directory")
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
parser.add_argument(
    "--measures", nargs="+", type=str, required=True, help="Measure to plot"
)
parser.add_argument("--category", type=str, required=True, help="Category to plot")
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for processing files"
)

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # Handle interval parameter measure adjustment
    if "interval" in args.parameter:
        args.measures = [
            measure.replace("peak_time", "peak_ratio") for measure in args.measures
        ]

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load and process data in batches
    df, layer_names = load_responses_in_batches(
        args.responses,
        args.test_outputs,
        data_arg_key=args.parameter,
        measures=args.measures,
        category=args.category,
        batch_size=args.batch_size,
    )

    # Save processed data
    print(f"Saving processed data to: {args.output}")
    df.to_csv(args.output, index=False)
    print(f"Processed data saved with {len(df)} total rows")
