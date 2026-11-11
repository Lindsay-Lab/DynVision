#!/usr/bin/env python
"""Remove test results whose stored samples differ from the rest of the experiment.

A test evaluation stores layer responses for the samples it saw, in batches of
``test_batch_size``, capped at ``store_test_responses`` batches. When that
setting, the evaluated dataset or the code changed between runs, the results of
one experiment end up resting on different samples: some runs store responses
for 384 samples, others for 256, 512 or only 32, and the evaluated image sets
differ too (1280 against 2560 images). Averaging such runs together mixes
different images, and a paired comparison is impossible.

Within one experiment, all runs must therefore rest on the same number of
responses, per layer and in total, and that number must be the largest the
experiment has. This script finds the runs that violate that, and with
``--apply`` deletes their stage-1 results (``test_data.csv``,
``test_outputs.csv``, ``test_responses.pt``) together with the stage-2
aggregate of their model directory, so Snakemake reruns those tests and
re-aggregates the experiment.

The target selection of an experiment is the most common one among its runs
that stored responses (ties go to the larger selection). Runs that stored no
responses at all -- noise experiments that only measure accuracy and
confidence -- are reported separately and left alone unless ``--include-empty``
is given.

Usage::

    python scripts/prune_test_results.py ROOT                 # report only
    python scripts/prune_test_results.py ROOT --apply         # delete and rerun
    python scripts/prune_test_results.py ROOT --apply --include-empty
    python scripts/prune_test_results.py ROOT --experiment interval \
        --model 'DyRCNNx8:*tsteps=20*' --seed 7000 --seed 7001

``ROOT`` is the project working directory holding ``reports/``. The evaluation
that produced a deleted result is rerun by Snakemake; nothing is retrained.

``--experiment``, ``--model`` and ``--seed`` restrict the scan to matching
directories (shell-style patterns, each repeatable). An experiment's target
selection is then taken from the scanned runs only, so a narrow scope can pick
a different target than the whole experiment would.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Run from a checkout without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dynvision.processing.sample_selection import (  # noqa: E402
    RESPONSE_SUFFIX,
    SampleSelection,
    sample_selection,
)

TEST_DATA = "test_data.csv"
STAGE_1_FILES = ("test_data.csv", "test_outputs.csv", "test_responses.pt")


@dataclass(frozen=True)
class Deviation:
    """One test result that does not match its experiment's selection."""

    experiment: str
    test_file: Path
    selection: Optional[SampleSelection]

    @property
    def test_dir(self) -> Path:
        return self.test_file.parent

    @property
    def model_dir(self) -> Path:
        """The ``{data_name}:{data_group}_{status}`` directory."""
        return self.test_file.parents[1]

    @property
    def aggregate(self) -> Path:
        """Stage-2 aggregate of the model directory, regenerated after a rerun."""
        return self.model_dir / TEST_DATA


def _subdirs(path: Path) -> List[Path]:
    with os.scandir(path) as entries:
        return sorted(Path(e.path) for e in entries if e.is_dir())


def _matches(name: str, patterns: Optional[Sequence[str]]) -> bool:
    return not patterns or any(fnmatchcase(name, p) for p in patterns)


def find_test_files(
    reports: Path,
    experiments: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[str]] = None,
    progress: bool = False,
) -> List[Path]:
    """Stage-1 results: ``<reports>/<experiment>/<model>/<data>/<test>/test_data.csv``.

    Args:
        reports: The ``reports/`` directory.
        experiments: Patterns an experiment directory must match (default: all).
        models: Patterns a model directory ``{model_name}{model_args}_{seed}``
            must match (default: all).
        seeds: Patterns the seed, the part after the last ``_``, must match.
        progress: Show a progress bar over the model directories.
    """
    model_dirs = [
        model_dir
        for experiment_dir in _subdirs(reports)
        if _matches(experiment_dir.name, experiments)
        for model_dir in _subdirs(experiment_dir)
        if _matches(model_dir.name, models)
        and _matches(model_dir.name.rpartition("_")[2], seeds)
    ]
    files = []
    for model_dir in tqdm(
        model_dirs, desc="scanning", unit="model", disable=not progress
    ):
        for data_dir in _subdirs(model_dir):
            for test_dir in _subdirs(data_dir):
                path = test_dir / TEST_DATA
                if path.is_file():
                    files.append(path)
    return sorted(files)


def read_selection(path: Path) -> Optional[SampleSelection]:
    """Selection of one stage-1 result, reading only the columns it needs."""
    header = pd.read_csv(path, nrows=0).columns
    if "sample_index" not in header:
        return None  # class-level result: one row per presentation, no samples
    columns = ["sample_index", "image_index"]
    columns += [c for c in header if c.endswith(RESPONSE_SUFFIX)]
    table = pd.read_csv(path, usecols=columns, low_memory=False)
    return sample_selection(table)


def target_selection(
    selections: Sequence[SampleSelection],
) -> Optional[SampleSelection]:
    """The selection most runs share; ties go to the one with more responses."""
    if not selections:
        return None
    counts = Counter(selections)
    return max(counts, key=lambda s: (counts[s], sum(n for _, n, _ in s.responses)))


def plan(
    selections: Dict[Path, Optional[SampleSelection]],
    include_empty: bool = False,
) -> Tuple[Dict[str, Optional[SampleSelection]], List[Deviation]]:
    """Compare every run of an experiment with that experiment's target.

    Args:
        selections: Selection of each stage-1 result, keyed by its ``test_data.csv``.
        include_empty: Also flag runs that stored no responses at all.

    Returns:
        ``(targets per experiment, deviations)``.
    """
    by_experiment: Dict[str, Dict[Path, Optional[SampleSelection]]] = defaultdict(dict)
    for path, selection in selections.items():
        by_experiment[path.parts[-5] if len(path.parts) >= 5 else "?"][
            path
        ] = selection

    targets: Dict[str, Optional[SampleSelection]] = {}
    deviations: List[Deviation] = []
    for experiment, runs in sorted(by_experiment.items()):
        stored = [
            s
            for s in runs.values()
            if s is not None and s.responses and any(n for _, n, _ in s.responses)
        ]
        target = target_selection(stored)
        targets[experiment] = target
        for path, selection in sorted(runs.items()):
            if selection is None or selection == target:
                continue
            if not any(n for _, n, _ in selection.responses) and not include_empty:
                continue
            deviations.append(Deviation(experiment, path, selection))
    return targets, deviations


def prune(deviations: Sequence[Deviation]) -> Tuple[int, int]:
    """Delete the deviating stage-1 results and their stage-2 aggregates.

    Returns:
        ``(removed files, removed aggregates)``.
    """
    removed_files = 0
    for deviation in deviations:
        for name in STAGE_1_FILES:
            path = deviation.test_dir / name
            if path.exists():
                path.unlink()
                removed_files += 1
    aggregates = {d.aggregate for d in deviations if d.aggregate.exists()}
    for aggregate in aggregates:
        aggregate.unlink()
    return removed_files, len(aggregates)


def report(
    selections: Dict[Path, Optional[SampleSelection]],
    targets: Dict[str, Optional[SampleSelection]],
    deviations: Sequence[Deviation],
) -> str:
    """A table of what each experiment has and what deviates from it."""
    lines = []
    for experiment, target in sorted(targets.items()):
        runs = [s for p, s in selections.items() if p.parts[-5] == experiment]
        stored = [s for s in runs if s is not None and s.responses]
        empty = sum(1 for s in stored if not any(n for _, n, _ in s.responses))
        deviating = [d for d in deviations if d.experiment == experiment]
        lines.append(
            f"{experiment:<20} runs={len(runs):<5} stored={len(stored):<5} "
            f"no-responses={empty:<4} target="
            f"{target.describe() if target else '-':<45} deviating={len(deviating)}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path, help="project directory holding reports/")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the deviating results (default: report only)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="also delete runs that stored no responses at all",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        metavar="PATTERN",
        help="only experiments matching this pattern (repeatable)",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="PATTERN",
        help="only model directories matching this pattern (repeatable)",
    )
    parser.add_argument(
        "--seed",
        action="append",
        metavar="PATTERN",
        help="only models with a matching seed (repeatable)",
    )
    parser.add_argument("--show", type=int, default=10, help="examples per experiment")
    parser.add_argument(
        "--no-progress", action="store_true", help="hide the progress bars"
    )
    args = parser.parse_args(argv)

    reports = args.root / "reports"
    if not reports.is_dir():
        parser.error(f"{reports} is not a directory")

    progress = not args.no_progress
    files = find_test_files(
        reports, args.experiment, args.model, args.seed, progress=progress
    )
    if not files:
        print(f"No stage-1 test results under {reports} in the given scope")
        return 0
    if args.model or args.seed:
        print(
            "Scoped by model/seed: each experiment's target comes from the "
            "scanned runs only."
        )

    selections = {
        path: read_selection(path)
        for path in tqdm(files, desc="reading", unit="file", disable=not progress)
    }
    targets, deviations = plan(selections, include_empty=args.include_empty)
    print(report(selections, targets, deviations))

    shown = Counter()
    for deviation in deviations:
        if shown[deviation.experiment] >= args.show:
            continue
        shown[deviation.experiment] += 1
        selection = deviation.selection
        have = selection.describe() if selection else "-"
        want = targets[deviation.experiment]
        print(
            f"  {deviation.experiment}: {deviation.test_file} has {have}; "
            f"experiment target {want.describe() if want else '-'}"
        )

    if not args.apply:
        print(
            f"\n{len(deviations)} test result(s) would be deleted. "
            f"Rerun with --apply to delete them, then rerun Snakemake."
        )
        return 0

    removed, aggregates = prune(deviations)
    print(f"\nDeleted {removed} file(s) and {aggregates} aggregate(s).")
    print(
        "Rerun the affected tests with, e.g.\n"
        "  snakemake -s workflow/Snakefile --allowed-rules test_model "
        "process_test_data <targets>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
