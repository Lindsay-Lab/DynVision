"""Sample selection of test runs, and a warning when compared runs differ.

A test run evaluates some set of samples, and records layer responses for a
subset of them (``store_test_responses`` batches; the other rows hold NaN).
Averages over runs that rest on different samples mix different images: in
the dt sweep of 2026-11, some dt=2 runs recorded responses for 32 samples and
others for 384, which moved the first-pulse height across intervals by 2%.

Whenever results are compared -- aggregated into one table, or loaded for a
plot -- :func:`warn_on_sample_mismatch` compares the selections of all runs
and logs one warning that lists the distinct selections.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_KEYS = ("sample_index", "image_index")
RESPONSE_SUFFIX = "_response_avg"
MAX_RUN_VALUES = 100  # a column with more distinct values is a measurement

# Columns that vary within a run: identifiers of a row, and measurements
_ROW_COLUMNS = {"time_ms", "label_valid", "idle_steps"}
_ROW_MARKERS = ("_index", "confidence", "accuracy", "_response", "classifier", "guess")


@dataclass(frozen=True)
class SampleSelection:
    """The samples a run evaluated, and per layer those it recorded responses
    for (layers can differ: V1 for 32 samples, V2 for 512)."""

    n_samples: int
    samples_digest: str
    responses: Tuple[Tuple[str, int, str], ...] = ()  # (layer, count, digest)

    def describe(self) -> str:
        text = f"{self.n_samples} samples"
        counts = {layer: n for layer, n, _ in self.responses}
        if len(set(counts.values())) == 1:
            return f"{text}, responses recorded for {next(iter(counts.values()))}"
        if counts:
            listed = ", ".join(f"{layer} {n}" for layer, n in counts.items())
            return f"{text}, responses recorded for {listed}"
        return text

    @property
    def digests(self) -> str:
        layers = {digest for _, _, digest in self.responses}
        return f"samples {self.samples_digest}, responses {'/'.join(sorted(layers))}"


def _digest(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "-"
    hashed = pd.util.hash_pandas_object(rows, index=False).to_numpy()
    return hashlib.sha1(hashed.tobytes()).hexdigest()[:12]


def sample_selection(df: pd.DataFrame) -> Optional[SampleSelection]:
    """Selection of one run; ``None`` for tables without ``sample_index``
    (e.g. class-level aggregates)."""
    if "sample_index" not in df.columns:
        return None
    keys = [key for key in SAMPLE_KEYS if key in df.columns]
    samples = df[keys].drop_duplicates().sort_values(keys)
    responses = []
    for column in sorted(c for c in df.columns if c.endswith(RESPONSE_SUFFIX)):
        recorded = df.loc[df[column].notna(), keys].drop_duplicates()
        recorded = recorded.sort_values(keys)
        layer = column[: -len(RESPONSE_SUFFIX)]
        responses.append((layer, len(recorded), _digest(recorded)))
    return SampleSelection(
        n_samples=len(samples),
        samples_digest=_digest(samples),
        responses=tuple(responses),
    )


def run_columns(df: pd.DataFrame) -> list:
    """Columns that identify a run in a combined table.

    Everything that is neither a row identifier, a measurement, nor a
    column with more than ``MAX_RUN_VALUES`` distinct values: the swept
    parameter and its ms column, the category, ``dt``, ``seed``, ``status``.
    """
    columns = []
    for column in df.columns:
        if column in _ROW_COLUMNS or any(m in column for m in _ROW_MARKERS):
            continue
        if df[column].nunique(dropna=False) > MAX_RUN_VALUES:
            continue
        columns.append(column)
    return columns


def warn_if_selections_differ(
    selections: Mapping[str, Optional[SampleSelection]], context: str = ""
) -> bool:
    """Log one warning if the runs in ``selections`` rest on different samples.

    Args:
        selections: Selection per run label; ``None`` entries are skipped.
        context: What is being compared, for the message.

    Returns:
        True if a warning was logged.
    """
    groups: Dict[SampleSelection, list] = {}
    for label, selection in selections.items():
        if selection is not None:
            groups.setdefault(selection, []).append(label)
    if len(groups) < 2:
        return False

    # Selections of equal counts but other samples are told apart by digest
    descriptions = [selection.describe() for selection in groups]
    lines = []
    for selection, labels in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tag = ""
        if descriptions.count(selection.describe()) > 1:
            tag = f" [{selection.digests}]"
        examples = "; ".join(labels[:3]) + (" ..." if len(labels) > 3 else "")
        lines.append(
            f"  - {selection.describe()}{tag}: {len(labels)} run(s), e.g. {examples}"
        )
    logger.warning(
        "Compared test runs rest on different samples%s: %d selections among "
        "%d runs. Averages across these runs mix different images.\n%s",
        f" ({context})" if context else "",
        len(groups),
        sum(len(labels) for labels in groups.values()),
        "\n".join(lines),
    )
    return True


def warn_on_sample_mismatch(
    df: pd.DataFrame,
    run_cols: Optional[Sequence[str]] = None,
    context: str = "",
) -> bool:
    """Compare the sample selections of the runs in a combined table.

    Args:
        df: Combined result table.
        run_cols: Columns that identify a run; inferred with
            :func:`run_columns` when ``None``.
        context: What is being compared, for the message.

    Returns:
        True if a warning was logged.
    """
    if "sample_index" not in df.columns:
        return False
    run_cols = list(run_columns(df) if run_cols is None else run_cols)
    if not run_cols:
        return False
    selections = {}
    for values, run in df.groupby(run_cols, dropna=False, observed=True, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        label = ", ".join(f"{c}={v}" for c, v in zip(run_cols, values))
        selections[label] = sample_selection(run)
    return warn_if_selections_differ(selections, context)
