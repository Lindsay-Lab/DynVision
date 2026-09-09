"""Seam for path layout: makes environment detection and directory layout
explicit inputs instead of import-time side effects.

Local and cluster layouts are two adapters selected by `Environment.is_cluster`
in `PathLayout.for_environment`. Neither hostname detection
(`detect_environment`) nor process-environment mutation (`set_wandb_dir`) runs
as a side effect of importing this module or constructing a `PathLayout` — both
are explicit calls made by the caller.

See: docs/development/planning/project-paths-seam.md
Issue: https://github.com/Lindsay-Lab/DynVision/issues/14
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

# Hostname substrings that indicate an HPC/cluster environment.
_CLUSTER_HOSTNAME_MARKERS = (
    "hpc",  # Generic HPC systems
    "log-",  # Login nodes
    "greene",  # NYU Greene
    "slurm",  # SLURM-based clusters
    "compute",  # Common compute node prefix
    "node",  # Generic compute nodes
    "cluster",  # Generic cluster systems
)


@dataclass(frozen=True)
class Environment:
    """Explicit description of the machine DynVision is running on.

    Replaces the import-time `os.popen("hostname")` shell-out with a value
    that can be constructed once (via `detect_environment()`) and passed
    around, or fabricated directly in tests (`Environment.local()` /
    `Environment.cluster()`) without touching the real filesystem or shell.
    """

    is_cluster: bool
    hostname: str = field(default="")

    @classmethod
    def local(cls) -> "Environment":
        """Build an explicit local (non-cluster) environment."""

        return cls(is_cluster=False)

    @classmethod
    def cluster(cls) -> "Environment":
        """Build an explicit cluster environment."""

        return cls(is_cluster=True)


def detect_environment() -> Environment:
    """Detect the current `Environment` from the machine's hostname.

    This performs the `hostname` shell-out that used to run at import time in
    `project_paths.py`. Call it explicitly (typically once, lazily, on first
    use) instead of relying on module import to trigger it.
    """

    hostname = os.popen("hostname").read()
    is_cluster = any(marker in hostname for marker in _CLUSTER_HOSTNAME_MARKERS)
    return Environment(is_cluster=is_cluster, hostname=hostname)


class PathLayout:
    """Resolved directory layout for a given `Environment`.

    Construct via `PathLayout.for_environment(...)` — the local/cluster
    adapters (`_local_layout`/`_cluster_layout`) are implementation details of
    that classmethod, not meant to be called directly.

    Attribute names match the pre-refactor `project_paths_class` exactly, so
    existing call sites (`project_paths.data.raw`, `project_paths.models`,
    `project_paths.scripts.configs`, ...) don't need to change.
    """

    # Personalization point for the scratch-partition prefix on cluster
    # layouts (`/scratch/<user_name>/...`). Subclasses (e.g.
    # `dynvision/project_paths.py`) override this; the generic seam defaults
    # to no user segment.
    user_name: str = ""

    def __init__(self, working_dir: Path, toolbox_dir: Path, environment: Environment):
        self.working_dir = working_dir
        self.toolbox_dir = toolbox_dir
        self._environment = environment
        self._set_base_paths(working_dir, toolbox_dir)

    @classmethod
    def for_environment(
        cls,
        env: Environment,
        working_dir: Optional[Path] = None,
        toolbox_dir: Optional[Path] = None,
    ) -> "PathLayout":
        """Build a `PathLayout` for `env`.

        This is the seam: local vs. cluster are two adapters selected here,
        instead of `if self.iam_on_cluster():` branches buried inside
        `__init__`.
        """

        if env.is_cluster:
            return cls._cluster_layout(working_dir, toolbox_dir, env)
        return cls._local_layout(working_dir, toolbox_dir, env)

    @classmethod
    def _local_layout(
        cls, working_dir: Optional[Path], toolbox_dir: Optional[Path], env: Environment
    ) -> "PathLayout":
        """Adapter for local (non-cluster) execution."""

        if working_dir is None:
            raise ValueError("working_dir must be provided for the local layout.")
        if toolbox_dir is None:
            raise ValueError("toolbox_dir must be provided for the local layout.")

        return cls(working_dir=working_dir, toolbox_dir=toolbox_dir, environment=env)

    @classmethod
    def _cluster_layout(
        cls, working_dir: Optional[Path], toolbox_dir: Optional[Path], env: Environment
    ) -> "PathLayout":
        """Adapter for cluster execution: moves large folders to scratch."""

        if working_dir is None:
            raise ValueError("working_dir must be provided for the cluster layout.")
        if toolbox_dir is None:
            raise ValueError("toolbox_dir must be provided for the cluster layout.")

        layout = cls(working_dir=working_dir, toolbox_dir=toolbox_dir, environment=env)
        layout._apply_scratch_overrides()
        return layout

    def _apply_scratch_overrides(self) -> None:
        """Move large folders to the scratch partition (cluster-only)."""

        scratch = Path("/scratch") / self.user_name
        self.data.raw = scratch / "data" / "raw"
        self.data.interim = scratch / "data" / "interim"
        self.data.processed = scratch / "data" / "processed"
        self.data.external = scratch / "data" / "external"
        self.models = scratch / self.working_dir.name / "models"
        self.reports = scratch / self.working_dir.name / "reports"
        self.large_logs = scratch / self.working_dir.name / "logs"

    def _set_base_paths(self, working_dir: Path, toolbox_dir: Path) -> None:
        logger.info("Toolbox directory: %s", toolbox_dir)
        logger.info("Working directory: %s", working_dir)

        self.data_path = working_dir / "data"
        self.data = SimpleNamespace(data=self.data_path)
        self.data.raw = self.data_path / "raw"
        self.data.external = self.data_path / "external"
        self.data.interim = self.data_path / "interim"
        self.data.processed = self.data_path / "processed"

        self.models = working_dir / "models"
        self.notebooks = working_dir / "notebooks"
        self.references = working_dir / "references"
        self.reports = working_dir / "reports"
        self.figures = working_dir / "figures"
        self.logs = working_dir / "logs"
        self.large_logs = working_dir / "logs"
        self.benchmarks = self.logs / "benchmarks"

        self.scripts_path = toolbox_dir
        self.scripts = SimpleNamespace(scripts=self.scripts_path)
        self.scripts.data = self.scripts_path / "data"
        self.scripts.utils = self.scripts_path / "utils"
        self.scripts.models = self.scripts_path / "models"
        self.scripts.losses = self.scripts_path / "losses"
        self.scripts.configs = self.scripts_path / "configs"
        self.scripts.workflow = self.scripts_path / "workflow"
        self.scripts.visualization = self.scripts_path / "visualization"

    def iam_on_cluster(self) -> bool:
        """Whether this layout was built for a cluster `Environment`.

        Kept for call sites that ask a constructed layout whether it's on a
        cluster (`dynvision/params/mode_registry.py`,
        `dynvision/utils/data_utils.py`). Prefer inspecting the `Environment`
        directly (`env.is_cluster`) in new code.
        """

        return self._environment.is_cluster


def set_wandb_dir(layout: PathLayout) -> None:
    """Set `WANDB_DIR` from a resolved `PathLayout`.

    The only place `os.environ` is mutated by the path-layout seam. Call this
    explicitly at runtime entry points that start a wandb run (e.g.
    `dynvision/runtime/train_model.py`) instead of relying on it having
    already happened as a side effect of constructing/importing a path
    layout.
    """

    os.environ["WANDB_DIR"] = str(layout.large_logs.resolve())
