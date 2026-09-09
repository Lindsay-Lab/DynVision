"""Personal path layout for this checkout, built on the `path_layout` seam.

Historically this module instantiated `project_paths_class()` at import time,
which shelled out to `hostname` and mutated `os.environ["WANDB_DIR"]` as a
side effect of the import statement alone (see
docs/development/planning/project-paths-seam.md, issue #14). The generic
detection/layout/mutation logic now lives in `dynvision.path_layout`; this
module only supplies the personal overrides (`project_name`, `user_name`,
local working-dir fallback) and exposes `project_paths` as a **lazy** proxy —
the underlying `PathLayout` is constructed on first attribute access, not at
import.

This file is intentionally personalized (see `.gitignore:12` and
`project_paths_template.py`, the checked-in template other users copy from).
"""

from pathlib import Path
from typing import Optional

from dynvision.path_layout import Environment, PathLayout, detect_environment


class PersonalPathLayout(PathLayout):
    """Personalized `PathLayout`: local/cluster adapters with defaults filled
    in for this user, matching the pre-refactor `project_paths_class`
    behavior exactly."""

    project_name = (
        "DynVision_Working"  # working directory name (separate from toolbox_name)
    )
    toolbox_name = "DynVision"
    user_name = "rg5022"

    @classmethod
    def _local_layout(
        cls,
        working_dir: Optional[Path],
        toolbox_dir: Optional[Path],
        env: Environment,
    ) -> "PersonalPathLayout":
        if working_dir is None:
            working_dir = Path("/home/rgutzen/01_PROJECTS/Modeling_Dynamical_Vision")
        if toolbox_dir is None:
            toolbox_dir = Path(__file__).resolve().parent
        return super()._local_layout(working_dir, toolbox_dir, env)

    @classmethod
    def _cluster_layout(
        cls,
        working_dir: Optional[Path],
        toolbox_dir: Optional[Path],
        env: Environment,
    ) -> "PersonalPathLayout":
        if working_dir is None:
            working_dir = Path.home() / cls.project_name
        if toolbox_dir is None:
            toolbox_dir = Path.home() / cls.toolbox_name / cls.toolbox_name.lower()
        layout = super()._cluster_layout(working_dir, toolbox_dir, env)
        layout.references = Path.home() / cls.toolbox_name / "references"
        return layout


class _LazyPathLayout:
    """Proxy that defers `PersonalPathLayout` construction (and the hostname
    detection inside it) to first attribute access.

    Keeps `from dynvision.project_paths import project_paths` and every
    existing `project_paths.<attr>` call site working unchanged, while
    removing the import-time side effects (`hostname` shell-out and
    `WANDB_DIR` mutation) that `project_paths = project_paths_class()` used
    to have.
    """

    def __init__(self) -> None:
        self._layout: Optional[PersonalPathLayout] = None

    def _ensure(self) -> PersonalPathLayout:
        if self._layout is None:
            env = detect_environment()
            self._layout = PersonalPathLayout.for_environment(env)
        return self._layout

    def __getattr__(self, name: str):
        return getattr(self._ensure(), name)


project_paths = _LazyPathLayout()
