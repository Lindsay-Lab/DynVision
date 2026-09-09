"""Template for `dynvision/project_paths.py`.

Copy this file to `dynvision/project_paths.py` (gitignored) and fill in your
own `project_name`, `toolbox_name`, `user_name`, and local `working_dir`
fallback. See `docs/development/planning/project-paths-seam.md` for how the
underlying `dynvision.path_layout` seam works, and issue #14 for why
`project_paths` is a lazy proxy rather than an eagerly-constructed singleton.
"""

from pathlib import Path
from typing import Optional

from dynvision.path_layout import Environment, PathLayout, detect_environment


class PersonalPathLayout(PathLayout):
    """Personalized `PathLayout`: fill in your own values below."""

    project_name = "rhythmic_visual_attention"  # working directory name (separate from toolbox_name)
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
            working_dir = Path("/home/rgutzen/01_PROJECTS/rhythmic_visual_attention")
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
        return super()._cluster_layout(working_dir, toolbox_dir, env)


class _LazyPathLayout:
    """Proxy that defers `PersonalPathLayout` construction (and the hostname
    detection inside it) to first attribute access. See
    `dynvision/project_paths.py` for the full explanation.
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
