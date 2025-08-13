"""
This module defines the `project_paths_class`, which provides a structured way to manage and access various project and toolbox directory paths for the 'rhythmic_visual_attention' project. It automatically detects if the code is running on a cluster and adapts paths accordingly, including moving large folders to a scratch partition when necessary.

Note:
    This file contains hardcoded paths and project-specific settings. 
    **You must adapt this file to fit your own path organization and project structure.**
    Update the `project_name`, `toolbox_name`, `user_name`, and default directory paths as needed for your environment.
"""

import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


class project_paths_class:

    this_file = Path(inspect.getfile(lambda: None)).resolve()
    project_name = "<name_of_project_folder>"
    toolbox_name = "DynVision"
    user_name = "<cluster_user_name>"

    def __init__(self, working_dir=None, toolbox_dir=None):
        if self.iam_on_cluster():
            working_dir = Path().home() / self.project_name
            toolbox_dir = Path().home() / self.toolbox_name / self.toolbox_name.lower()

        if working_dir is None:
            working_dir = Path(f"~/{self.project_name}").expanduser().resolve()
        if toolbox_dir is None:
            toolbox_dir = self.this_file.parents[0].resolve()

        self.working_dir = working_dir
        self.toolbox_dir = toolbox_dir

        self._set_paths(working_dir=working_dir, toolbox_dir=self.toolbox_dir)

        if self.iam_on_cluster():
            # move large folders to scratch partition
            self.data.raw = Path("/scratch") / self.user_name / "data" / "raw"
            self.data.processed = (
                Path("/scratch") / self.user_name / "data" / "processed"
            )
            self.models = (
                Path("/scratch") / self.user_name / self.project_name / "models"
            )
            self.reports = (
                Path("/scratch") / self.user_name / self.project_name / "reports"
            )
            self.large_logs = (
                Path("/scratch") / self.user_name / self.project_name / "logs"
            )

        os.environ["WANDB_DIR"] = str(self.large_logs.resolve())
        return None

    def _set_paths(self, working_dir, toolbox_dir=None):
        if toolbox_dir is None:
            toolbox_dir = working_dir
        elif working_dir is None:
            working_dir = toolbox_dir
        elif working_dir is None and toolbox_dir is None:
            raise ValueError("Either working_dir or toolbox_dir must be provided.")
        else:
            pass
        logging.info(f"Toolbox directory: {self.toolbox_dir}")
        logging.info(f"Working directory: {self.working_dir}")

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
        self.figures = working_dir / "reports" / "figures"
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
        self.scripts.features = self.scripts_path / "features"
        self.scripts.workflow = self.scripts_path / "workflow"
        self.scripts.visualization = self.scripts_path / "visualization"
        return None

    def iam_on_cluster(self):
        host_name = os.popen("hostname").read()
        # look for common cluster names
        cluster_names = [
            "hpc",  # Generic HPC systems
            "log-",  # Login nodes
            "greene",  # NYU Greene
            "slurm",  # SLURM-based clusters
            "compute",  # Common compute node prefix
            "node",  # Generic compute nodes
            "cluster",  # Generic cluster systems
        ]
        return any(x in host_name for x in cluster_names)


project_paths = project_paths_class()
