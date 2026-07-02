# Cluster Integration Guide

This guide explains how to run DynVision workflows on high-performance computing (HPC) clusters. It covers setup, configuration, and best practices for efficient cluster execution.

## Overview

DynVision provides two methods for running workflows on HPC clusters:

1. **Basic Execution**: Submit a single job script that handles environment setup and runs the entire workflow sequentially within one SLURM allocation.
2. **Advanced Execution**: Use Snakemake's SLURM executor plugin to submit and manage individual jobs in parallel.

### Architecture

The two execution methods use different scripts:

```
dynvision/cluster/
├── snakejob.sh               # Basic: single-job submission script
├── snakecharm.sh              # Advanced: wrapper (runs _snakecharm.sh in background)
├── _snakecharm.sh             # Advanced: core orchestration script
├── executor_wrapper.sh        # Advanced: per-job container/environment setup
├── clean_env_vars.sh          # Utility: cleans PyTorch rank variables
└── profiles/
   └── slurm/
      └── config.yaml          # Advanced: SLURM resource profile
```

**Basic execution** runs snakemake inside a single SLURM job (optionally inside a Singularity container). All workflow rules execute sequentially within that allocation.

**Advanced execution** runs snakemake as an orchestrator on the cluster's native filesystem. Snakemake then submits each rule as a separate SLURM job, where `executor_wrapper.sh` handles container/environment activation per job. This enables parallel execution of independent rules.

## Prerequisites

Before running workflows on a cluster, ensure:

1. **Cluster Access**: You have access to a compute cluster with read/write access to a home and scratch directory.

2. **Getting DynVision**: Download or sync the DynVision folder into your cluster directory.

3. **Data Access**: If you are working with non-standard datasets that cannot be automatically downloaded within the workflow execution, make sure they are available on the cluster.

4. **Path Management**: Review `dynvision/project_paths.py` and set the alternative cluster paths according to your needs (scratch partitions, log directories, etc.).

5. **Main Environment**: Set up an environment with DynVision and all its dependencies installed. Depending on your system, this may involve setting up a Singularity/Apptainer image, a Docker container, or a native conda environment. See the [Installation Guide](installation.md).

## Basic Execution (Single Job)

The basic method uses a single SLURM job script to run the entire workflow. This is simpler to set up but less efficient for large workflows since all rules run sequentially within one allocation.

1. **Create Job Script**

   Use the provided template `dynvision/cluster/snakejob.sh` and adapt it to your cluster. The key sections to configure:

   ```slurm
   #!/usr/bin/env bash
   #SBATCH -o /path/to/logs/slurm/%j.out
   #SBATCH -e /path/to/logs/slurm/%j.err
   #SBATCH --time=2:00:00
   #SBATCH --mem=32G
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=16
   #SBATCH --gres=gpu:1
   #SBATCH --account=your_account

   module purge

   # Option A: Using Singularity container
   singularity exec --nv \
       --overlay /path/to/overlay.ext3:ro \
       /path/to/container.sif \
       bash -c "
   source /ext3/env.sh
   conda activate myenv

   cd ../workflow/
   snakemake \$@ \
       --cores 16 \
       --resources gpu=1 cpu=16
   "

   # Option B: Using native conda environment (no container)
   # conda activate myenv
   # cd ../workflow/
   # snakemake $@ --cores 16 --resources gpu=1 cpu=16
   ```

   The script passes any CLI arguments (`$@`) to snakemake, so you can specify targets, config overrides, etc.

2. **Submit Job**
   ```bash
   # Run default workflow
   sbatch snakejob.sh

   # Run specific target with config
   sbatch snakejob.sh --config model_name=DyRCNNx4 data_name=cifar100
   ```

3. **Monitor Job**
   ```bash
   squeue -u $USER
   ```

   It is also recommended to make use of the Weights & Biases integration to monitor training progress online.

## Advanced Execution (Parallel Jobs)

The advanced method uses Snakemake's [SLURM executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) to submit each workflow rule as a separate SLURM job. This enables parallel execution and better resource utilization.

### 1. Executor Environment (snake-env)

The advanced execution requires a **separate lightweight environment** on the cluster's native filesystem (outside any container). This environment only needs snakemake and the SLURM executor plugin --- it does NOT need DynVision or its dependencies.

Why? Snakemake must be able to call SLURM commands (`sbatch`, `srun`, `sacct`) directly, which are typically not available inside Singularity containers.

Create the environment using conda:

```bash
conda create -n snake-env -c conda-forge -c bioconda python=3.12 snakemake snakemake-executor-plugin-slurm -y
```

> **Important**: Use Python 3.12 (not 3.13+). Newer Python versions may have compatibility issues with snakemake dependencies.

Verify the installation:
```bash
conda activate snake-env
snakemake --version          # should show 9.x
snakemake --help | grep slurm   # should list slurm as an executor option
```

> **Note**: If your cluster provides snakemake via a module (e.g., `module load bioinformatics/...`), make sure the conda environment's snakemake takes precedence. The module version may not include the SLURM executor plugin. When in doubt, run `which snakemake` after activation to confirm it points to the conda environment.

### 2. Configure `_snakecharm.sh`

Edit `dynvision/cluster/_snakecharm.sh` to activate your snake-env. The key section to adapt:

```bash
source ~/.bashrc
module purge
module load anaconda3/2025.06      # or however conda is made available on your system
conda activate /path/to/snake-env  # the environment created in step 1
```

Important details:

- Use `conda activate`, not `source activate` (which is deprecated and may fail)
- `module purge` first to avoid conflicts with system-provided snakemake versions
- Only load the minimal modules needed to make conda available

### 3. Configure `executor_wrapper.sh`

Edit `dynvision/cluster/executor_wrapper.sh` to match your cluster's container and environment setup. The key configuration variables at the top of the file:

```bash
readonly CONTAINER_IMAGE="/path/to/your/container.sif"
readonly CONDA_ENV="your_env_name"

readonly OVERLAYS=(
    "/path/to/your/overlay.ext3:ro"
    # Add additional overlay mounts as needed (e.g., dataset squashfs)
)
```

If your cluster does not use Singularity, you can modify `executor_wrapper.sh` to activate a native conda environment instead of launching a container.

### 4. Configure SLURM Profile

The SLURM profile (`dynvision/cluster/profiles/slurm/config.yaml`) defines resource allocations for each workflow rule. Key settings to adapt:

```yaml
# Account / billing
slurm_account: "your_account_name"

# Log directory
slurm-logdir: "/path/to/logs/slurm"

# Default resources (applied to all rules unless overridden)
default-resources:
  runtime: 1800       # in seconds (see note below)
  mem_mb: 32000       # in MB
  cpus_per_task: 16

# Rule-specific resources
set-resources:
  train_model:
    mem: 80G
    runtime: 48000     # 800 minutes
    gpu: 1
    tasks_per_gpu: 0   # see note below
    cpus_per_task: 16
```

**Critical notes for the SLURM profile:**

- **Runtime is in seconds**: The SLURM executor plugin divides the `runtime` value by 60 before passing it to SLURM's `--time` flag (which expects minutes). So `runtime: 1800` results in `--time=30` (30 minutes). Always specify runtime in seconds.

- **GPU resources**: The plugin's `gpu` key maps to SLURM's `--gpus` flag. However, not all clusters support `--gpus` (job-level GPU allocation). If you get `Requested node configuration is not available` errors with `gpu: 1`, test your cluster directly:
  ```bash
  # Test which GPU flag format your cluster supports:
  sbatch --gpus=1 --mem=1G -t 1 --wrap="nvidia-smi"           # job-level (plugin default)
  sbatch --gres=gpu:1 --mem=1G -t 1 --wrap="nvidia-smi"       # per-node (traditional)
  sbatch --gpus-per-node=1 --mem=1G -t 1 --wrap="nvidia-smi"  # per-node (alternative)
  ```
  If `--gpus` fails, use `slurm_extra: "'--gpus-per-node=1'"` instead of `gpu: 1`. The plugin blocks `--gres` in `slurm_extra` but allows `--gpus-per-node`.

- **Disable `--ntasks-per-gpu`**: When using the `gpu` resource key, the plugin may automatically add `--ntasks-per-gpu=1`, which can cause job rejection on some clusters. Set `tasks_per_gpu: 0` to suppress this. (Not needed if using `slurm_extra` for GPU allocation instead.)

- **GPU model selection**: If you need a specific GPU type, use `gpu_model` (not `constraint`). Check your cluster's available GPU names with `sinfo` or cluster documentation.

- **Partitions**: Many clusters auto-assign partitions based on requested resources. Check your cluster documentation before adding `slurm_partition` --- on some systems, specifying a partition manually can cause errors.

- **`slurm_account`**: Must be set in `default-resources` and/or in each rule under `set-resources`. Check your available accounts with `sacctmgr show associations user=$USER`.

### 5. Use Snakecharm Wrapper

Navigate to the `dynvision/` directory and run:

```bash
# Run all default targets
./cluster/snakecharm.sh

# Run a specific target with config
./cluster/snakecharm.sh manuscript_figures --config test_batch=32

# Dry run
./cluster/snakecharm.sh -n
```

Pass arguments as you would to a regular `snakemake` command. Each argument should be passed separately (do not wrap multiple arguments in quotes).

The wrapper runs the workflow in the background via `nohup`. Check the log file printed to stdout for progress:

```bash
tail -f /path/to/logs/slurm/snakecharm_<id>.log
```

### 6. Monitor Execution

```bash
# Check your running/pending SLURM jobs
squeue -u $USER

# Check a specific job's resource usage
sacct -j <job_id> --format=JobID,Elapsed,Timelimit,State,MaxRSS

# View snakemake orchestrator log
tail -f /path/to/logs/slurm/snakecharm_<id>.log

# View a specific rule's SLURM log
ls /path/to/logs/slurm/
```

## Available Cluster Scripts

DynVision ships with ready-to-use cluster scripts and configuration templates in `dynvision/cluster/`:

| Script | Purpose |
|--------|---------|
| `snakecharm.sh` | Wrapper that runs the workflow in the background with logging |
| `_snakecharm.sh` | Core orchestration: loads modules, activates conda, runs Snakemake with slurm profile |
| `snakejob.sh` | Single SLURM job submission script (with inline SBATCH directives) |
| `executor_wrapper.sh` | Container wrapper for training — handles distributed env setup |
| `setup_distributed_execution.sh` | Distributed training setup: GPU detection, NCCL config, MASTER_ADDR/PORT |
| `profiles/slurm/config.yaml` | Snakemake slurm profile: default resources, per-rule overrides |

These scripts are templates — you **must** fill in your own system-specific values before use. Look for `<YOUR_...>` placeholders in each file.

## Cluster Path Setup

Before running workflows, set your project paths in `dynvision/project_paths.py`. The file includes an `iam_on_cluster()` check that auto-detects cluster nodes via hostname patterns (`hpc`, `greene`, `slurm`, `compute`). When detected, large data directories (data, models, reports, logs) are redirected to scratch partitions.

## Environment Adaptation

DynVision automatically adapts to cluster environments:

1. **Environment Detection**
   ```python
   # In project_paths.py
   def iam_on_cluster(self):
       host_name = os.popen("hostname").read()
       cluster_names = ["hpc", "greene", "slurm", "compute"]
       return any(x in host_name for x in cluster_names)
   ```

2. **Path Management**
      - Large data directories move to scratch partitions
      - Logs redirect to appropriate locations
      - Container mounts configured automatically
   ```python
   # in project_paths.py
   if self.iam_on_cluster():
      # move large folders to scratch partition
      self.data.raw = Path("/scratch") / self.user_name / "data"
      self.models = (
            Path("/scratch") / self.user_name / self.project_name / "models"
      )
      self.reports = (
            Path("/scratch") / self.user_name / self.project_name / "reports"
      )
      self.large_logs = (
            Path("/scratch") / self.user_name / self.project_name / "logs"
      )
   ```

3. **Resource Scaling**
      - Batch sizes adjust for development vs. production
      - Progress bars disable on compute nodes
      - Memory limits scale based on available resources
   ```yaml
   # Debugging settings
   debug_batch_size: 3
   debug_check_val_every_n_epoch: 1
   debug_log_every_n_steps: 1
   debug_accumulate_grad_batches: 1
   debug_enable_progress_bar: True
   ```

## Troubleshooting

### Common Issues

- **`invalid choice: 'slurm'` for executor**: The snakemake binary being used doesn't have the SLURM plugin installed. Check `which snakemake` to confirm it points to your snake-env. System module versions of snakemake may shadow the conda version.

- **Jobs timing out immediately**: Check that `runtime` values in the SLURM profile are in **seconds** (the plugin divides by 60). Use `sacct -j <jobid> --format=Timelimit` to verify the actual time limit SLURM received.

- **`Requested node configuration not available`**: Often caused by `--ntasks-per-gpu` being set automatically. Add `tasks_per_gpu: 0` to the rule's resources. Can also be caused by invalid `constraint` or `slurm_partition` values.

- **`--gres not allowed in slurm_extra`**: Use the `gpu` resource key instead. The SLURM executor plugin manages GRES allocation automatically.

- **`assert self.workflow.is_main_process`**: Known bug in snakemake 9.14.0--9.14.4. Upgrade to snakemake >= 9.14.5.

- **`command not found` errors in `_snakecharm.sh`**: Use `conda activate` (not `source activate`). Ensure `module purge` doesn't remove conda itself --- load the anaconda module after purging.

## Related Resources

- [Snakemake SLURM Executor Plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
- [Snakemake CLI Reference](https://snakemake.readthedocs.io/en/stable/executing/cli.html)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [DynVision Configuration Reference](../reference/configuration.md)
- [Workflow Organization](workflows.md)