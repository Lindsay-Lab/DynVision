# Cluster Scripts

This folder contains example scripts for executing DynVision workflows on a compute cluster with a SLURM queuing system. The scripts use Snakemake's cluster profile to submit individual rules as SLURM jobs and manage dependencies automatically.

**Note**: These scripts are templates — you **must** fill in your own system-specific values before use. Look for `<YOUR_...>` placeholders in each file.

## Quick Setup

1. **Replace all `<YOUR_...>` placeholders** in the cluster scripts and config:
   - `<YOUR_PROJECT_PATH>` — path to your working directory (models, data, logs)
   - `<YOUR_SLURM_ACCOUNT>` — your SLURM account name
   - `<YOUR_OVERLAY_PATH>` — path to your Singularity overlay image
   - `<YOUR_CONTAINER_IMAGE_PATH>` — path to the CUDA container image
   - `<YOUR_CONDA_MODULE>` — the conda module to load (e.g. `anaconda3/2025.06`)
   - `<YOUR_SNAKEMAKE_ENV>` — conda environment with Snakemake installed
   - `<YOUR_CONDA_ENV>` — conda environment for running DynVision
   - `<GPU_TYPE>` — GPU model available on your cluster (e.g. `h200`, `a100`, `v100`)

2. **Set your project paths** in `dynvision/project_paths.py` — cluster execution auto-detects via SLURM environment variables.

## Scripts

| Script | Purpose |
|--------|---------|
| `snakecharm.sh` | Wraps `_snakecharm.sh` and runs the workflow in the background with logging |
| `_snakecharm.sh` | Core cluster execution: loads modules, activates conda, runs Snakemake with slurm profile |
| `snakejob.sh` | Submits a single SLURM job (with SBATCH directives inline) for direct execution |
| `snakejob_example.sh` | Example SLURM submission script with extended time and GPU constraint |
| `executor_wrapper.sh` | Singularity container wrapper for training — handles distributed env setup |
| `interactive.sh` | Queues one interactive job (`srun --pty`) or a detached command (`--sbatch`) inside the same container and conda env a rule runs in |
| `setup_distributed_execution.sh` | Distributed training setup: GPU detection, NCCL config, MASTER_ADDR/PORT |
| `profiles/slurm/config.yaml` | Snakemake slurm profile: default resources, per-rule overrides |

## Usage

### Snakecharm (background workflow manager)

Runs the entire Snakemake workflow in the background, submitting individual rules as SLURM jobs via the cluster profile:

```bash
cd dynvision/cluster
./snakecharm.sh --config experiment=duration model_name=DyRCNNx4 data_name=cifar100
```

Check progress:
```bash
tail -f ../../logs/slurm/snakecharm_<args>.log
```

### Direct SLURM submission

For when you need a single node to run all rules sequentially:

```bash
cd dynvision/cluster
sbatch snakejob.sh
```

Or with custom arguments to Snakemake (after editing the script to pass `$@`):
```bash
sbatch snakejob.sh -- --config experiment=contrast
```

### Interactive session

Run a shell, or one command, in exactly the environment a rule runs in (same
container image, overlays and conda env as `executor_wrapper.sh`):

```bash
dynvision/cluster/interactive.sh                    # shell in the container
dynvision/cluster/interactive.sh python script.py   # one command
TIME=4:00:00 dynvision/cluster/interactive.sh       # longer wall time
```

Fill in its `<YOUR_...>` placeholders with the values of `executor_wrapper.sh`,
or set them per call (`PROJECT_DIR`, `ACCOUNT`, `CONTAINER_IMAGE`, `CONDA_ENV`,
`OVERLAYS` as a space-separated list, `CONDA_SH`); the script stops while one
is left. `PROJECT_DIR` sets the working directory and the log directory;
`PYTHONPATH_REPO=1` puts this checkout on `PYTHONPATH`; `ACCOUNT`, `MEM`,
`CPUS`, `GRES`, `CONSTRAINT` override the resource request.

### Distributed training

For multi-GPU jobs, edit the `train_model_distributed` section in `profiles/slurm/config.yaml` and ensure `setup_distributed_execution.sh` is sourced before running your workflow.

## Adapting to Your Cluster

1. **SLURM accounts**: Replace all `<YOUR_SLURM_ACCOUNT>` placeholders with your institution's account name
2. **Container paths**: Update container image paths to match your cluster's available images
3. **Conda modules**: The module used to load conda varies by cluster — check `module avail` for available options
4. **GPU types**: Replace `<GPU_TYPE>` with the GPU model on your cluster
5. **Overlay images**: Create your own SquashFS overlay with your conda environment, or use a different container strategy

## Non-SLURM Clusters

For PBS, SGE, LSF, or other schedulers:
- Create a new profile in `profiles/<scheduler>/config.yaml`
- Snakemake supports these via executor plugins — see the [Snakemake docs](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles) for details
