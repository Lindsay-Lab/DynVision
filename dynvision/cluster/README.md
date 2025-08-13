# Cluster Scripts

NOTE: The presented cluster files are personalized to a specific cluster environment setup. They are thought to provide a template to facilitate scaling up workflows on HPC resources by letting snakemake itself submit rules as individual jobs. They do need to be adapted to your environment (e.g. conda and singularity commands)!

This folder contains example scripts for executing the workflow on a compute cluster with a SLURM queuing system.

## Scripts

- **snakejob.sh**: This script is used to submit a job to the SLURM scheduler. It sets up the environment and runs the Snakemake workflow.
- **snakecharm.sh**: This script runs the Snakemake workflow in the background and logs the output to a file.

## Usage

1. **snakejob.sh**:
   - Submit the job to the SLURM scheduler using the following command:
     ```bash
     sbatch snakejob.sh
     ```

2. **snakecharm.sh**:
   - Run the workflow in the background using the following command:
     ```bash
     ./snakecharm.sh [snakemake arguments]
     ```
   - Check the log file for details:
     ```bash
     tail -f ../../logs/slurm/snakecharm_[arguments].log
     ```

## Notes

- Path settings are globally defined in `../project_paths.py`.
- You may adapt the `snakecharm` script for other non-SLURM compute clusters by defining other Snakemake cluster profiles in the `../workflow/profiles` folder and setting the corresponding configurations in its `config.yaml` file.
- Modify the SLURM job parameters (e.g., time, memory, CPUs) as needed for your specific requirements.
