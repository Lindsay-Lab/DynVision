#!/usr/bin/env bash
# Interactive SLURM job inside the workflow's container and conda env.
#
# Mirrors what every rule runs in: the container image, overlays and conda
# env of executor_wrapper.sh. Fill in the <YOUR_...> placeholders below with
# the same values as in executor_wrapper.sh, or set them per call through the
# environment variables of the same name. The script refuses to run while a
# placeholder is left.
#
# Usage (from the login node):
#
#   ./interactive.sh                     # shell inside the container
#   ./interactive.sh python -c 'import torch; print(torch.cuda.is_available())'
#   TIME=4:00:00 ./interactive.sh        # longer wall time
#   PYTHONPATH_REPO=1 ./interactive.sh   # put the DynVision checkout on PYTHONPATH
#   ACCOUNT=<account> ./interactive.sh   # other slurm account
#
# Options:
#   --sbatch   submit with sbatch instead of srun --pty (detached; use for a
#              command, not for a shell)
#   --help     this text
#
# Notebook users: after the shell opens, run `jupyter lab --no-browser
# --ip=$(hostname -f)` and tunnel to the node you are on.

set -euo pipefail

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT")/../.." && pwd)"    # the DynVision checkout

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SCRIPT"
}
case "${1:-}" in --help | -h) usage; exit 0 ;; esac
# The working directory that holds models/, reports/ and logs/.
PROJECT_DIR="${PROJECT_DIR:-<YOUR_PROJECT_PATH>}"

# Every <YOUR_...> value must be filled in, here or through the environment.
require_filled() {
    local name value
    for name in "$@"; do
        value="${!name}"
        if [[ "$value" == *"<YOUR_"* ]]; then
            echo "$name is unset ($value): fill it in at the top of $SCRIPT or set $name=..." >&2
            exit 1
        fi
    done
}
require_filled PROJECT_DIR
requested_project="$PROJECT_DIR"

if ! PROJECT_DIR="$(cd "$requested_project" 2>/dev/null && pwd)"; then
    echo "PROJECT_DIR=$requested_project does not exist; set PROJECT_DIR=/path" >&2
    exit 1
fi
LOG_DIR="$PROJECT_DIR/logs/slurm"

# --- resource request (match cluster/snakejob.sh and the snakemake profile) ---
TIME="${TIME:-2:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-16}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-dynvision-interactive}"
ACCOUNT="${ACCOUNT:-<YOUR_SLURM_ACCOUNT>}"
CONSTRAINT="${CONSTRAINT:-}"             # e.g. CONSTRAINT="a100|h100"

# --- container (same values as cluster/executor_wrapper.sh) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-<YOUR_CONTAINER_IMAGE_PATH>}"
CONDA_ENV="${CONDA_ENV:-<YOUR_CONDA_ENV>}"
# conda.sh inside the overlay that holds the env
CONDA_SH="${CONDA_SH:-/ext3/miniforge3/etc/profile.d/conda.sh}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
# Space-separated list of overlays, each with its mount mode
if [[ -n "${OVERLAYS:-}" ]]; then
    read -r -a OVERLAYS <<< "$OVERLAYS"
else
    OVERLAYS=(
        "<YOUR_OVERLAY_PATH>/env.ext3:ro"
        # dataset overlays, e.g. "/path/to/imagenet-train.sqf:ro"
    )
fi
require_filled ACCOUNT CONTAINER_IMAGE CONDA_ENV
for overlay in "${OVERLAYS[@]}"; do
    if [[ "$overlay" == *"<YOUR_"* ]]; then
        echo "overlay $overlay is unset: fill in OVERLAYS at the top of $SCRIPT or set OVERLAYS=..." >&2
        exit 1
    fi
done

# --- inside the container: the workflow's environment -----------------------
if [[ "${1:-}" == "--inside" ]]; then
    shift
    # source conda.sh only, and start from a clean PYTHONPATH
    source "$CONDA_SH"
    unset PYTHONPATH
    conda activate "$CONDA_ENV"

    cd "$WORKDIR"
    if [[ "${PYTHONPATH_REPO:-0}" == "1" ]]; then
        export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi

    echo "================ container ================"
    echo "host:      $(hostname)"
    echo "slurm job: ${SLURM_JOB_ID:-none}"
    echo "conda env: $CONDA_DEFAULT_ENV"
    echo "python:    $(python --version 2>&1)"
    echo "workdir:   $PWD"
    echo "pythonpath:${PYTHONPATH:- (unset)}"
    if command -v nvidia-smi >/dev/null; then nvidia-smi -L; fi
    python - <<'PY' || true
try:
    import torch
    print(f"torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}, "
          f"devices: {torch.cuda.device_count()}")
except Exception as exc:  # noqa: BLE001 - diagnostics only
    print(f"torch: not importable ({exc})")
try:
    import dynvision
    print(f"dynvision: {dynvision.__file__}")
except Exception as exc:  # noqa: BLE001
    print(f"dynvision: not importable ({exc}); set PYTHONPATH_REPO=1 for the checkout")
PY
    echo "==========================================="

    if [[ $# -gt 0 ]]; then
        exec "$@"
    fi
    exec bash -i
    # NOT reached
fi

# --- on the node: enter the container --------------------------------------
if [[ "${1:-}" == "--inner" ]]; then
    shift
    overlay_args=()
    for overlay in "${OVERLAYS[@]}"; do
        overlay_args+=(--overlay "$overlay")
    done
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec singularity exec --nv \
        --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
        --env "WORKDIR=$WORKDIR" \
        --env "PYTHONPATH_REPO=${PYTHONPATH_REPO:-0}" \
        --env "PROJECT_DIR=$PROJECT_DIR" \
        --env "REPO_ROOT=$REPO_ROOT" \
        --env "CONDA_SH=$CONDA_SH" \
        --env "CONDA_ENV=$CONDA_ENV" \
        "${overlay_args[@]}" \
        "$CONTAINER_IMAGE" \
        "$SCRIPT" --inside "$@"
fi

# --- on the login node: queue the job --------------------------------------
case "${1:-}" in
    --sbatch) shift; MODE=sbatch ;;
    *) MODE=interactive ;;
esac

mkdir -p "$LOG_DIR"
if ! command -v singularity >/dev/null; then
    module load singularity 2>/dev/null || true
fi
[[ -r "$CONTAINER_IMAGE" ]] || { echo "container image not readable: $CONTAINER_IMAGE" >&2; exit 1; }

# srun --pty rejects --output/--input/--error, so the log flags are added only
# for the sbatch path; an interactive session writes to its own terminal.
common=(--job-name="$JOB_NAME" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS"
        --mem="$MEM" --time="$TIME" --account="$ACCOUNT" --gres="$GRES")
if [[ -n "$CONSTRAINT" ]]; then common+=(--constraint="$CONSTRAINT"); fi

if [[ "$MODE" == sbatch ]]; then
    if [[ $# -eq 0 ]]; then
        echo "--sbatch needs a command, e.g. --sbatch python foo.py" >&2
        exit 2
    fi
    # A payload script on shared storage, not --wrap: the arguments survive
    # quoting intact (bash %q), and /tmp may be node-local.
    payload="$LOG_DIR/interactive_payload_$(date +%Y%m%dT%H%M%S).sh"
    {
        echo '#!/usr/bin/env bash'
        printf '%q ' "$SCRIPT" --inner "$@"
        echo
    } > "$payload"
    sbatch "${common[@]}" --output="$LOG_DIR/%j.out" --error="$LOG_DIR/%j.err" "$payload"
    echo "queued $payload; follow with: tail -f $LOG_DIR/<jobid>.out"
    exit 0
fi

if [[ ! -t 1 ]]; then
    echo "interactive mode needs a terminal; use --sbatch <command>" >&2
    exit 2
fi
echo "queueing interactive job ($TIME, $MEM, $CPUS cpus, $GRES, account $ACCOUNT)"
exec srun --pty "${common[@]}" "$SCRIPT" --inner "$@"
