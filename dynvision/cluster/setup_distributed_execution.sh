#!/bin/bash
# setup_distributed_execution.sh
set +u

# Set distributed training variables first
MASTER_PORT=$(python -c 'import random; print(random.randint(10000, 65535))')
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "============= SLURM Configuration Check ============="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not_set}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-not_set}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-not_set}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-not_set}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-not_set}"
echo "SLURM_CPUS_PER_NODE: ${SLURM_CPUS_PER_NODE:-not_set}"
echo "SLURM_TASKS_PER_NODE: ${SLURM_TASKS_PER_NODE:-not_set}"
echo "SLURM_GPUS: ${SLURM_GPUS:-not_set}"
echo "SLURM_CPUS: ${SLURM_CPUS:-not_set}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-not_set}"
echo "SLURM_JOB_CPUS: ${SLURM_JOB_CPUS:-not_set}"
echo "SLURM_LOCALID: ${SLURM_LOCALID:-not_set}"
echo "SLURM_PROCID: ${SLURM_PROCID:-not_set}"
echo "SLURM_NODEID: ${SLURM_NODEID:-not_set}"

echo "============= GPU Detection ============="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

# Determine distributed training based on GPU availability using environment variables
# Count GPUs from CUDA_VISIBLE_DEVICES or SLURM_JOB_GPUS
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Count comma-separated GPU IDs in CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -E '^[0-9]+$' | wc -l)
    echo "GPU count from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_COUNT"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
    # Count comma-separated GPU IDs in SLURM_JOB_GPUS
    CUDA_VISIBLE_COUNT=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -E '^[0-9]+$' | wc -l)
    echo "GPU count from SLURM_JOB_GPUS: $CUDA_VISIBLE_COUNT"
else
    # Fallback: check if any GPU device files exist
    CUDA_VISIBLE_COUNT=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l | tr -d '[:space:]' || echo "0")
    echo "GPU count from device files: $CUDA_VISIBLE_COUNT"
fi

CUDA_VISIBLE_COUNT=$(echo "$CUDA_VISIBLE_COUNT" | tr -d '[:space:]') # Trim whitespace or newlines

echo "============= Training Mode ============="
# Use the actual number of visible GPUs for distributed training decision
if [ "$CUDA_VISIBLE_COUNT" -gt 1 ]; then
    echo "Multiple GPUs detected ($CUDA_VISIBLE_COUNT), enabling distributed training"
    USE_DISTRIBUTED=true
    # For PyTorch Lightning DDP, we use the GPU count as world size
    WORLD_SIZE=$CUDA_VISIBLE_COUNT
else
    echo "Single or no GPU detected ($CUDA_VISIBLE_COUNT), using single GPU training"
    USE_DISTRIBUTED=false
    WORLD_SIZE=1
fi

NODE_RANK=${SLURM_NODEID:-0}
LOCAL_RANK=${SLURM_LOCALID:-0}
RANK=${SLURM_PROCID:-0}
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
GPU_PER_NODE=$(( ${CUDA_VISIBLE_COUNT:-1} / ${NUM_NODES:-1} )) || GPU_PER_NODE=1

# Export all variables for both host and container environments
export MASTER_PORT MASTER_ADDR WORLD_SIZE NODE_RANK LOCAL_RANK RANK USE_DISTRIBUTED NUM_NODES GPU_PER_NODE

echo "============= Network Configuration ============="
# Initialize variables to avoid unbound variable errors
NCCL_SOCKET_IFNAME=""

# Set NCCL and training configuration only if using distributed
if [ "$USE_DISTRIBUTED" = "true" ]; then
    echo "Configuring NCCL for distributed training..."
    export NCCL_DEBUG=INFO
    export NCCL_TIMEOUT=1800
    # export NCCL_IB_DISABLE=1       # Disable InfiniBand if causing issues
    export TORCH_NCCL_BLOCKING_WAIT=0
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_DISTRIBUTED_DEBUG=DETAIL  # For debugging
    export PYTHONFAULTHANDLER=1
    export PYTHONUNBUFFERED=1
    export CUDA_LAUNCH_BLOCKING=0
    
    # Try to detect network interface automatically
    if ip addr show ib0 >/dev/null 2>&1; then
        export NCCL_SOCKET_IFNAME=ib0
        export NCCL_IB_DISABLE=0
        export NCCL_IB_HCA=mlx5_0:1
        echo "Using InfiniBand interface ib0"
    elif ip addr show eth0 >/dev/null 2>&1; then
        export NCCL_SOCKET_IFNAME=eth0
        echo "Using Ethernet interface eth0"
    else
        export NCCL_SOCKET_IFNAME=""
        echo "WARNING: Could not detect network interface, using defaults"
    fi

    # Better error handling
    set -e
    trap 'echo "Job interrupted or failed"; exit 1' SIGUSR1
    echo "Node: $SLURMD_NODENAME"
    echo "GPUs: $CUDA_VISIBLE_DEVICES"
    echo "Memory per CPU: ${SLURM_MEM_PER_CPU}MB"
    
    # Singularity environment variables (explicit container variables)
    export SINGULARITYENV_WORLD_SIZE=$WORLD_SIZE
    export SINGULARITYENV_RANK=$RANK
    export SINGULARITYENV_LOCAL_RANK=$LOCAL_RANK
    export SINGULARITYENV_MASTER_ADDR=$MASTER_ADDR
    export SINGULARITYENV_MASTER_PORT=$MASTER_PORT
    export SINGULARITYENV_NCCL_DEBUG=$NCCL_DEBUG
    export SINGULARITYENV_NCCL_TIMEOUT=$NCCL_TIMEOUT
    export SINGULARITYENV_NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
    export SINGULARITYENV_NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-""}
    export SINGULARITYENV_NCCL_IB_HCA=${NCCL_IB_HCA:-""}
    export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING
    export SINGULARITYENV_PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER
    export SINGULARITYENV_CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING
else
    echo "Single GPU training - skipping NCCL configuration"
fi

export SINGULARITYENV_USE_DISTRIBUTED=$USE_DISTRIBUTED
export SINGULARITYENV_CUDA_VISIBLE_COUNT=$CUDA_VISIBLE_COUNT

echo "============= Configuration Summary ============="
echo "Job ID: ${SLURM_JOB_ID:-not_set}"
echo "Node List: ${SLURM_JOB_NODELIST:-not_set}"
echo "Number of Nodes: ${SLURM_JOB_NUM_NODES:-not_set}"
echo "GPU Count: $CUDA_VISIBLE_COUNT"
echo "Use Distributed: $USE_DISTRIBUTED"
echo "World Size: $WORLD_SIZE"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node Rank: $NODE_RANK"
echo "Local Rank: $LOCAL_RANK"
echo "Global Rank: $RANK"
echo "Network Interface: ${NCCL_SOCKET_IFNAME:-not_set}"
