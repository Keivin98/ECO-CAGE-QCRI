#!/bin/bash
# Smart job submission script that tries partitions in priority order
# Priority: gpu-H200 > gpu-A100 > gpu-all

set -e

SCRIPT="test_percentile_30M.sh"
REQUIRED_GPUS=2
USER=$(whoami)

echo "=================================="
echo "Smart Job Submission"
echo "=================================="

# Function to check available resources on a partition
check_partition() {
    local partition=$1
    local gres_type=$2

    # Count idle/mix nodes
    idle_nodes=$(sinfo -p ${partition} -h -o "%t" | grep -E "idle|mix" | wc -l)

    # Count user's pending jobs on this partition
    pending_jobs=$(squeue -u ${USER} -p ${partition} -h -t PD | wc -l)

    echo "  Partition: ${partition}"
    echo "    Available nodes: ${idle_nodes}"
    echo "    Your pending jobs: ${pending_jobs}"

    # Return 0 if partition looks good, 1 otherwise
    if [ ${idle_nodes} -gt 0 ] && [ ${pending_jobs} -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Function to submit job to a specific partition
submit_job() {
    local partition=$1
    local gres_type=$2
    local account=$3
    local qos=$4

    echo ""
    echo "Submitting to ${partition}..."

    # Create temporary submission script with modified SBATCH directives
    TEMP_SCRIPT=$(mktemp)
    sed "s|#SBATCH -p .*|#SBATCH -p ${partition}|" ${SCRIPT} | \
    sed "s|#SBATCH --gres gpu:.*|#SBATCH --gres gpu:${gres_type}:${REQUIRED_GPUS}|" | \
    sed "s|#SBATCH -A .*|#SBATCH -A ${account}|" | \
    sed "s|#SBATCH -q .*|#SBATCH -q ${qos}|" > ${TEMP_SCRIPT}

    # Submit the job
    JOBID=$(sbatch ${TEMP_SCRIPT} | awk '{print $4}')
    rm ${TEMP_SCRIPT}

    echo "Submitted job array: ${JOBID}"
    echo ""
    echo "Monitor with: watch -n 5 'squeue -u ${USER}'"
    echo "Check output: tail -f outs/fp4_scale_test_${JOBID}_*.out"

    return 0
}

# Try partitions in priority order
echo ""
echo "Checking resource availability..."
echo ""

# Option 1: gpu-H200
if check_partition "gpu-H200" "H200"; then
    echo ""
    echo "✓ gpu-H200 looks good!"
    submit_job "gpu-H200" "H200_141GB" "H200" "h200_qos"
    exit 0
else
    echo "  → Skipping (no resources or pending jobs)"
fi

# Option 2: gpu-A100
if check_partition "gpu-A100" "A100"; then
    echo ""
    echo "✓ gpu-A100 looks good!"
    submit_job "gpu-A100" "A100_80GB" "A100" "a100_qos"
    exit 0
else
    echo "  → Skipping (no resources or pending jobs)"
fi

# Option 3: gpu-all (fallback)
echo ""
echo "⚠ Falling back to gpu-all partition"
idle_any=$(sinfo -p gpu-all -h -o "%t" | grep -E "idle|mix" | wc -l)
echo "  Available gpu-all nodes: ${idle_any}"

if [ ${idle_any} -gt 0 ]; then
    submit_job "gpu-all" "V100" "v100" "v100_qos"
else
    echo ""
    echo "❌ No resources available on any partition!"
    echo "   Try again later or submit manually with:"
    echo "   sbatch ${SCRIPT}"
    exit 1
fi
