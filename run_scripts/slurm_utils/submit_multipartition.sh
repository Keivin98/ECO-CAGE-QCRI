#!/bin/bash
# Multi-partition job submission that spreads array jobs across available resources
# Usage: ./submit_multipartition.sh <script_name>

set -e

SCRIPT="${1:-test_percentile_sweep.sh}"
USER=$(whoami)

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script '$SCRIPT' not found!"
    exit 1
fi

echo "=================================="
echo "Multi-Partition Job Submission"
echo "=================================="
echo "Script: $SCRIPT"
echo ""

# Extract array range from script
ARRAY_RANGE=$(grep "^#SBATCH --array=" "$SCRIPT" | sed 's/#SBATCH --array=//')
if [ -z "$ARRAY_RANGE" ]; then
    echo "Error: No --array directive found in $SCRIPT"
    exit 1
fi

# Parse array range (supports formats like "1-6", "1-10:2")
START=$(echo $ARRAY_RANGE | cut -d'-' -f1)
END=$(echo $ARRAY_RANGE | cut -d'-' -f2 | cut -d':' -f1)
TOTAL_JOBS=$((END - START + 1))

echo "Array range: $ARRAY_RANGE (${TOTAL_JOBS} jobs total)"
echo ""

# Define partition configurations in priority order
declare -a PARTITIONS=("gpu-H200" "gpu-A100" "gpu-all")
declare -a GRES_TYPES=("H200_141GB" "A100_80GB" "V100")
declare -a ACCOUNTS=("H200" "A100" "v100")
declare -a QOS=("h200_qos" "a100_qos" "v100_qos")

# Function to check how many jobs we can submit to a partition
check_capacity() {
    local partition=$1

    # Count idle/mix nodes
    local idle=$(sinfo -p ${partition} -h -o "%t" | grep -E "idle|mix" | wc -l)

    # Count user's running + pending jobs on this partition
    local running=$(squeue -u ${USER} -p ${partition} -h -t R | wc -l)
    local pending=$(squeue -u ${USER} -p ${partition} -h -t PD | wc -l)

    echo "${idle} ${running} ${pending}"
}

# Function to submit a subset of array jobs
submit_subset() {
    local partition=$1
    local gres=$2
    local account=$3
    local qos=$4
    local start_idx=$5
    local end_idx=$6

    if [ $start_idx -gt $end_idx ]; then
        return
    fi

    local array_spec="${start_idx}-${end_idx}"

    echo "  → Submitting jobs ${array_spec} to ${partition}..."

    # Create temp script with modified SBATCH directives
    TEMP_SCRIPT=$(mktemp)
    sed "s|#SBATCH -p .*|#SBATCH -p ${partition}|" ${SCRIPT} | \
    sed "s|#SBATCH --gres gpu:.*|#SBATCH --gres gpu:${gres}:2|" | \
    sed "s|#SBATCH -A .*|#SBATCH -A ${account}|" | \
    sed "s|#SBATCH -q .*|#SBATCH -q ${qos}|" | \
    sed "s|#SBATCH --array=.*|#SBATCH --array=${array_spec}|" > ${TEMP_SCRIPT}

    JOBID=$(sbatch ${TEMP_SCRIPT} 2>&1 | grep -oP '(?<=Submitted batch job )\d+' || echo "FAILED")
    rm ${TEMP_SCRIPT}

    if [ "$JOBID" != "FAILED" ]; then
        echo "    ✓ Submitted job array ${JOBID} (${array_spec})"
    else
        echo "    ✗ Submission failed for ${array_spec}"
    fi
}

# Distribute jobs across partitions
current_job=$START
submitted_any=false

for i in "${!PARTITIONS[@]}"; do
    if [ $current_job -gt $END ]; then
        break
    fi

    partition="${PARTITIONS[$i]}"
    gres="${GRES_TYPES[$i]}"
    account="${ACCOUNTS[$i]}"
    qos="${QOS[$i]}"

    read idle running pending <<< $(check_capacity "$partition")

    echo "Checking ${partition}:"
    echo "  Idle nodes: ${idle}"
    echo "  Your running jobs: ${running}"
    echo "  Your pending jobs: ${pending}"

    # Heuristic: if there are idle nodes and not too many pending, submit some jobs
    if [ $idle -gt 0 ] && [ $pending -lt 3 ]; then
        # Estimate how many jobs we can submit (conservative: 2 per idle node)
        capacity=$((idle * 2))
        jobs_to_submit=$((END - current_job + 1))

        # Don't exceed estimated capacity
        if [ $jobs_to_submit -gt $capacity ]; then
            jobs_to_submit=$capacity
        fi

        end_job=$((current_job + jobs_to_submit - 1))

        if [ $end_job -le $END ]; then
            submit_subset "$partition" "$gres" "$account" "$qos" "$current_job" "$end_job"
            current_job=$((end_job + 1))
            submitted_any=true
        fi
    else
        echo "  → Skipping (no capacity or too many pending)"
    fi
    echo ""
done

if [ "$submitted_any" = true ]; then
    echo "=================================="
    echo "✓ Jobs submitted!"
    echo "Monitor: watch -n 5 'squeue -u ${USER}'"
    echo "=================================="
else
    echo "=================================="
    echo "❌ Could not submit any jobs!"
    echo "All partitions are busy or at capacity."
    echo "=================================="
    exit 1
fi
