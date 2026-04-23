#!/bin/bash -l
#SBATCH -J cleanup_100M
#SBATCH -o outs/cleanup_100M_%j.out
#SBATCH -e outs/cleanup_100M_%j.err
#SBATCH -p gpu-H200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=crirdchpxd004,crirdchpxd006
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --time=00:05:00

echo "=========================================="
echo "Post-job cleanup: 100M scaling datasets"
echo "Node: $(hostname)"
echo "=========================================="

if [ -d /scratch/keisufaj_datasets ]; then
    SIZE=$(du -sh /scratch/keisufaj_datasets 2>/dev/null | cut -f1)
    echo "Found /scratch/keisufaj_datasets: ${SIZE}"
    rm -rf /scratch/keisufaj_datasets
    echo "✓ Deleted"
else
    echo "Already clean or auto-cleaned by SLURM"
fi

echo "Cleanup complete on $(hostname)"