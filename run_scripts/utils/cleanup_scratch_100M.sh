#!/bin/bash

# Cleanup /scratch datasets from 100M scaling jobs
# Run this AFTER jobs 277389_1 and 277389_2 complete
# Nodes: crirdchpxd004, crirdchpxd006

echo "=========================================="
echo "Cleaning /scratch on 100M scaling nodes"
echo "=========================================="

# Node 1: crirdchpxd004 (Job 277389_1)
echo ""
echo "Cleaning crirdchpxd004..."
srun --partition=gpu-H200 \
     --nodes=1 \
     --nodelist=crirdchpxd004 \
     --job-name=cleanup_scratch \
     bash -c 'du -sh /scratch/keisufaj_datasets 2>/dev/null || echo "Already clean"; rm -rf /scratch/keisufaj_datasets && echo "Deleted /scratch/keisufaj_datasets"'

# Node 2: crirdchpxd006 (Job 277389_2)
echo ""
echo "Cleaning crirdchpxd006..."
srun --partition=gpu-H200 \
     --nodes=1 \
     --nodelist=crirdchpxd006 \
     --job-name=cleanup_scratch \
     bash -c 'du -sh /scratch/keisufaj_datasets 2>/dev/null || echo "Already clean"; rm -rf /scratch/keisufaj_datasets && echo "Deleted /scratch/keisufaj_datasets"'

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
