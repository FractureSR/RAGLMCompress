#!/bin/bash
# Submit: sbatch scripts/slurm-bgpt-image.sh
# Run BGPT image compression test on Tiny-ImageNet BMPs (5 images).
#
# Results (no polling needed): when the job finishes, check:
#   cat slurm-bgpt-image-<JOBID>.out    # stdout (compression ratios, etc.)
#   cat slurm-bgpt-image-<JOBID>.err    # stderr (errors if any)
# Reconstructed images (if successful): output-img/reconstructed_*.bmp

#SBATCH --job-name=bgpt-image
#SBATCH --partition=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm-bgpt-image-%j.out
#SBATCH -e slurm-bgpt-image-%j.err
# Email when job ends or fails: uncomment both lines below, set your email, then:
#   scancel <JOBID>   # cancel current job if already submitted
#   sbatch scripts/slurm-bgpt-image.sh
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.email@uwaterloo.ca

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH=.

# Use source activate per watgpu docs (not conda run)
source activate ragllm

echo "=== SLURM BGPT Image Compression ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working dir: $(pwd)"
echo ""

python -m evaluation.BGPTCompress

echo ""
echo "End: $(date)"
