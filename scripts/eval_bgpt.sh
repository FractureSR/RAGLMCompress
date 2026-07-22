#!/usr/bin/env bash
set -euo pipefail

# Image
# python evaluation/eval_bgpt.py \
#    --modality image \
#    --dataset  datasets/clic2024/bmp \
#    --model    pretrained/bgpt/weights-image.pth \
#    --n-samples 100 \
#    --image-patch-width 32 --image-patch-height 32 \
#    --device "${DEVICE:-cuda:0}" \
#    --output results/bgpt_image.csv

# Audio baseline on the same held-out samples persisted by RAC preparation.
#python evaluation/eval_bgpt.py \
#    --modality audio \
#    --dataset results/rac_vctk/eval_samples.pkl \
#    --model    pretrained/bgpt/weights-audio.pth \
#    --n-samples 50 \
#    --device "${DEVICE:-cuda:0}" \
#    --no-decompress


python evaluation/eval_bgpt.py \
    --modality image \
    --dataset  results/rac_eurosat_forest/eval_samples.pkl \
    --model    pretrained/bgpt/weights-image.pth \
    --n-samples 500 \
    --image-patch-width 32 --image-patch-height 32 \
    --device "${DEVICE:-cuda:3}" \
    --no-decompress
