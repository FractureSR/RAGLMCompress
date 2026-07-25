#!/usr/bin/env bash
set -euo pipefail

# Image
# python evaluation/eval_bgpt.py \
#     --modality image \
#     --dataset  results/rac_medmnist_bloodmnist_128_train/eval_samples.pkl \
#     --model    pretrained/bgpt/weights-image.pth \
#     --n-samples 100 \
#     --image-patch-width 32 --image-patch-height 32 \
#     --device "${DEVICE:-cuda:3}" \
#     --no-decompress

# Audio baseline on the same held-out samples persisted by RAC preparation.
python evaluation/eval_bgpt.py \
    --modality audio \
    --dataset results/rac_ljspeech_wav/eval_samples.pkl \
    --model    pretrained/bgpt/weights-audio.pth \
    --n-samples 100 \
    --device "${DEVICE:-cuda:3}" \
    --no-decompress
