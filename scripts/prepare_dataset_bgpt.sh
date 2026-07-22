#!/usr/bin/env bash
# Build the bGPT RAC retrieval database (base byte chunks + byte-k-gram index).
# No training, no precomputed retrieval — eval_rac_bgpt.py chunks + retrieves the
# held-out eval samples live. A condition is one complete image patch or PCM
# chunk; eval derives the remaining payload from the fixed media body budget.
set -euo pipefail

# Image
# python utils/prepare_rac_data_bgpt.py \
#    --dataset datasets/clic2024/bmp --modality image \
#    --n-samples 400 --base-frac 0.5 \
#    --image-patch-width 16 --image-patch-height 16 \
#    --retriever bm25 --kgram 4 --seed 42 \
#    --out results/rac_img_db

# Audio
# python utils/prepare_rac_data_bgpt.py \
#    --dataset datasets/vctk_wav --modality audio \
#    --n-samples 200 \
#    --base-frac 0.5 \
#    --audio-chunk-bytes 512 \
#    --retriever bm25 \
#    --kgram 8 \
#    --seed 42 \
#    --out results/rac_vctk

python utils/prepare_rac_data_bgpt.py \
    --dataset datasets/eurosat/Forest --modality image \
    --n-samples 2000 --base-frac 0.5 \
    --image-patch-width 32 --image-patch-height 8 \
    --retriever bm25 --kgram 4 --seed 42 \
    --out results/rac_eurosat_forest
