#!/usr/bin/env bash
# Build the bGPT RAC retrieval database (base byte chunks + byte-k-gram index).
# No training, no precomputed retrieval — eval_rac_bgpt.py chunks + retrieves the
# held-out eval samples live. Keep chunk_patches*(max_cond+1)+2 <= 512 by choosing
# --patch-px (image) / --chunk-ms (audio) small enough for the prefix budget.
set -euo pipefail

# Image (clic2024 BMPs)
python utils/prepare_rac_data_bgpt.py \
    --dataset datasets/clic2024/bmp --modality image \
    --n-samples 400 --base-frac 0.5 \
    --patch-px 16 --patch-size 16 \
    --retriever bm25 --kgram 4 --seed 42 \
    --out results/rac_img_db

# Audio (preprocessed People's Speech WAVs) — smaller chunks keep prefix+payload
# within the context. Run download_peoples_speech_microset.py first.
# python utils/prepare_rac_data_bgpt.py \
#     --dataset datasets/peoples_speech_microset_wav --modality audio \
#     --n-samples 200 --base-frac 0.5 \
#     --chunk-ms 250 --patch-size 16 \
#     --retriever bm25 --kgram 4 --seed 42 \
#     --out results/rac_audio_db
