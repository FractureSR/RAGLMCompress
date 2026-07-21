#!/usr/bin/env bash
set -euo pipefail

# python evaluation/eval_rac_bgpt.py \
#    --database results/rac_img_db \
#    --model pretrained/bgpt/weights-image.pth \
#    --m 16 \
#    --image-payload-width 32 --image-payload-height 24 \
#    --device "${DEVICE:-cuda:0}" \
#    --no-decompress

# Audio:
python evaluation/eval_rac_bgpt.py \
    --database results/rac_vctk \
    --model pretrained/bgpt/weights-audio.pth \
    --m 4 \
    --device "${DEVICE:-cuda:0}" \
    --n-samples 50 \
    --cascade \
    --cascade-max-cond 2 \
    --cascade-retriever \
    --calibrate \
    --calib-samples 10 \
    --no-decompress
