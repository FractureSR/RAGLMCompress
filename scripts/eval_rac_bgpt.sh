#!/usr/bin/env bash
set -euo pipefail

#python evaluation/eval_rac_bgpt.py \
#    --database results/rac_medmnist_bloodmnist_128_train/ \
#    --model pretrained/bgpt/weights-image.pth \
#    --m 4 \
#    --n-samples 100 \
#    --image-payload-width 32 --image-payload-height 24 \
#    --cascade \
#    --cascade-max-cond 1 \
#    --cascade-retriever \
#    --device "${DEVICE:-cuda:3}" \
#    --no-decompress

# Audio:
#python evaluation/eval_rac_bgpt.py \
#    --database results/rac_ljspeech_wav \
#    --model pretrained/bgpt/weights-audio.pth \
#    --m 4 \
#    --device "${DEVICE:-cuda:3}" \
#    --n-samples 0 \
#    --no-decompress

python evaluation/eval_rac_bgpt.py \
    --database results/rac_medmnist_bloodmnist_128_train/ \
    --model pretrained/bgpt/weights-image.pth \
    --m 4 \
    --n-samples 100 \
    --image-payload-width 32 --image-payload-height 28 \
    --device "${DEVICE:-cuda:3}" \
    --no-decompress
