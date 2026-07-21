#!/usr/bin/env bash
set -euo pipefail

python evaluation/eval_rac_llm.py \
    --database results/rac_arxiv_cl \
    --model pretrained/SmolLM2-135M \
    --n-docs 100 \
    --m 4 \
    --cascade \
    --cascade-top-k 4 \
    --cascade-retriever \
    --calibrate \
    --calib-docs 10 \
    --device "${DEVICE:-cuda:0}" \
    --no-decompress
