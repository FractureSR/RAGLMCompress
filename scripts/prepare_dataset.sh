#!/usr/bin/env bash
# Build the RAC retrieval database (base corpus + index). No training, no test
# pieces, no precomputed retrieval — eval_rac.py chunks + retrieves the held-out
# eval docs live.
set -euo pipefail

python utils/prepare_rac_data.py \
    --dataset datasets/codeparrot_github_code/C.jsonl \
    --n-docs 4000 \
    --base-frac 0.5 \
    --chunk-size 512 --chunk-overlap 0 \
    --retriever bm25 \
    --seed 42 \
    --model pretrained/SmolLM2-135M \
    --out results/rac_c_db
