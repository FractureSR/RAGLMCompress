#!/usr/bin/env bash
# Build the RAC retrieval database (base corpus, held-out docs, and index).
# No training or precomputed retrieval: eval_rac_llm.py chunks and retrieves
# the persisted held-out documents live.
set -euo pipefail

python utils/prepare_rac_data_llm.py \
    --dataset datasets/arxiv_tex/cs_cl.jsonl \
    --n-docs 300 \
    --base-frac 0.5 \
    --chunk-size 512 --chunk-overlap 0 \
    --retriever bm25 \
    --seed 42 \
    --model pretrained/SmolLM2-135M \
    --out results/rac_arxiv_cl
