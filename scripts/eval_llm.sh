python evaluation/eval_llm.py \
    --dataset  results/rac_arxiv_cl/eval_docs.jsonl \
    --model    pretrained/SmolLM2-135M \
    --n-docs 100 \
    --device cuda:0 \
    --max-tokens 8192 \
    --no-decompress
    # --save-compressed compressed \
    # --save-decompressed decompressed \