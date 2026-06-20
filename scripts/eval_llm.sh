python evaluation/eval_llm.py \
    --dataset  datasets/codeparrot_github_code/C.jsonl \
    --model    pretrained/SmolLM2-135M \
    --n-docs 100 \
    --device cuda:5 \
    --max-tokens 2048 \
    --no-decompress
    # --save-compressed compressed \
    # --save-decompressed decompressed \