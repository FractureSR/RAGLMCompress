python evaluation/eval_rac.py \
    --database results/rac_c_db \
    --model pretrained/SmolLM2-135M \
    --m 16 \
    --n-docs 100 \
    --device cuda:5 \
    --max-tokens 2048 \
    --no-decompress
