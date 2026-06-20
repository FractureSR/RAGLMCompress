# RAGLMCompress

Large-model-powered compression experiments for text, RAG-conditioned text,
images, and audio.

## Project Layout

- `compression/`: reusable compressor classes.
  - `LLMCompressor`: token-level compression with a causal LM.
  - `BGPTCompressor`: byte-level compression with bGPT.
  - `RACCompressor`: retrieval-augmented compression — conditions the frozen LM
    on latent prefixes encoded from retrieved patterns (`ChunkEncoder`), with an
    optional `Reranker` selecting the most useful chunks.
- `arithmetic_coder/`: verified range-coder wrapper and probability utilities.
- `utils/`: shared dataset, modality, retrieval (`rag_utils`), bGPT byte-preparation,
  and evaluation helpers.
- `train/`: RAC data preparation and encoder/reranker training.
- `evaluation/`: CLI benchmark scripts.
- `notebooks/`: runnable demos and diagnostics.

## RAC (Retrieval-Augmented Compression)

Compress domain data conditioned on patterns retrieved from a base corpus. The
frozen LM still produces the coding distribution; retrieval supplies a learned
conditioning prefix that lowers the data's entropy.

Pipeline (text):

The base/train/test split is reproduced deterministically from the `--dataset`
and split args (`--chunk-size/--base-frac/--train-frac/--seed`), so it is never
persisted — pass the *same* dataset/split args to every step.  Only the expensive
retriever + retrieval caches are saved (in `prepare`'s `--out`, referenced as
`--data` by the later steps).  Defaults match across scripts, so the short form
below is consistent.

```
DS="--dataset datasets/cosmopedia-100k --n-docs 2000"   # + any --chunk-size/--base-frac/... overrides

# 1. Build the retriever (BM25 + Qwen3-Embedding, fused with RRF) over the i.i.d.
#    base and cache top-k retrieval for the train/test queries.
python train/prepare_rac_data.py $DS --top-k 32 --out datasets/rac/cosmopedia

# 2. Train the ChunkEncoder (softmin-weighted per-chunk cross-entropy). This also
#    dumps a CE cache (into --data) used as reranker labels.
python train/train_encoder.py $DS --data datasets/rac/cosmopedia \
    --model pretrained/SmolLM2-135M --out pretrained/rac_encoder/cosmopedia

# 3. Train the Reranker on the cached marginal CE reductions.
python train/train_reranker.py $DS --data datasets/rac/cosmopedia \
    --model pretrained/SmolLM2-135M --out pretrained/rac_reranker/cosmopedia

# 4. Evaluate RAC vs baselines on the test split.
python evaluation/eval_rac.py $DS --data datasets/rac/cosmopedia \
    --model pretrained/SmolLM2-135M --encoder pretrained/rac_encoder/cosmopedia \
    --reranker pretrained/rac_reranker/cosmopedia --methods none,token_rag,rac,rac_rerank
```

The RAC components (`rag_utils`, `ChunkEncoder`, `Reranker`, `RACCompressor`) are
written against injected interfaces (a backbone compressor, an embedding source,
pluggable retrieval scorers) so the planned audio/image-on-bGPT variant can reuse
them once `BGPTCompressor` gains an `inputs_embeds` path.

## Notes

Decoding intentionally pads the partially decoded sequence to the original full
length before each model forward pass. This mirrors the encoder prefill numerics
and should not be replaced with ordinary autoregressive decoding.

Prepare local datasets and pretrained models under the paths used by the
notebooks or pass explicit paths to the evaluation scripts.
