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

Compress data by conditioning the frozen LM on the raw tokens of similar chunks
retrieved from a base corpus **using the data itself**. RAC ≠ RAG: we already
have the data `x`, retrieve with `x`, and re-encode `x` more cheaply. Because we
are compressing we can *measure* each candidate's exact effect on the code length
and keep the **oracle** (best of top-k) — or no condition if none beats its
side-info cost. No training.

Pipeline (text) — no training, no precomputed retrieval:

```
# 1. Build the retrieval database: fix a slice of the dataset as the base corpus,
#    chunk it, and index it (BM25 syntactic by default; no embedding model needed).
python utils/prepare_rac_data.py \
    --dataset datasets/codeparrot_github_code/C.jsonl --n-docs 4000 \
    --base-frac 0.5 --chunk-size 512 --retriever bm25 \
    --model pretrained/SmolLM2-135M --out results/rac_c_db

# 2. Evaluate oracle RAC vs the no-condition baseline on the held-out docs,
#    chunked + retrieved live (mirrors eval_llm). Add --cascade / --calibrate.
python evaluation/eval_rac.py --database results/rac_c_db \
    --model pretrained/SmolLM2-135M --m 16 --n-docs 200 --device cuda:0
```

The chosen base ids are transmitted as side information (the decoder can't re-run
retrieval — the query is the unknown data); their bit cost (fixed, or a static
table built with `--calibrate`) is charged for honest bpb/ratio. The retriever
(`utils/rag_utils.py`) and the oracle compressor (`compression/rac_compressor.py`)
are modality-agnostic, so the planned audio/image-on-bGPT variant reuses them once
`BGPTCompressor` grows a byte-prefix path.

## Notes

Decoding intentionally pads the partially decoded sequence to the original full
length before each model forward pass. This mirrors the encoder prefill numerics
and should not be replaced with ordinary autoregressive decoding.

Prepare local datasets and pretrained models under the paths used by the
notebooks or pass explicit paths to the evaluation scripts.
