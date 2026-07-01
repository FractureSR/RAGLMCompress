# RAGLMCompress

Large-model-powered compression experiments for text, RAG-conditioned text,
images, and audio. Every compressor turns a frozen model's next-symbol
distribution into an arithmetic code, so the compressed size is the model's
exact code length for the data.

## Project Layout

- `compression/`: reusable compressor classes.
  - `LLMCompressor`: token-level compression with a causal LM. An optional
    `PromptContext` prepends conditioning tokens before the data; the coder
    skips those prefix positions.
  - `BGPTCompressor`: byte-level compression with bGPT.
  - `RACCompressor`: retrieval-augmented compression — prepends the **raw tokens**
    of retrieved base chunks as the LM prefix and keeps the **oracle** choice
    (best of top-k by exact code length), or no condition if none beats its
    side-info cost. No training.
  - `rac_index.py`: index side-info coders (`FixedIndexCoder`,
    `CalibratedIndexCoder`) that price the transmitted base-chunk ids.
  - `base_compressor.py` / `types.py`: shared arithmetic-coding kernels and
    dataclasses (`CompressedData`, `LMScore`, `PromptContext`).
- `arithmetic_coder/`: verified range-coder wrapper and probability utilities.
- `utils/`: shared dataset, modality, retrieval (`rag_utils`), bGPT
  byte-preparation, and evaluation helpers. `prepare_rac_data.py` builds the RAC
  retrieval database.
- `evaluation/`: CLI benchmark scripts (`eval_llm`, `eval_rac`, `eval_bgpt`,
  image codec baselines).
- `scripts/`: dataset preparation and ready-to-run eval invocations.
- `bgpt/`: upstream bGPT model/training code.

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
table built with `--calibrate`) is charged for honest bpb/ratio.

**Prefix budget.** Each condition is one full base chunk of `chunk_size` tokens,
and at most `max_cond` conditions are prepended per piece (`max_cond` = number of
`--cascade` levels, or 1 when cascade is off). The prefix budget therefore
satisfies `max_ctx == chunk_size * max_cond`, which `eval_rac.py` derives by
default and `RACCompressor` asserts — this guarantees no condition is silently
truncated. Pass `--max-ctx` only if you want to override it (it is validated
against the same invariant).

The retriever (`utils/rag_utils.py`) and the oracle compressor
(`compression/rac_compressor.py`) are modality-agnostic, so the planned
audio/image-on-bGPT variant reuses them once `BGPTCompressor` grows a
byte-prefix path.

## Notes

Decoding intentionally pads the partially decoded sequence to the original full
length before each model forward pass. This mirrors the encoder prefill numerics
and should not be replaced with ordinary autoregressive decoding.

`eval_rac.py` defaults to `--no-decompress` for speed: the oracle selects
conditions from the model's (fp16) scored code lengths and reports the honest
arithmetic-coded size without a decode round-trip. Run a small slice **without**
`--no-decompress` to confirm `roundtrip_ok` before trusting a new configuration.

Prepare local datasets and pretrained models under the paths used by the
evaluation scripts, or pass explicit paths.
