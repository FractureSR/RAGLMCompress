# RAGLMCompress

Large-model-powered compression experiments for text, RAG-conditioned text,
images, and audio. Every compressor turns a frozen model's next-symbol
distribution into an arithmetic code. Evaluations report the emitted coder
bytes and, for RAC, the transmitted condition-id cost.

## Project Layout

- `compression/`: reusable compressor classes.
  - `LLMCompressor`: token-level compression with a causal LM. An optional
    `PromptContext` prepends conditioning tokens before the data; the coder
    skips those prefix positions.
  - `BGPTCompressor`: byte-level compression with bGPT. An optional per-segment
    `prefixes` prepends raw conditioning **bytes** before the payload. A fixed
    decoder-known WAV/BMP header and these conditions are model context only;
    the coder skips them — the byte-domain analog of `LLMCompressor`'s
    `PromptContext`.
  - `RACLLMCompressor` (`rac_llm_compressor.py`): retrieval-augmented compression
    — prepends the **raw tokens** of retrieved base chunks as the LM prefix and
    keeps the **oracle** choice (best of top-k by ideal model code length), or no
    condition if none beats its side-info cost. No training.
  - `RACBGPTCompressor` (`rac_bgpt_compressor.py`): the byte-domain analog of
    `RACLLMCompressor` — the same oracle-RAC over `BGPTCompressor`'s `prefixes`
    path instead of `LLMCompressor`'s token prompt (`RACBGPTCompressor` :
    `BGPTCompressor` :: `RACLLMCompressor` : `LLMCompressor`). Conditions are whole
    retrieved base byte chunks.
  - `rac_index.py`: index side-info coders (`FixedIndexCoder`,
    `CalibratedIndexCoder`) that price the transmitted base-chunk ids.
  - `base_compressor.py` / `types.py`: shared arithmetic-coding kernels and
    dataclasses (`CompressedData`, `LMScore`, `PromptContext`).
- `arithmetic_coder/`: verified range-coder wrapper and probability utilities.
- `utils/`: shared dataset, modality, retrieval (`rag_utils`), bGPT
  byte-preparation, and evaluation helpers. `prepare_rac_data_llm.py` /
  `prepare_rac_data_bgpt.py` build the RAC retrieval database for text / bytes.
- `evaluation/`: CLI benchmark scripts (`eval_llm`, `eval_bgpt`, `eval_rac_llm`,
  `eval_rac_bgpt`, image codec baselines).
- `scripts/`: dataset preparation and ready-to-run eval invocations.
- `bgpt/`: upstream bGPT model/training code.

## RAC (Retrieval-Augmented Compression)

Compress data by conditioning the frozen LM on the raw tokens of similar chunks
retrieved from a base corpus **using the data itself**. RAC ≠ RAG: we already
have the data `x`, retrieve with `x`, and re-encode `x` more cheaply. Because we
are compressing we can score each candidate under the same frozen model and keep
the **oracle** (best of top-k by summed NLL) — or no condition if none beats its
side-info cost. No training.

The same pipeline exists for **text** (LLM tokens) and **bytes** (audio/image on
bGPT) — one commutative diagram of file names:

| | base compressor | oracle RAC | build DB | evaluate |
|---|---|---|---|---|
| text | `LLMCompressor` | `RACLLMCompressor` | `prepare_rac_data_llm.py` | `eval_rac_llm.py` |
| bytes | `BGPTCompressor` | `RACBGPTCompressor` | `prepare_rac_data_bgpt.py` | `eval_rac_bgpt.py` |

Pipeline (text) — no training, no precomputed retrieval:

```
# 1. Build the retrieval database: fix a slice of the dataset as the base corpus,
#    chunk it, and index it (BM25 syntactic by default; no embedding model needed).
python utils/prepare_rac_data_llm.py \
    --dataset datasets/codeparrot_github_code/C.jsonl --n-docs 4000 \
    --base-frac 0.5 --chunk-size 512 --retriever bm25 \
    --model pretrained/SmolLM2-135M --out results/rac_c_db

# 2. Evaluate oracle RAC vs the no-condition baseline on the held-out docs,
#    chunked + retrieved live (mirrors eval_llm). Add --cascade / --calibrate.
python evaluation/eval_rac_llm.py --database results/rac_c_db \
    --model pretrained/SmolLM2-135M --m 16 --n-docs 200 --device cuda:0
```

Pipeline (bytes) — identical shape, over image patches / audio chunks. The base
corpus is chunked into fixed-size **byte** units, indexed by byte-k-gram BM25
(`make_bgpt_retriever`); the eval chunks + retrieves the held-out samples live
(mirrors `eval_bgpt`). bGPT receives a canonical decoder-known WAV/BMP header as
free model context, but the header is neither stored nor measured. Only payload
bytes are coded. Baseline payloads are 8000 B (audio) or 3072 B (image); RAC
reserves condition bytes inside that same body budget and does not pad missing
conditions.

Audio input is standardized before prepare/evaluation: dataset download scripts
must export uncompressed mono, 8-bit PCM WAV files while preserving the source
sample rate. Each WAV is validated and split into header-free PCM_U8 payloads by
byte count:

```
python scripts/download_ljspeech_data.py \
    --output datasets/ljspeech_wav --limit-mb 64

python utils/prepare_rac_data_bgpt.py \
    --dataset datasets/ljspeech_wav --modality audio \
    --n-samples 200 --base-frac 0.5 --audio-chunk-bytes 512 \
    --out results/rac_audio_db
```

The free WAV context is the checkpoint's fixed 44-byte 8 kHz training header;
it does not resample the native-rate PCM or become part of the output. A
decompressed payload is wrapped with its source sample rate when saved as WAV.

```
# 1. Build the byte retrieval database (image example).
python utils/prepare_rac_data_bgpt.py \
    --dataset datasets/clic2024/bmp --modality image --n-samples 400 \
    --base-frac 0.5 --image-patch-width 16 --image-patch-height 16 \
    --out results/rac_img_db

# 2. Evaluate oracle RAC over bGPT on the held-out samples the DB persisted.
python evaluation/eval_rac_bgpt.py --database results/rac_img_db \
    --model pretrained/bgpt/weights-image.pth --m 16 \
    --image-payload-width 32 --image-payload-height 24 --device cuda:0
```

For image databases, `eval_samples.pkl` stores held-out source paths rather than
duplicating image files, so the source dataset must remain at those paths.
bGPT RAC databases are format-versioned; rebuild an older database when eval
reports a schema mismatch rather than mixing header-bearing and raw chunks.

The chosen base ids are transmitted as side information (the decoder can't re-run
retrieval — the query is the unknown data); their bit cost (fixed, or a static
table built with `--calibrate`) is charged for honest bpb/ratio.

**Prefix budget.** Each condition is one full base chunk of `chunk_size` raw
bytes, and at most `max_cond` conditions are prepended per piece (`max_cond` =
`--cascade-max-cond` when cascade is enabled, or 1 otherwise). Evaluation sets
`max_ctx = chunk_size * max_cond`, and the RAC compressor asserts the same
invariant so no condition is silently truncated.

**Image shapes.** `--image-patch-width/height` on database preparation define
the retrieved condition shape. RAC evaluation requires both
`--image-payload-width` and `--image-payload-height`; baseline evaluation likewise
requires both `--image-patch-width` and `--image-patch-height`. The pipeline only
validates the supplied rectangle against BMP row alignment and the exact byte
budget. It never infers either dimension.

The retriever (`utils/rag_utils.py`) and the index coders
(`compression/rac_index.py`) are modality-agnostic and shared as-is between the
text and byte pipelines; only the featuriser differs (`make_text_retriever` vs
`make_bgpt_retriever`, byte-k-gram BM25).

## Notes

Decoding intentionally pads the partially decoded sequence to the original full
length before each model forward pass. This mirrors the encoder prefill numerics
and should not be replaced with ordinary autoregressive decoding.

The eval scripts support `--no-decompress` for speed: the oracle selects
conditions from the model's scored code lengths and reports the honest
arithmetic-coded size without a decode round-trip. Run a small slice **without**
`--no-decompress` to confirm `roundtrip_ok` before trusting a new configuration.
For bytes, decode must batch each unit exactly as it was coded — `eval_rac_bgpt`
and `RACBGPTCompressor` handle this — because bGPT's iterative decode is
batch-composition sensitive.

Prepare local datasets and pretrained models under the paths used by the
evaluation scripts, or pass explicit paths.
