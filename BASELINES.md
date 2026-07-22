# Whole-sample compression baselines

The baseline suite uses the held-out split persisted by the RAC preparation
scripts. It never re-splits the source dataset at evaluation time and it never
compresses the JSON/Pickle container that stores the split.

## Data protocol

| Modality | Evaluation source | Canonical payload |
|---|---|---|
| text | `<database>/eval_docs.jsonl` | one complete document encoded as UTF-8 |
| image | `<database>/eval_samples.pkl` | one complete decoded RGB8 image |
| audio | `<database>/eval_samples.pkl` | one complete header-free mono PCM_U8 clip |

Image paths in `eval_samples.pkl` are dereferenced at evaluation time, so the
source image dataset must remain available. WAV containers are parsed only to
obtain the PCM frames and native sample rate; their headers are not part of the
payload.

All standalone codecs receive a complete logical sample. There is deliberately
no matched-window result: generic codecs may use all context in a document,
image, or clip.

`sample_id` follows the existing neural evaluators and is the position inside
the persisted held-out split (`doc000000`, `image000000`, ...). `source_index`
is also emitted and identifies the sample in the original pre-split dataset.
`input_sha256` freezes the actual canonical bytes seen by a run; this matters
especially for image databases, whose prepare artifact stores paths rather than
copies of the source pixels. Delta rows likewise record `reference_sha256` for
the selected full base sample.

## Methods

- All modalities: Zstandard `--ultra -22 -T1`, OpenZL serial, and the
  reference cmix CLI (which exposes no portable version flag).
- Image: optimized lossless PNG, lossless JPEG XL, and lossless WebP.
- Audio: FLAC over the canonical PCM_U8 samples.
- Delta: Zstandard `--patch-from`, bsdiff, and open-vcdiff.

OpenZL and cmix are external tools. cmix is selected by the modality defaults
but is exceptionally memory/runtime intensive; use an explicit `--codecs` list
to omit it for a shorter run. Missing tools are reported explicitly; they are
never omitted silently.

## Whole-reference delta protocol

Delta codecs receive a complete target sample and a complete base sample. The
existing chunk retriever is reused to propose references: retrieved chunk IDs
are mapped to their source sample IDs and de-duplicated to obtain the top-m
whole-sample references.

For images, the codec still receives whole-image RGB8. Only the retrieval query
is converted into the exact BGR/bottom-up/aligned BMP patches used when the
prepared byte index was built, then concatenated into one whole-image query.
This prevents an RGB/BGR representation mismatch without introducing a
matched-window compression track.

The selected reference ID is side information. With `N` complete base samples,
the default fixed coder charges:

```text
selected reference: ceil(log2(max(N, 2))) + 1 stop bit
no reference:        1 stop bit
```

For every target and delta codec, all proposed references are actually encoded.
Selection minimizes `patch_bits + reference_id_bits + stop_bit`. The adaptive
result also considers standalone Zstandard; choosing it transmits only the stop
symbol. Decoding is verified using only the selected base sample, transmitted
patch, and benchmark-shared sample metadata.

## Accounting

Every output row records the full artifact size, side-information bits, total
bits, compression/decompression wall time, codec version/settings, selected
reference (when applicable), and a byte-exact round-trip result. Aggregate BPB
and compression ratio are micro-averages from total original and compressed
bits, not averages of per-sample ratios.

`bpb` means bits per canonical input byte. Image rows additionally report
semantic `bpp = total_bits / (width * height)`.

Image patches are clipped to the source image rather than padded out to the
nominal rectangle, so the bGPT evaluators' `original_bytes` is the image's own
byte count and shares a denominator with these whole-image baselines. The one
residual difference is BMP's four-byte row alignment, which adds padding only
when a patch column's width is not a multiple of four pixels; it is zero for
every currently used geometry. Prefer BPP for image tables regardless, and
build them with `scripts/collect_results.py`, which puts every method on the
source image's `width * height` and reports the padding share it had to
account for.

The main track follows the repository convention that sample metadata is
benchmark-shared: generic/delta codecs and bGPT do not pay separately for image
dimensions, audio rate, or boundaries. PNG/JPEG XL/FLAC remain conservative in
that their standard artifacts still contain format framing. A future fully
self-contained track should charge equivalent serialized metadata to every
method, rather than subtracting standard-container headers.

The optional `zstd-dict` track trains only on complete prepared base samples.
Its dictionary size is reported separately as shared state and it is never
trained on held-out evaluation samples. The requested OpenZL track uses the
untrained `serial` profile; a future trained OpenZL track must follow the same
base-only/shared-state rule.

## Running

```bash
# All applicable standalone codecs. Unavailable optional tools are named and skipped.
python evaluation/eval_baselines.py \
  --database results/rac_c_db --modality text \
  --codecs zstd,openzl,cmix --allow-missing \
  --output results/baselines/text.csv \
  --summary-output results/baselines/text_summary.csv

# Whole-reference delta methods; use the same N/calibration tail as RAC.
python evaluation/eval_delta_baselines.py \
  --database results/rac_c_db --modality text --m 16 \
  --n-samples 200 --calib-samples 50 \
  --codecs zstd-patch,bsdiff,open-vcdiff \
  --output results/baselines/text_delta.csv
```

The convenience wrapper `scripts/eval_baselines.sh MODALITY DATABASE` runs both
commands. Required external executables are `zstd`, `zli` (OpenZL), `cmix`,
`cjxl`/`djxl`, `flac`, `bsdiff`/`bspatch`, and `vcdiff`, depending on modality.
