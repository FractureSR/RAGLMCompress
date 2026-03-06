# Analysis: Image Modality, Encoders, Design Decisions, and Steps

This document summarizes the **modality**, **encoder comparison** (with pros/cons and results), **design decisions**, and **steps** taken for the image compression work, for team and mentor review.

## 1. Modality

- **Chosen modality:** **Image** (lossless compression of image bytes).
- **Data format:** Images are stored as **BMP** for the bGPT pipeline. Input JPEGs (Tiny-ImageNet) are converted to BMP once via `scripts/jpeg-to-bmp-test-set.py`; bGPT then sees raw BMP bytes.
- **Why BMP:** The bGPT image checkpoint and the existing `evaluation/BGPTCompress.py` pipeline are built around byte-level prediction on image data; BMP provides a simple, lossless raw format. Patches (e.g. 32×32) are split from each BMP, each patch is compressed/decompressed with bGPT + arithmetic coding, then merged back. The process is **lossless**: decompressed bytes are byte-identical to the original.
- **RAG:**: The image pipeline currently does **not** use RAG; the strong compression ratio comes from the pretrained bGPT image model.

## 2. Encoder comparison: results and pros/cons

All results below are on the **same** 5 Tiny-ImageNet BMPs (total original size **61,710 bytes**). Each encoder is **lossless** (decode recovers the original exactly, except for bGPT we verify 5/5 byte-identical reconstruction).

### Results table

| Encoder | Total compressed (B) | Overall ratio | Notes |
|--------|-----------------------|---------------|--------|
| **bGPT** (weights-image.pth) | 25,333 | **2.4360×** | Learned; patch-level + byte-level model + arithmetic coding. |
| **WebP lossless** | 30,156 | **2.0464×** | Modern image codec; BMP→PNG→WebP. |
| **PNG** | 43,700 | **1.4121×** | Classic lossless; DEFLATE-based. |
| **zstd** (on raw BMP) | 55,696 | **1.1080×** | Generic compressor; no image structure. |

Higher ratio = better compression (smaller output for the same input).

### Pros and cons by encoder

| Encoder | Pros | Cons |
|--------|------|------|
| **bGPT** | Learned byte-level model; best ratio here. Same architecture supports other modalities (text, audio). Fits existing arithmetic-coding framework. | Requires large checkpoint and GPU for speed; slower and more complex than classic codecs. |
| **WebP lossless** | Often beats PNG; widely supported in modern stacks. | Less universal than PNG in some tooling; still hand-designed, image-specific. |
| **PNG** | Mature, widely supported, fast; no extra model. | Hand-designed; outperformed by bGPT and WebP here. Image-specific. |
| **zstd** | Fast, strong generic compressor; works on any bytes. | Ignores image structure; lowest ratio here. Not semantically aware. |

---

## 3. Design decisions

- **Dataset:** Tiny-ImageNet subset (5 images, one class: `n02795169_0` … `n02795169_4`) to keep runs fast and reproducible. Conversion JPEG→BMP is scripted and documented.
- **Baselines (PNG, WebP, zstd):** Implemented as **offline** steps (external tools produce files in `baselines/`); `evaluation/image-codec-baselines.py` only measures sizes and writes the comparison. This avoids hard dependency on specific system binaries in the main code path and keeps the script portable.
- **Losslessness:** All reported numbers are for **lossless** compression. bGPT reconstruction is checked byte-by-byte (5/5 match). No lossy mode is used.
- **SLURM:** Image compression runs are submitted via `scripts/slurm-bgpt-image.sh` so heavy work runs on compute nodes; logs and `output-img/results_summary.txt` are used for reporting.
