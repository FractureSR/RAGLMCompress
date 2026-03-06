# Image compression docs (image-comp)

All docs for the image compression pipeline use lowercase, hyphenated names with the `-image-comp` suffix.

| Doc | Purpose |
|-----|--------|
| **image-comp-setup.md** | How to set up env, weights, dataset, and run bGPT + baselines (SLURM or local). |
| **image-comp-analysis.md** | Modality, encoder comparison table, design decisions, steps. |
| **encoder-comparison-image-comp.md** | *Generated* by `evaluation/image-codec-baselines.py`. Latest numeric comparison (bGPT vs PNG, WebP, zstd) and pros/cons. Overwritten each time you run the baselines script. |
| **team-explanation-image-comp.md** | Short team-facing summary (progress, modality, data flow, results). |

Run outputs (reconstructed images, `results_summary.txt`) live in `output-img/` (gitignored). The only generated file under `docs/` is `encoder-comparison-image-comp.md`.
