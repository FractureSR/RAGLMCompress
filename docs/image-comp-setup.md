# Setup Guide — Image Compression (bGPT + Baselines)

This guide gets you from a fresh clone to running the image compression pipeline and baselines so you can easily run some experiments.

## 1. Repository and branch
```bash
# Clone (if you haven’t already)
git clone https://github.com/FractureSR/RAGLMCompress.git
cd RAGLMCompress

# Switch to the image-compression branch
git checkout -b image-compression
```

## 2. Python environment
Use either **Conda** (recommended on clusters) or **venv**.

### Option A: Conda
```bash
# Create env (if needed)
conda create -n ragllm python=3.10 -y
conda activate ragllm

# Install dependencies (from repo root)
cd /path/to/RAGLMCompress
pip install -r train/requirements.txt
pip install Pillow

# Optional: full bGPT deps for other tasks
pip install -r bgpt/requirements.txt
```

### Option B: venv
```bash
cd /path/to/RAGLMCompress
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r train/requirements.txt
pip install Pillow
# Optional: full bGPT deps for other tasks
pip install -r bgpt/requirements.txt
```

## 3. Requirements overview

| File | Purpose |
|------|--------|
| `train/requirements.txt` | Core deps for evaluation: torch, transformers, datasets, etc. |
| `bgpt/requirements.txt` | bGPT-specific (e.g. samplings); optional for run if using fallback in code |
| Pillow | Required for `utils/bmp_utils.py` (JPEG/BMP conversion and patch split/merge) |

## 4. Download pretrained weights (bGPT image)

The image compression pipeline expects the bGPT image checkpoint in the repo.

```bash
cd /path/to/RAGLMCompress
mkdir -p pretrained/bgpt

# Download (choose one)
# Option 1: wget
wget "https://huggingface.co/sander-wood/bgpt/resolve/main/weights-image.pth?download=1" \
  -O pretrained/bgpt/weights-image.pth

# Option 2: Hugging Face CLI (if installed and logged in)
huggingface-cli download sander-wood/bgpt weights-image.pth --local-dir pretrained/bgpt
```

After this, `evaluation/BGPTCompress.py` will load `./pretrained/bgpt/weights-image.pth` when running the image test. If the file is missing, the code falls back to a randomly initialized model (ratio will be &lt; 1).

## 5. Dataset: Tiny-ImageNet BMPs

Images are stored as BMPs for bGPT. Convert from JPEG once:

```bash
cd /path/to/RAGLMCompress

# Source: datasets/tiny-imagenet-200/train/jpeg/*.JPEG
# Output: datasets/tiny-imagenet-200/bmp/*.bmp
python scripts/jpeg-to-bmp-test-set.py
```

Default paths are set in the script; no extra config needed.

## 6. Running the bGPT image compression

### Local (GPU or CPU)

```bash
cd /path/to/RAGLMCompress
export PYTHONPATH=.
python -m evaluation.BGPTCompress
```

**Output folders:**
- **`temp-img/`** — Temporary working files: split patches (`split/`), per-patch compressed `.bin` files (`compressed/`), and decompressed patches (`decompressed/`). You can delete this after a run; it is recreated next time. (Gitignored.)
- **`output-img/`** — Final results: `reconstructed_*.bmp` (one per input image) and `results_summary.txt` (sizes and ratios). Keep for reporting. (Gitignored.) The baselines script writes the encoder comparison to `docs/encoder-comparison-image-comp.md`.

Outputs:
- `output-img/reconstructed_*.bmp` — reconstructed images  
- `output-img/results_summary.txt` — sizes and ratios  
- Console log with per-image and overall compression ratio  

### SLURM (recommended on shared clusters)

```bash
cd /path/to/RAGLMCompress
sbatch scripts/slurm-bgpt-image.sh
```

Then:
```bash
# Check job status
squeue -u $USER

# After it finishes, view logs (replace JOBID with the number from sbatch output)
cat slurm-bgpt-image-JOBID.out
cat slurm-bgpt-image-JOBID.err
cat output-img/results_summary.txt
```

Optional: in `scripts/slurm-bgpt-image.sh`, uncomment and set `#SBATCH --mail-user=...` and `#SBATCH --mail-type=END,FAIL` to get email when the job ends.

## 7. Running baseline codecs (PNG, WebP, zstd)

From repo root, run the **image-codec-baselines** script. It can generate baseline files for you (no need to copy shell commands) and/or write the comparison report.

**Options:**

| Command | What it does |
|--------|----------------|
| `python evaluation/image-codec-baselines.py` | Run codecs (convert, cwebp, zstd) then report. |
| `python evaluation/image-codec-baselines.py --run-baselines` | Only generate PNG/WebP/zstd files. |
| `python evaluation/image-codec-baselines.py --report-only` | Only load existing baselines + bGPT summary and write comparison. |

The script uses `subprocess` to call `convert` (ImageMagick), `cwebp`, and `zstd` when you use `--run-baselines` or the default. If a tool is not on your PATH, that codec is skipped. Run from repo root:

```bash
cd /path/to/RAGLMCompress
python evaluation/image-codec-baselines.py
```

This prints per-encoder ratios and overwrites `docs/encoder-comparison-image-comp.md`.

For analysis (modality, encoders, design decisions, steps), see **docs/image-comp-analysis.md**.
