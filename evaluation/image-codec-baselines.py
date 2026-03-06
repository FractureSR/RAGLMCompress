"""
Image codec baselines: compare bGPT vs PNG, WebP, zstd (lossless).

Run from repo root. Options:
  --run-baselines   Generate baseline files (PNG, WebP, zstd) via system tools.
  --report-only     Only load existing baselines and bGPT summary, write comparison.
  (default)         Run baselines then report (run + report).

Uses subprocess to call convert (ImageMagick), cwebp, zstd so users don't need
to copy shell commands. Requires these on PATH when using --run-baselines.
"""
import argparse
import glob
import os
import subprocess
import sys
from typing import Dict, List, Tuple

# Run from repo root so relative paths work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _compute_baseline_generic(
    name: str,
    bmp_glob: str,
    out_dir: str,
    ext: str,
) -> Tuple[List[Dict], Dict]:
    """
    Generic helper to compute a baseline given precomputed files.

    - `name`: label to print (e.g. 'PNG', 'WebP', 'zstd')
    - `bmp_glob`: glob for BMP sources
    - `out_dir`: directory containing compressed files
    - `ext`: extension of compressed files (e.g. '.png', '.webp', '.zst')
    """
    bmp_files = sorted(glob.glob(bmp_glob))
    if not bmp_files:
        raise FileNotFoundError(f"No BMP files found matching pattern: {bmp_glob}")

    results: List[Dict] = []
    total_original = 0
    total_compressed = 0

    print(f"{name} baseline:")
    for bmp in bmp_files:
        base = os.path.splitext(os.path.basename(bmp))[0]
        out_path = os.path.join(out_dir, f"{base}{ext}")
        if not os.path.isfile(out_path):
            raise FileNotFoundError(f"Expected {name} file not found for {bmp}: {out_path}")

        original_size = os.path.getsize(bmp)
        comp_size = os.path.getsize(out_path)
        ratio = original_size / comp_size if comp_size else 0.0

        total_original += original_size
        total_compressed += comp_size

        results.append(
            {
                "name": base,
                "original_bytes": original_size,
                "compressed_bytes": comp_size,
                "ratio": ratio,
            }
        )
        print(
            f"{base}: {original_size} -> {comp_size} bytes, "
            f"ratio={ratio:.4f}"
        )

    overall_ratio = total_original / total_compressed if total_compressed else 0.0
    overall = {
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
        "overall_ratio": overall_ratio,
    }
    print(
        f"Total: {total_original} -> {total_compressed} bytes, "
        f"overall ratio={overall_ratio:.4f}"
    )

    return results, overall


def compute_png_baseline(
    bmp_glob: str = "datasets/tiny-imagenet-200/bmp/*.bmp",
    png_dir: str = "baselines/png",
) -> Tuple[List[Dict], Dict]:
    """Compute lossless PNG baseline on the Tiny-ImageNet BMP images."""
    return _compute_baseline_generic("PNG", bmp_glob, png_dir, ".png")


def compute_webp_baseline(
    bmp_glob: str = "datasets/tiny-imagenet-200/bmp/*.bmp",
    webp_dir: str = "baselines/webp",
) -> Tuple[List[Dict], Dict]:
    """Compute lossless WebP baseline on the Tiny-ImageNet BMP images."""
    return _compute_baseline_generic("WebP lossless", bmp_glob, webp_dir, ".webp")


def compute_zstd_baseline(
    bmp_glob: str = "datasets/tiny-imagenet-200/bmp/*.bmp",
    zstd_dir: str = "baselines/zstd",
) -> Tuple[List[Dict], Dict]:
    """Compute zstd baseline (on raw BMP bytes) on the Tiny-ImageNet BMP images."""
    return _compute_baseline_generic("zstd", bmp_glob, zstd_dir, ".zst")


def load_bgpt_summary(
    summary_path: str = "output-img/results_summary.txt",
) -> Dict:
    """
    Parse the existing bGPT summary file produced by BGPTCompress.

    Expects lines like:
      Total original size: 61710 bytes
      Total compressed size: 25333 bytes
      Overall compression ratio: 2.4360
    """
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"bGPT summary file not found: {summary_path}")

    data: Dict[str, float] = {}
    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Total original size:"):
                parts = line.split()
                data["total_original_bytes"] = float(parts[3])
            elif line.startswith("Total compressed size:"):
                parts = line.split()
                data["total_compressed_bytes"] = float(parts[3])
            elif line.startswith("Overall compression ratio:"):
                parts = line.split()
                data["overall_ratio"] = float(parts[3])

    if not data:
        raise ValueError(f"Could not parse bGPT summary from {summary_path}")

    return data


def write_encoder_comparison_markdown(
    output_path: str = "docs/encoder-comparison-image-comp.md",
    bgpt_stats: Dict = None,
    png_stats: Dict = None,
    webp_stats: Dict = None,
    zstd_stats: Dict = None,
) -> None:
    """
    Write a small markdown file comparing bGPT vs PNG on the Tiny-ImageNet subset,
    including pros and cons for each encoder.
    """
    lines = []
    lines.append("# Encoder comparison (image-comp)")
    lines.append("")
    lines.append("Lossless image encoders: bGPT vs PNG, WebP, zstd.")
    lines.append("")
    if bgpt_stats:
        lines.append("## bGPT (Image checkpoint `weights-image.pth`)")
        lines.append("")
        lines.append(f"- **Total original size**: {int(bgpt_stats['total_original_bytes'])} bytes")
        lines.append(f"- **Total compressed size**: {int(bgpt_stats['total_compressed_bytes'])} bytes")
        lines.append(f"- **Overall compression ratio**: **{bgpt_stats['overall_ratio']:.4f}×**")
        lines.append("")
        lines.append("- **Pros**:")
        lines.append("  - Learned byte-level model tuned to images; very strong compression.")
        lines.append("  - Same architecture can also handle other modalities (text, audio, CPU states).")
        lines.append("  - Integrates naturally with the existing arithmetic-coding framework.")
        lines.append("- **Cons**:")
        lines.append("  - Requires shipping a large model checkpoint and GPU/accelerator for speed.")
        lines.append("  - Encode/decode is slower and more complex than classic codecs.")
        lines.append("")

    if png_stats:
        lines.append("## PNG (classic lossless codec)")
        lines.append("")
        lines.append(f"- **Total original size**: {int(png_stats['total_original_bytes'])} bytes")
        lines.append(f"- **Total compressed size**: {int(png_stats['total_compressed_bytes'])} bytes")
        lines.append(f"- **Overall compression ratio**: **{png_stats['overall_ratio']:.4f}×**")
        lines.append("")
        lines.append("- **Pros**:")
        lines.append("  - Very mature and widely supported lossless image format.")
        lines.append("  - Fast encode/decode; no large model or extra dependencies.")
        lines.append("- **Cons**:")
        lines.append("  - Hand-designed; can be outperformed by strong learned models like bGPT.")
        lines.append("  - Image-specific; does not unify with other modalities.")
        lines.append("")

    if webp_stats:
        lines.append("## WebP lossless (modern image codec)")
        lines.append("")
        lines.append(f"- **Total original size**: {int(webp_stats['total_original_bytes'])} bytes")
        lines.append(f"- **Total compressed size**: {int(webp_stats['total_compressed_bytes'])} bytes")
        lines.append(f"- **Overall compression ratio**: **{webp_stats['overall_ratio']:.4f}×**")
        lines.append("")
        lines.append("- **Pros**:")
        lines.append("  - Often better compression than PNG in lossless mode.")
        lines.append("  - Supported by many modern browsers and tools.")
        lines.append("- **Cons**:")
        lines.append("  - Less universally supported than PNG in some tooling.")
        lines.append("  - Still image-specific and hand-designed.")
        lines.append("")

    if zstd_stats:
        lines.append("## zstd (generic compressor on BMP bytes)")
        lines.append("")
        lines.append(f"- **Total original size**: {int(zstd_stats['total_original_bytes'])} bytes")
        lines.append(f"- **Total compressed size**: {int(zstd_stats['total_compressed_bytes'])} bytes")
        lines.append(f"- **Overall compression ratio**: **{zstd_stats['overall_ratio']:.4f}×**")
        lines.append("")
        lines.append("- **Pros**:")
        lines.append("  - Very fast, strong general-purpose compressor.")
        lines.append("  - No image-specific assumptions; works on any byte stream.")
        lines.append("- **Cons**:")
        lines.append("  - Ignores image structure; usually worse than image-specific codecs or bGPT on images.")
        lines.append("  - Not directly comparable to semantically-aware models.")
        lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nEncoder comparison written to: {output_path}")


def _which(cmd: str) -> str:
    """Return path for cmd if on PATH, else empty string."""
    return subprocess.run(
        ["which", cmd],
        capture_output=True,
        text=True,
    ).stdout.strip() or ""


def run_baselines(
    bmp_glob: str = "datasets/tiny-imagenet-200/bmp/*.bmp",
    png_dir: str = "baselines/png",
    webp_dir: str = "baselines/webp",
    zstd_dir: str = "baselines/zstd",
) -> None:
    """
    Generate baseline files by running system tools (convert, cwebp, zstd).
    Call from repo root. Skips a codec if its tool is not on PATH.
    """
    bmp_files = sorted(glob.glob(bmp_glob))
    if not bmp_files:
        print(f"No BMP files found for {bmp_glob}; run scripts/jpeg-to-bmp-test-set.py first.")
        return

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(webp_dir, exist_ok=True)
    os.makedirs(zstd_dir, exist_ok=True)

    convert_exe = _which("convert")
    cwebp_exe = _which("cwebp")
    zstd_exe = _which("zstd")

    # PNG: ImageMagick convert
    if convert_exe:
        print("Generating PNG baselines (ImageMagick convert)...")
        for bmp in bmp_files:
            base = os.path.splitext(os.path.basename(bmp))[0]
            png = os.path.join(png_dir, f"{base}.png")
            r = subprocess.run([convert_exe, bmp, png], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  Warning: convert failed for {bmp}: {r.stderr or r.stdout}")
        print("  Done.")
    else:
        print("Skipping PNG: 'convert' (ImageMagick) not on PATH.")

    # WebP: cwebp -lossless (from PNG; cwebp often doesn't accept BMP)
    if cwebp_exe and convert_exe:
        print("Generating WebP baselines (cwebp -lossless from PNG)...")
        for bmp in bmp_files:
            base = os.path.splitext(os.path.basename(bmp))[0]
            png = os.path.join(png_dir, f"{base}.png")
            webp = os.path.join(webp_dir, f"{base}.webp")
            if not os.path.isfile(png):
                print(f"  Warning: PNG missing {png}, skipping WebP.")
                continue
            r = subprocess.run([cwebp_exe, "-lossless", png, "-o", webp], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  Warning: cwebp failed for {png}: {r.stderr or r.stdout}")
        print("  Done.")
    elif not cwebp_exe:
        print("Skipping WebP: 'cwebp' not on PATH.")

    # zstd on raw BMP
    if zstd_exe:
        print("Generating zstd baselines...")
        for bmp in bmp_files:
            base = os.path.splitext(os.path.basename(bmp))[0]
            zst = os.path.join(zstd_dir, f"{base}.zst")
            r = subprocess.run([zstd_exe, "-q", "-19", bmp, "-o", zst], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  Warning: zstd failed for {bmp}: {r.stderr or r.stdout}")
        print("  Done.")
    else:
        print("Skipping zstd: 'zstd' not on PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image codec baselines: run and/or report (bGPT vs PNG, WebP, zstd)."
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Generate baseline files (PNG, WebP, zstd) using system tools.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only load existing baselines and bGPT summary, write comparison; do not run codecs.",
    )
    args = parser.parse_args()

    # Ensure we're in repo root
    os.chdir(_REPO_ROOT)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    # Default: both run and report. --run-baselines only: run codecs. --report-only: only report.
    do_run = args.run_baselines or (not args.report_only and not args.run_baselines)
    do_report = args.report_only or (not args.run_baselines)

    if do_run:
        print("=" * 60)
        print("Running baseline codecs (convert, cwebp, zstd)")
        print("=" * 60)
        run_baselines()
        print()

    if do_report:
        print("=" * 60)
        print("Loading bGPT summary and baseline stats, writing comparison")
        print("=" * 60)
        try:
            bgpt_stats = load_bgpt_summary()
            print("Loaded bGPT stats from output-img/results_summary.txt")
        except Exception as e:
            print(f"Warning: could not load bGPT summary: {e}")
            bgpt_stats = None

        try:
            _, png_overall = compute_png_baseline()
        except Exception as e:
            print(f"Warning: could not compute PNG baseline: {e}")
            png_overall = None

        try:
            _, webp_overall = compute_webp_baseline()
        except Exception as e:
            print(f"Warning: could not compute WebP baseline: {e}")
            webp_overall = None

        try:
            _, zstd_overall = compute_zstd_baseline()
        except Exception as e:
            print(f"Warning: could not compute zstd baseline: {e}")
            zstd_overall = None

        write_encoder_comparison_markdown(
            output_path="docs/encoder-comparison-image-comp.md",
            bgpt_stats=bgpt_stats,
            png_stats=png_overall,
            webp_stats=webp_overall,
            zstd_stats=zstd_overall,
        )


if __name__ == "__main__":
    main()

