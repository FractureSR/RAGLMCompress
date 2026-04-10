"""
Image codec baselines: compare PNG, WebP, zstd (lossless).

Run from repo root.

Behavior:
  - Generate PNG baselines with ffmpeg using high lossless compression settings.
  - Generate WebP baselines with ffmpeg using lossless mode and highest effort.
  - Generate zstd baselines on raw BMP files with ultra compression.
  - Then print per-file and overall compression statistics.

Requirements on PATH:
  - ffmpeg
  - zstd
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Run from repo root so relative paths work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PathsConfig:
    """Centralized path configuration for image codec baseline experiments."""
    dataset_root: str = "datasets/clic2024"
    baseline_root: str = "baselines"

    @property
    def bmp_glob(self) -> str:
        return os.path.join(self.dataset_root, "bmp", "*.bmp")

    @property
    def png_dir(self) -> str:
        return os.path.join(self.baseline_root, "png")

    @property
    def webp_dir(self) -> str:
        return os.path.join(self.baseline_root, "webp")

    @property
    def zstd_dir(self) -> str:
        return os.path.join(self.baseline_root, "zstd")


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
        print(f"{base}: {original_size} -> {comp_size} bytes, ratio={ratio:.4f}")

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
    print()

    return results, overall


def compute_png_baseline(config: PathsConfig) -> Tuple[List[Dict], Dict]:
    """Compute lossless PNG baseline on the BMP images."""
    return _compute_baseline_generic("PNG", config.bmp_glob, config.png_dir, ".png")


def compute_webp_baseline(config: PathsConfig) -> Tuple[List[Dict], Dict]:
    """Compute lossless WebP baseline on the BMP images."""
    return _compute_baseline_generic("WebP lossless", config.bmp_glob, config.webp_dir, ".webp")


def compute_zstd_baseline(config: PathsConfig) -> Tuple[List[Dict], Dict]:
    """Compute zstd baseline (on raw BMP bytes) on the BMP images."""
    return _compute_baseline_generic("zstd", config.bmp_glob, config.zstd_dir, ".zst")


def _which(cmd: str) -> str:
    """Return full path for cmd if on PATH, else empty string."""
    return shutil.which(cmd) or ""


def _run_command(cmd: List[str], desc: str) -> None:
    """Run a subprocess command and raise a readable error on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        stderr = (r.stderr or "").strip()
        stdout = (r.stdout or "").strip()
        msg = stderr if stderr else stdout
        raise RuntimeError(f"{desc} failed: {msg}")


def run_baselines(config: PathsConfig) -> None:
    """
    Generate baseline files using ffmpeg (PNG/WebP) and zstd.

    PNG:
      ffmpeg -i input.bmp -c:v png -compression_level 9 output.png

    WebP:
      ffmpeg -i input.bmp -c:v libwebp -lossless 1 -compression_level 6 -quality 100 output.webp

    zstd:
      zstd -q --ultra -22 input.bmp -o output.zst
    """
    bmp_files = sorted(glob.glob(config.bmp_glob))
    if not bmp_files:
        print(f"No BMP files found for {config.bmp_glob}.")
        return

    os.makedirs(config.png_dir, exist_ok=True)
    os.makedirs(config.webp_dir, exist_ok=True)
    os.makedirs(config.zstd_dir, exist_ok=True)

    ffmpeg_exe = _which("ffmpeg")
    zstd_exe = _which("zstd")

    if not ffmpeg_exe:
        print("Skipping PNG/WebP: 'ffmpeg' not on PATH.")
    else:
        # PNG with ffmpeg
        print("Generating PNG baselines (ffmpeg, lossless, compression_level=9)...")
        for bmp in bmp_files:
            base = os.path.splitext(os.path.basename(bmp))[0]
            png = os.path.join(config.png_dir, f"{base}.png")
            try:
                _run_command(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-loglevel", "error",
                        "-i", bmp,
                        "-c:v", "png",
                        "-compression_level", "9",
                        png,
                    ],
                    f"ffmpeg PNG encode for {bmp}",
                )
            except Exception as e:
                print(f"  Warning: {e}")
        print("  Done.")

        # WebP with ffmpeg
        print("Generating WebP baselines (ffmpeg, lossless, compression_level=6)...")
        for bmp in bmp_files:
            base = os.path.splitext(os.path.basename(bmp))[0]
            webp = os.path.join(config.webp_dir, f"{base}.webp")
            try:
                _run_command(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-loglevel", "error",
                        "-i", bmp,
                        "-c:v", "libwebp",
                        "-lossless", "1",
                        "-compression_level", "6",
                        "-quality", "100",
                        webp,
                    ],
                    f"ffmpeg WebP encode for {bmp}",
                )
            except Exception as e:
                print(f"  Warning: {e}")
        print("  Done.")

    # zstd on raw BMP bytes
    if zstd_exe:
        print("Generating zstd baselines (--ultra -22)...")
        for bmp in bmp_files:
            base = os.path.splitext(os.path.basename(bmp))[0]
            zst = os.path.join(config.zstd_dir, f"{base}.zst")
            r = subprocess.run(
                [zstd_exe, "-q", "--ultra", "-22", bmp, "-o", zst],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"  Warning: zstd failed for {bmp}: {r.stderr or r.stdout}")
        print("  Done.")
    else:
        print("Skipping zstd: 'zstd' not on PATH.")

    print()


def print_summary_table(stats: Dict[str, Dict]) -> None:
    """Print a compact summary table for all available baselines."""
    print("=" * 60)
    print("Baseline summary")
    print("=" * 60)
    print(f"{'Codec':<15}{'Original':>15}{'Compressed':>15}{'Ratio':>12}")
    print("-" * 60)

    for codec_name, overall in stats.items():
        if overall is None:
            print(f"{codec_name:<15}{'N/A':>15}{'N/A':>15}{'N/A':>12}")
        else:
            print(
                f"{codec_name:<15}"
                f"{int(overall['total_original_bytes']):>15}"
                f"{int(overall['total_compressed_bytes']):>15}"
                f"{overall['overall_ratio']:>12.4f}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image codec baselines: run and print results (PNG, WebP, zstd)."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/clic2024",
        help="Dataset root directory, e.g. datasets/clic2024",
    )
    parser.add_argument(
        "--baseline-root",
        default="baselines",
        help="Root directory for generated baseline files.",
    )
    args = parser.parse_args()

    # Ensure we're in repo root
    os.chdir(_REPO_ROOT)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    config = PathsConfig(
        dataset_root=args.dataset_root,
        baseline_root=args.baseline_root,
    )

    print("=" * 60)
    print("Running baseline codecs (ffmpeg for PNG/WebP, zstd for .zst)")
    print("=" * 60)
    run_baselines(config)

    png_overall = None
    webp_overall = None
    zstd_overall = None

    print("=" * 60)
    print("Computing baseline statistics")
    print("=" * 60)

    try:
        _, png_overall = compute_png_baseline(config)
    except Exception as e:
        print(f"Warning: could not compute PNG baseline: {e}\n")

    try:
        _, webp_overall = compute_webp_baseline(config)
    except Exception as e:
        print(f"Warning: could not compute WebP baseline: {e}\n")

    try:
        _, zstd_overall = compute_zstd_baseline(config)
    except Exception as e:
        print(f"Warning: could not compute zstd baseline: {e}\n")

    print_summary_table(
        {
            "PNG": png_overall,
            "WebP": webp_overall,
            "zstd": zstd_overall,
        }
    )


if __name__ == "__main__":
    main()