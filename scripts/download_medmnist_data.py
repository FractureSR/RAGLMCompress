#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


RGB_DATASETS = {"pathmnist", "dermamnist", "retinamnist", "bloodmnist"}
RECORD_ID = "10519652"


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-c", "--show-progress", "-O", str(output), url],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an RGB MedMNIST dataset and export it as BMP."
    )
    parser.add_argument("dataset", choices=sorted(RGB_DATASETS))
    parser.add_argument("--size", type=int, choices=[28, 64, 128, 224], default=28)
    parser.add_argument("--output", type=Path, default=Path("medmnist_bmp"))
    parser.add_argument("--npz", type=Path, help="Use an existing NPZ file.")
    args = parser.parse_args()

    suffix = "" if args.size == 28 else f"_{args.size}"
    filename = f"{args.dataset}{suffix}.npz"
    npz_path = args.npz or args.output / "_downloads" / filename

    if not npz_path.exists():
        url = f"https://zenodo.org/records/{RECORD_ID}/files/{filename}"
        print(f"Downloading {url}")
        download(url, npz_path)

    export_root = args.output / f"{args.dataset}_{args.size}"

    # Temporarily extract .npy files to support memory mapping for large datasets.
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(npz_path) as archive:
            archive.extractall(tmp)

        tmp = Path(tmp)

        for split in ("train", "val", "test"):
            images = np.load(tmp / f"{split}_images.npy", mmap_mode="r")
            labels = np.load(tmp / f"{split}_labels.npy", mmap_mode="r")

            split_dir = export_root / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for i, (image, label) in enumerate(zip(images, labels)):
                label = int(np.asarray(label).reshape(-1)[0])
                output_path = split_dir / f"{i:06d}_label_{label}.bmp"

                Image.fromarray(
                    np.asarray(image, dtype=np.uint8),
                    "RGB",
                ).save(output_path)

            print(f"{split}: exported {len(images)} images to {split_dir}")


if __name__ == "__main__":
    main()