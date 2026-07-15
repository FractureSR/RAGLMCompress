#!/usr/bin/env python3
import argparse
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"


def download(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-c", "--show-progress", "-O", str(output), URL],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download EuroSAT RGB and convert all images to BMP."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eurosat_bmp"),
        help="Output directory.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        help="Use an existing EuroSAT_RGB.zip instead of downloading.",
    )
    args = parser.parse_args()

    zip_path = args.zip or args.output / "_downloads" / "EuroSAT_RGB.zip"

    if not zip_path.exists():
        print(f"Downloading {URL}")
        download(zip_path)

    count = 0

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            path = PurePosixPath(member)

            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

            # Remove the top-level EuroSAT_RGB directory.
            parts = path.parts
            if parts[0] == "EuroSAT_RGB":
                parts = parts[1:]

            output_path = args.output.joinpath(*parts).with_suffix(".bmp")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member) as source:
                with Image.open(source) as image:
                    image.convert("RGB").save(output_path, format="BMP")

            count += 1

    print(f"Exported {count} images to {args.output}")


if __name__ == "__main__":
    main()