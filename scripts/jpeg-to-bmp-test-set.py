#!/usr/bin/env python3
"""
Convert JPEG images to BMP for bGPT test set.
Reads from a source folder (e.g. tiny-imagenet jpeg folder) and writes BMPs
to a target folder used by BGPTCompress.
"""
import os
import sys

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bmp_utils import convert_images_to_bmp

# Default: tiny-imagenet-200 train jpeg -> bmp in same dataset tree
SOURCE_JPEG_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "tiny-imagenet-200", "train", "jpeg"
)
BMP_OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "tiny-imagenet-200", "bmp"
)


if __name__ == "__main__":
    print("=" * 60)
    print("JPEG to BMP test set conversion")
    print("=" * 60)
    print(f"Source (JPEG): {SOURCE_JPEG_FOLDER}")
    print(f"Output (BMP):  {BMP_OUTPUT_FOLDER}")
    print()
    convert_images_to_bmp(
        source_folder=SOURCE_JPEG_FOLDER,
        output_folder=BMP_OUTPUT_FOLDER,
    )
    print("Done. Use TEST_DATASET_IMAGE = 'datasets/tiny-imagenet-200/bmp/*.bmp' for BGPTCompress.")
