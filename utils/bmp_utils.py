import os
from glob import glob
from PIL import Image
from tqdm import tqdm


def convert_images_to_bmp(source_folder: str, output_folder: str):
    """
    Convert images of any format (PNG, JPEG, etc.) to BMP format.

    :param source_folder: Folder containing source images
    :param output_folder: Folder to save converted BMP images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Support common image formats
    supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(source_folder, ext)))
        image_files.extend(glob(os.path.join(source_folder, ext.upper())))

    if not image_files:
        print(f"No image files found in {source_folder}")
        return

    print(f"Converting {len(image_files)} images to BMP format...")
    for filepath in tqdm(image_files):
        try:
            with Image.open(filepath) as img:
                # Convert RGBA to RGB
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                elif img.mode not in ["RGB", "L"]:  # L is grayscale
                    img = img.convert("RGB")

                # Get original filename without extension
                filename = os.path.basename(filepath)
                name_without_ext = os.path.splitext(filename)[0]

                # Save as BMP
                output_path = os.path.join(output_folder, f"{name_without_ext}.bmp")
                img.save(output_path, "BMP")

        except Exception as e:
            print(f"Error converting {filepath}: {e}")

    print(f"Conversion complete. BMP files saved to {output_folder}")


def split_bmp_to_patches(source_folder: str, output_folder: str, patch_size: int):
    """
    Split BMP images into square patches.

    :param source_folder: Folder containing BMP images
    :param output_folder: Folder to save patches (each image gets its own subfolder)
    :param patch_size: Size of square patches (e.g., 32 means 32x32 patches)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    bmp_files = glob(os.path.join(source_folder, "*.bmp"))

    if not bmp_files:
        print(f"No BMP files found in {source_folder}")
        return

    print(f"Splitting {len(bmp_files)} BMP images into {patch_size}x{patch_size} patches...")
    for filepath in tqdm(bmp_files):
        try:
            with Image.open(filepath) as img:
                width, height = img.size

                # Create subfolder for this image's patches
                filename = os.path.basename(filepath)
                image_name = os.path.splitext(filename)[0]
                patches_subfolder = os.path.join(output_folder, image_name)
                if not os.path.exists(patches_subfolder):
                    os.makedirs(patches_subfolder)

                # Split into patches
                for row_idx, y in enumerate(range(0, height, patch_size)):
                    for col_idx, x in enumerate(range(0, width, patch_size)):
                        # Calculate actual patch size (may be smaller at edges)
                        actual_width = min(patch_size, width - x)
                        actual_height = min(patch_size, height - y)

                        box = (x, y, x + actual_width, y + actual_height)
                        patch = img.crop(box)

                        # Save patch with row_col naming
                        patch_filename = f"patch_{row_idx}_{col_idx}.bmp"
                        patch_path = os.path.join(patches_subfolder, patch_filename)
                        patch.save(patch_path, "BMP")

        except Exception as e:
            print(f"Error splitting {filepath}: {e}")

    print(f"Splitting complete. Patches saved to {output_folder}")


def merge_patches_to_bmp(patches_folder: str, output_path: str, patch_size: int):
    """
    Merge square patches back into a single BMP image.

    :param patches_folder: Folder containing patch files (patch_row_col.bmp format)
    :param output_path: Path to save the merged BMP image
    :param patch_size: Base size of patches (for regular patches, edge patches may be smaller)
    """
    if not os.path.isdir(patches_folder):
        print(f"Error: Directory {patches_folder} does not exist.")
        return

    # Find all BMP patch files
    patch_files = [f for f in os.listdir(patches_folder) if f.lower().endswith(".bmp")]
    if not patch_files:
        print(f"Error: No BMP files found in {patches_folder}")
        return

    # Parse patch positions and collect info
    patch_info = {}
    max_row, max_col = -1, -1

    for filename in patch_files:
        try:
            # Expected format: patch_row_col.bmp
            parts = os.path.splitext(filename)[0].split("_")
            if len(parts) != 3 or parts[0] != "patch":
                print(f"Warning: Skipping {filename} - invalid naming format")
                continue

            row = int(parts[1])
            col = int(parts[2])

            max_row = max(max_row, row)
            max_col = max(max_col, col)

            patch_path = os.path.join(patches_folder, filename)
            patch_info[(row, col)] = patch_path

        except (IndexError, ValueError) as e:
            print(f"Warning: Skipping {filename} - cannot parse row/col: {e}")
            continue

    if not patch_info:
        print("Error: No valid patch files found")
        return

    # Determine total image size by examining edge patches
    max_width = 0
    max_height = 0

    for row in range(max_row + 1):
        for col in range(max_col + 1):
            if (row, col) not in patch_info:
                print(f"Warning: Missing patch at position ({row}, {col})")
                continue

            with Image.open(patch_info[(row, col)]) as patch:
                expected_x = col * patch_size
                expected_y = row * patch_size

                max_width = max(max_width, expected_x + patch.width)
                max_height = max(max_height, expected_y + patch.height)

    # Create merged image
    merged_image = Image.new("RGB", (max_width, max_height))

    # Paste all patches
    for (row, col), patch_path in patch_info.items():
        with Image.open(patch_path) as patch:
            paste_x = col * patch_size
            paste_y = row * patch_size
            merged_image.paste(patch, (paste_x, paste_y))

    # Save merged image
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_image.save(output_path, "BMP")
    print(f"Merged image saved to: {output_path}")


def resize_bmp(input_dir: str, output_dir: str, target_size: tuple):
    """
    Resize all BMP images in a directory.

    :param input_dir: Directory containing input BMP images
    :param output_dir: Directory to save resized BMP images
    :param target_size: Tuple of (width, height) for the target size
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    bmp_files = glob(os.path.join(input_dir, "*.bmp"))
    bmp_files += glob(os.path.join(input_dir, "*.BMP"))

    if not bmp_files:
        print(f"No BMP files found in {input_dir}")
        return

    print(f"Found {len(bmp_files)} BMP images in {input_dir}")
    print(f"Resizing all images to: {target_size}")

    for input_path in tqdm(bmp_files, desc="Resizing BMP images"):
        try:
            with Image.open(input_path) as img:
                original_size = img.size

                # 保证输出为 BMP 常见模式
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                elif img.mode not in ["RGB", "L"]:
                    img = img.convert("RGB")

                resized_img = img.resize(target_size, Image.LANCZOS)

                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, filename)

                resized_img.save(output_path, "BMP")
                print(f"Resized {filename}: {original_size} -> {target_size}")

        except Exception as e:
            print(f"Error resizing {input_path}: {e}")

    print(f"All resized BMP images have been saved to: {output_dir}")


if __name__ == "__main__":
    SOURCE_IMAGE_FOLDER = "datasets/clic2024/png"
    BMP_OUTPUT_FOLDER = "datasets/clic2024/bmp"
    RESIZED_BMP_OUTPUT_FOLDER = "datasets/clic2024/bmp_resized"

    # Convert all images to BMP
    """
    print("=" * 60)
    print("Converting images to BMP format")
    print("=" * 60)
    convert_images_to_bmp(
        source_folder=SOURCE_IMAGE_FOLDER,
        output_folder=BMP_OUTPUT_FOLDER
    )
    """

    print("=" * 60)
    print("Resizing images")
    print("=" * 60)
    resize_bmp(
        input_dir=BMP_OUTPUT_FOLDER,
        output_dir=RESIZED_BMP_OUTPUT_FOLDER,
        target_size=(64, 64)
    )
