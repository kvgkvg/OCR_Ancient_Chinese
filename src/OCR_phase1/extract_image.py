import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_images_source_dir, get_images_with_id_dir


def extract_images_with_id(source_dir, target_dir):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    target_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise ValueError(f"Source directory not found: {source_dir}")

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}

    copied_count = 0

    for subdir in source_path.iterdir():
        if not subdir.is_dir():
            continue

        subfolder_name = subdir.name

        for image_file in subdir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                original_name = image_file.stem
                extension = image_file.suffix

                new_name = f"{subfolder_name}_{original_name}{extension}"

                target_file = target_path / new_name

                shutil.copy2(image_file, target_file)
                copied_count += 1

    return copied_count


def split_images_into_parts(source_dir, num_parts=4):
    source_path = Path(source_dir)

    if not source_path.exists():
        raise ValueError(f"Source directory not found: {source_dir}")

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}

    all_images = sorted([
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    total_images = len(all_images)
    if total_images == 0:
        raise ValueError("No images found in source directory")

    images_per_part = total_images // num_parts
    remainder = total_images % num_parts

    start_idx = 0
    for i in range(num_parts):
        part_name = f"part_{i + 1}"
        part_dir = source_path / part_name
        part_dir.mkdir(exist_ok=True)

        part_size = images_per_part + (1 if i < remainder else 0)
        end_idx = start_idx + part_size

        for image_file in all_images[start_idx:end_idx]:
            target_file = part_dir / image_file.name
            shutil.move(str(image_file), str(target_file))

        start_idx = end_idx

    return num_parts, total_images


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "split":
        source_dir = sys.argv[2] if len(sys.argv) > 2 else get_images_with_id_dir()
        num_parts = int(sys.argv[3]) if len(sys.argv) > 3 else 4

        parts, total = split_images_into_parts(source_dir, num_parts)

    else:
        source_dir = sys.argv[1] if len(sys.argv) > 1 else get_images_source_dir()
        target_dir = sys.argv[2] if len(sys.argv) > 2 else get_images_with_id_dir()

        count = extract_images_with_id(source_dir, target_dir)
