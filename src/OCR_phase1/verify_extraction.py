from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_images_source_dir, get_images_with_id_dir


def verify_extraction(source_dir, target_dir):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Source directory not found: {source_dir}")
        return False

    if not target_path.exists():
        print(f"Target directory not found: {target_dir}")
        return False

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}

    source_images = {}
    for subdir in source_path.iterdir():
        if not subdir.is_dir():
            continue

        subfolder_name = subdir.name
        for image_file in subdir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                expected_name = f"{subfolder_name}_{image_file.name}"
                source_images[expected_name] = image_file

    target_images = {
        f.name: f for f in target_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    }

    print(f"Images in source subdirectories: {len(source_images)}")
    print(f"Images in target directory: {len(target_images)}")
    print()

    missing = []
    for expected_name in source_images:
        if expected_name not in target_images:
            missing.append(expected_name)

    extra = []
    for target_name in target_images:
        if target_name not in source_images:
            extra.append(target_name)

    if missing:
        print(f"Missing {len(missing)} images in target:")
        for name in missing[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()

    if extra:
        print(f"Extra {len(extra)} images in target (not found in source):")
        for name in extra[:10]:
            print(f"  - {name}")
        if len(extra) > 10:
            print(f"  ... and {len(extra) - 10} more")
        print()

    if not missing and not extra:
        print("✓ All images match perfectly!")
        print()

        subdirs = {}
        for name in source_images:
            subfolder = name.split('_')[0]
            subdirs[subfolder] = subdirs.get(subfolder, 0) + 1

        print("Distribution by subfolder:")
        for subfolder, count in sorted(subdirs.items()):
            print(f"  {subfolder}: {count} images")

        return True

    return False


def verify_split(split_dir, num_parts=4):
    split_path = Path(split_dir)

    if not split_path.exists():
        print(f"Split directory not found: {split_dir}")
        return False

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}

    part_dirs = [split_path / f"part_{i+1}" for i in range(num_parts)]

    missing_parts = []
    for i, part_dir in enumerate(part_dirs, 1):
        if not part_dir.exists():
            missing_parts.append(f"part_{i}")

    if missing_parts:
        print(f"Missing part directories: {', '.join(missing_parts)}")
        return False

    part_images = {}
    total_images = 0

    for i, part_dir in enumerate(part_dirs, 1):
        images = [
            f.name for f in part_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        part_images[f"part_{i}"] = images
        total_images += len(images)

    print(f"Total images across all parts: {total_images}")
    print()

    print("Distribution across parts:")
    counts = []
    for part_name in sorted(part_images.keys()):
        count = len(part_images[part_name])
        counts.append(count)
        print(f"  {part_name}: {count} images")
    print()

    if counts:
        min_count = min(counts)
        max_count = max(counts)
        diff = max_count - min_count

        if diff <= 1:
            print(f"✓ Distribution is balanced! (difference: {diff})")
        else:
            print(f"⚠ Distribution is unbalanced (difference: {diff})")
        print()

    all_images = set()
    duplicates = []
    for part_name, images in part_images.items():
        for img in images:
            if img in all_images:
                duplicates.append(img)
            else:
                all_images.add(img)

    if duplicates:
        print(f"⚠ Found {len(duplicates)} duplicate images:")
        for img in duplicates[:10]:
            print(f"  - {img}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
        return False
    else:
        print("✓ No duplicate images found")

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "split":
        split_dir = sys.argv[2] if len(sys.argv) > 2 else get_images_with_id_dir()
        num_parts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        verify_split(split_dir, num_parts)
    else:
        source_dir = sys.argv[1] if len(sys.argv) > 1 else get_images_source_dir()
        target_dir = sys.argv[2] if len(sys.argv) > 2 else get_images_with_id_dir()
        verify_extraction(source_dir, target_dir)
