#!/usr/bin/env python3
import sys
from pathlib import Path
from config import set_image_dir, get_image_dir

def main():
    if len(sys.argv) < 2:
        print("Usage: python set_config.py <image_dir>")
        print("Example: python set_config.py /path/to/images")
        print()
        print("Current settings:")
        print(f"  IMAGE_DIR: {get_image_dir()}")
        sys.exit(1)

    image_dir = sys.argv[1]
    set_image_dir(image_dir)
    print(f"✓ IMAGE_DIR set to: {get_image_dir()}")

    print()
    print("Environment variable has been set for this session.")
    print("To make it permanent, add this to your ~/.bashrc or ~/.zshrc:")
    print(f"  export IMAGE_DIR='{get_image_dir()}'")

if __name__ == "__main__":
    main()
