import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

IMAGE_DIR = os.getenv('IMAGE_DIR', str(PROJECT_ROOT / 'test_image'))

IMAGES_SOURCE_DIR = os.getenv('IMAGES_SOURCE_DIR', str(PROJECT_ROOT / 'images'))

IMAGES_WITH_ID_DIR = os.getenv('IMAGES_WITH_ID_DIR', str(PROJECT_ROOT / 'images_with_id'))

def get_image_dir():
    return IMAGE_DIR

def get_images_source_dir():
    return IMAGES_SOURCE_DIR

def get_images_with_id_dir():
    return IMAGES_WITH_ID_DIR

def set_image_dir(path):
    global IMAGE_DIR
    IMAGE_DIR = str(Path(path).resolve())
    os.environ['IMAGE_DIR'] = IMAGE_DIR
