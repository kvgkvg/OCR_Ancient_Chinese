import json
import cv2
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_image_dir


def draw_polygon(img, points, color, thickness=2, label=None, font_scale=0.5):
    pts = np.array(points, dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    if label:
        x_min = min([p[0] for p in points])
        y_min = min([p[1] for p in points])

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x_min, y_min - text_h - 5),
                      (x_min + text_w, y_min), color, -1)

        cv2.putText(img, label, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)


def visualize_annotations(image_path, annotations, output_path,
                          show_text=False, box_thickness=2):
    img = cv2.imread(image_path)
    if img is None:
        return None

    result = img.copy()

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    comment_anns = annotations.get('comment', [])
    for ann in comment_anns:
        points = ann['bbox']
        text = ann['transcription'] if show_text else None
        draw_polygon(result, points, RED, box_thickness, text)

    content_anns = annotations.get('content', [])
    for ann in content_anns:
        points = ann['bbox']
        text = ann['transcription'] if show_text else None
        draw_polygon(result, points, GREEN, box_thickness, text)

    legend_y = 50
    cv2.putText(result, f"Comment boxes: {len(comment_anns)}", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
    cv2.putText(result, f"Content boxes: {len(content_anns)}", (20, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)

    cv2.imwrite(output_path, result)

    return result


def process_json_file(json_path, image_dir, output_dir, show_text=False, box_thickness=2):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for idx, item in enumerate(data):
        filename = item['filename']
        annotations = item['annotations']

        image_path = os.path.join(image_dir, os.path.basename(filename))

        if not os.path.exists(image_path):
            continue

        output_filename = f"viz_{os.path.basename(filename)}"
        output_path = os.path.join(output_dir, output_filename)

        n_comment = len(annotations.get('comment', []))
        n_content = len(annotations.get('content', []))

        result = visualize_annotations(
            image_path,
            annotations,
            output_path,
            show_text=show_text,
            box_thickness=box_thickness
        )

        if result is None:
            continue


def create_side_by_side_comparison(json_path, image_dir, output_path,
                                   sample_indices=None, max_samples=5):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if sample_indices is None:
        import random
        sample_indices = random.sample(
            range(len(data)), min(max_samples, len(data)))

    results = []

    for idx in sample_indices:
        item = data[idx]
        filename = item['filename']
        annotations = item['annotations']

        image_path = os.path.join(image_dir, os.path.basename(filename))

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        viz = img.copy()

        RED = (0, 0, 255)
        GREEN = (0, 255, 0)

        for ann in annotations.get('comment', []):
            pts = np.array(ann['bbox'], dtype=np.int32)
            cv2.polylines(viz, [pts], True, RED, 2)

        for ann in annotations.get('content', []):
            pts = np.array(ann['bbox'], dtype=np.int32)
            cv2.polylines(viz, [pts], True, GREEN, 2)

        max_height = 800
        h, w = viz.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            viz = cv2.resize(viz, (new_w, max_height))

        results.append(viz)

    if not results:
        return

    comparison = np.hstack(results)

    cv2.imwrite(output_path, comparison)


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else f'{get_image_dir()}/output_final_debug.json'
    image_dir = sys.argv[2] if len(sys.argv) > 2 else get_image_dir()
    output_dir = sys.argv[3] if len(sys.argv) > 3 else f'{get_image_dir()}/visualizations'

    process_json_file(
        json_path=json_path,
        image_dir=image_dir,
        output_dir=output_dir,
        show_text=False,
        box_thickness=3
    )
