from paddleocr import TextRecognition
import cv2
import json
from typing import List, Dict, Tuple
from pathlib import Path
import copy
import numpy as np
import sys
import os
import re
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_image_dir

rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")


def polygon_to_bbox(points: List[List[int]]) -> List[int]:
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def is_chinese_char(char: str) -> bool:
    """Check if a character is Chinese"""
    if not char:
        return False
    return '\u4e00' <= char <= '\u9fff'


def filter_chinese_only(text: str) -> str:
    """Filter out non-Chinese characters from text"""
    return ''.join(char for char in text if is_chinese_char(char))


def calculate_x_overlap_ratio(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate overlap ratio along x-axis only.

    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    x1_min, _, x1_max, _ = bbox1
    x2_min, _, x2_max, _ = bbox2

    # Calculate overlap along x-axis
    overlap_start = max(x1_min, x2_min)
    overlap_end = min(x1_max, x2_max)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_length = overlap_end - overlap_start

    # Calculate lengths along x-axis
    length1 = x1_max - x1_min
    length2 = x2_max - x2_min

    # Avoid division by zero
    if length1 == 0 or length2 == 0:
        return 0.0

    # Return the ratio relative to the shorter bbox
    min_length = min(length1, length2)
    return overlap_length / min_length


def group_bboxes_by_overlap(annotations: List[Dict], overlap_threshold: float, sort_right_to_left: bool = False) -> List[Dict]:
    """
    Group bounding boxes based on x-axis overlap ratio.

    Args:
        annotations: List of annotation dictionaries
        overlap_threshold: Minimum overlap ratio to group bboxes
        sort_right_to_left: If True, sort from right to left then top to bottom.
                           If False, sort only top to bottom.

    Returns:
        List of grouped annotations
    """
    if not annotations:
        return []

    # Convert polygon to bbox if needed
    bboxes_with_data = []
    for ann in annotations:
        if 'bbox' in ann:
            points = ann['bbox']
        elif 'points' in ann:
            points = ann['points']
        else:
            continue

        bbox = polygon_to_bbox(points)
        bboxes_with_data.append({
            'bbox': bbox,
            'annotation': ann,
            'grouped': False
        })

    groups = []

    for i, item in enumerate(bboxes_with_data):
        if item['grouped']:
            continue

        # Start a new group with current bbox
        current_group = [i]
        item['grouped'] = True

        # Find all bboxes that overlap with any bbox in the current group
        changed = True
        while changed:
            changed = False
            for j, other_item in enumerate(bboxes_with_data):
                if other_item['grouped']:
                    continue

                # Check if this bbox overlaps with any bbox in the current group
                for group_idx in current_group:
                    overlap_ratio = calculate_x_overlap_ratio(
                        bboxes_with_data[group_idx]['bbox'],
                        other_item['bbox']
                    )

                    if overlap_ratio >= overlap_threshold:
                        current_group.append(j)
                        other_item['grouped'] = True
                        changed = True
                        break

        groups.append(current_group)

    # Create grouped annotations
    grouped_annotations = []

    for group_indices in groups:
        if len(group_indices) == 1:
            # Single bbox - keep as is but filter Chinese only
            idx = group_indices[0]
            ann = copy.deepcopy(bboxes_with_data[idx]['annotation'])
            transcription = ann.get('transcription', '')
            filtered_text = filter_chinese_only(transcription)
            ann['transcription'] = filtered_text
            grouped_annotations.append(ann)
        else:
            # Multiple bboxes - merge them
            group_bboxes = [bboxes_with_data[idx] for idx in group_indices]

            # Sort within group
            if sort_right_to_left:
                # Sort by: x descending (right to left), then y ascending (top to bottom)
                group_bboxes.sort(key=lambda x: (-x['bbox'][0], x['bbox'][1]))
            else:
                # Sort by: y ascending (top to bottom)
                group_bboxes.sort(key=lambda x: x['bbox'][1])

            # Calculate merged bbox
            all_x_mins = [item['bbox'][0] for item in group_bboxes]
            all_y_mins = [item['bbox'][1] for item in group_bboxes]
            all_x_maxs = [item['bbox'][2] for item in group_bboxes]
            all_y_maxs = [item['bbox'][3] for item in group_bboxes]

            merged_bbox = [
                min(all_x_mins),
                min(all_y_mins),
                max(all_x_maxs),
                max(all_y_maxs)
            ]

            # Convert bbox to polygon format
            x_min, y_min, x_max, y_max = merged_bbox
            merged_polygon = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]

            # Concatenate transcriptions and filter Chinese only
            transcriptions = [
                item['annotation'].get('transcription', '')
                for item in group_bboxes
            ]
            combined_text = ''.join(transcriptions)
            filtered_text = filter_chinese_only(combined_text)

            # Calculate average confidence
            confidences = [
                item['annotation'].get('confidence', 0.0)
                for item in group_bboxes
            ]
            avg_confidence = sum(confidences) / \
                len(confidences) if confidences else 0.0

            # Create merged annotation
            merged_annotation = {
                'bbox': merged_polygon,
                'transcription': filtered_text,
                'confidence': avg_confidence
            }

            grouped_annotations.append(merged_annotation)

    return grouped_annotations


def sort_annotations(annotations: List[Dict]) -> List[Dict]:
    """
    Sort annotations by reading order: right to left, top to bottom.

    Args:
        annotations: List of annotation dictionaries with 'bbox' field

    Returns:
        Sorted list of annotations
    """
    annotations_with_coords = []
    for ann in annotations:
        if 'bbox' in ann:
            points = ann['bbox']
        elif 'points' in ann:
            points = ann['points']
        else:
            annotations_with_coords.append({
                'annotation': ann,
                'x_min': 0,
                'y_min': 0
            })
            continue

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min = min(x_coords)
        y_min = min(y_coords)

        annotations_with_coords.append({
            'annotation': ann,
            'x_min': x_min,
            'y_min': y_min
        })

    # Sort by: right to left (descending x), then top to bottom (ascending y)
    sorted_data = sorted(
        annotations_with_coords,
        key=lambda a: (-a['x_min'], a['y_min'])
    )

    return [item['annotation'] for item in sorted_data]


def should_skip_ocr(transcription: str, confidence: float, threshold: float) -> tuple[bool, str]:
    transcription = transcription.strip() if transcription else ""

    if transcription == "":
        return False, "empty transcription"

    if confidence >= threshold:
        return True, f"has text + high confidence ({confidence:.3f})"
    else:
        return False, f"has text but low confidence ({confidence:.3f})"


def should_rollback(original_text: str,
                    original_conf: float,
                    new_text: str,
                    new_conf: float,
                    rollback_threshold: float = 0.4) -> tuple[bool, str]:
    conf_drop = original_conf - new_conf

    if original_text.strip() == "":
        return False, "original_empty"

    if conf_drop > rollback_threshold:
        return True, f"large_drop({conf_drop:.3f})"

    return False, "accept"


def preprocess_crop(crop: np.ndarray, enable_preprocessing: bool = True) -> np.ndarray:
    if not enable_preprocessing:
        return crop

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape[:2]
    if h < 32:
        scale = 32 / h
        new_w = int(w * scale)
        gray = cv2.resize(gray, (new_w, 32), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return result


def process_annotations(image_path: str,
                        annotations: Dict,
                        confidence_threshold: float = 0.6,
                        rollback_threshold: float = 0.4,
                        enable_preprocessing: bool = True,
                        save_crops: bool = True,
                        crops_dir: str = "ocr_crops") -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        return annotations

    updated_annotations = copy.deepcopy(annotations)

    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)
        image_name = Path(image_path).stem
        image_crops_dir = os.path.join(crops_dir, image_name)
        os.makedirs(image_crops_dir, exist_ok=True)

        original_dir = os.path.join(image_crops_dir, "original")
        preprocessed_dir = os.path.join(image_crops_dir, "preprocessed")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(preprocessed_dir, exist_ok=True)

    stats = {
        'total': 0,
        'skipped_high_conf': 0,
        'processed': 0,
        'accepted': 0,
        'rolled_back': 0,
        'crops_saved': 0
    }

    all_crops_info = []

    for category in ['comment', 'content']:
        if category not in updated_annotations:
            continue

        crops_for_ocr = []
        valid_indices = []
        crop_info = []

        for idx, ann in enumerate(updated_annotations[category]):
            stats['total'] += 1

            transcription = ann.get('transcription', '')
            confidence = ann.get('confidence', 0.0)

            skip, reason = should_skip_ocr(
                transcription, confidence, confidence_threshold)

            if skip:
                stats['skipped_high_conf'] += 1
                continue

            bbox = polygon_to_bbox(ann['bbox'])
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_original = image[y_min:y_max, x_min:x_max]

            if cropped_original.size == 0 or cropped_original.shape[0] < 5:
                continue

            cropped_preprocessed = preprocess_crop(
                cropped_original, enable_preprocessing)

            crops_for_ocr.append(cropped_preprocessed)
            valid_indices.append(idx)

            crop_info.append({
                'category': category,
                'index': idx,
                'original': cropped_original.copy(),
                'preprocessed': cropped_preprocessed,
                'transcription': transcription,
                'confidence': confidence,
                'bbox': bbox
            })

        if not crops_for_ocr:
            continue

        if save_crops and crops_for_ocr:
            for info in crop_info:
                filename = f"{info['category']}_{info['index']:03d}_conf{info['confidence']:.3f}.png"

                cv2.imwrite(os.path.join(original_dir, filename),
                            info['original'])
                cv2.imwrite(os.path.join(preprocessed_dir,
                            filename), info['preprocessed'])

                stats['crops_saved'] += 1

        try:
            outputs = rec_model.predict(crops_for_ocr, batch_size=16)

            for output, idx, crop_idx in zip(outputs, valid_indices, range(len(crop_info))):
                try:
                    result_dict = output.json
                    new_text = result_dict['res'].get('rec_text', '')
                    new_conf = result_dict['res'].get('rec_score', 0.0)
                except Exception as e:
                    continue

                original_text = updated_annotations[category][idx]['transcription']
                original_conf = updated_annotations[category][idx].get(
                    'confidence', 0.0)

                rollback, rollback_reason = should_rollback(
                    original_text,
                    original_conf,
                    new_text,
                    new_conf,
                    rollback_threshold
                )

                if rollback:
                    stats['rolled_back'] += 1
                    final_text = original_text
                    final_conf = original_conf
                    action = "rolled_back"
                else:
                    stats['accepted'] += 1
                    stats['processed'] += 1
                    final_text = new_text
                    final_conf = new_conf
                    action = "accepted"

                updated_annotations[category][idx]['transcription'] = final_text
                updated_annotations[category][idx]['confidence'] = final_conf

                all_crops_info.append({
                    'filename': f"{category}_{idx:03d}_conf{original_conf:.3f}.png",
                    'category': category,
                    'index': idx,
                    'original_text': original_text,
                    'original_confidence': original_conf,
                    'ocr_text': new_text,
                    'ocr_confidence': new_conf,
                    'final_text': final_text,
                    'final_confidence': final_conf,
                    'action': action,
                    'rollback_reason': rollback_reason if rollback else None
                })

        except Exception as e:
            import traceback
            traceback.print_exc()

    if save_crops and all_crops_info:
        manifest = {
            'config': {
                'confidence_threshold': confidence_threshold,
                'rollback_threshold': rollback_threshold,
                'preprocessing': enable_preprocessing
            },
            'crops': all_crops_info
        }
        manifest_path = os.path.join(image_crops_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Sort annotations by reading order before returning
    for category in ['comment', 'content']:
        if category in updated_annotations and updated_annotations[category]:
            updated_annotations[category] = sort_annotations(
                updated_annotations[category])

    return updated_annotations


def apply_bbox_grouping_stages(data: List[Dict], output_dir: str) -> List[Dict]:
    """
    Apply two-stage bounding box grouping.

    Stage 1: Group with overlap threshold 0.75, sort top to bottom
    Stage 2: Group with overlap threshold 0.2, sort right to left then top to bottom

    Args:
        data: List of items with annotations
        output_dir: Directory to save intermediate and final results

    Returns:
        Final grouped data
    """
    # Stage 1: Group with threshold 0.75, sort top to bottom
    stage1_results = []
    for item in data:
        annotations = item['annotations']
        grouped_annotations = {}

        for category in ['comment', 'content']:
            if category in annotations and annotations[category]:
                grouped = group_bboxes_by_overlap(
                    annotations[category],
                    overlap_threshold=0.75,
                    sort_right_to_left=False
                )
                # Sort the grouped bboxes: top to bottom
                grouped = sort_annotations(grouped)
                grouped_annotations[category] = grouped

        stage1_results.append({
            'filename': item['filename'],
            'annotations': grouped_annotations
        })

    # Save Stage 1 results
    stage1_output = os.path.join(output_dir, 'output_bbox1.json')
    with open(stage1_output, 'w', encoding='utf-8') as f:
        json.dump(stage1_results, f, ensure_ascii=False, indent=2)
    print(f"Stage 1 results saved to: {stage1_output}")

    # Stage 2: Group with threshold 0.2, sort right to left and top to bottom
    stage2_results = []
    for item in stage1_results:
        annotations = item['annotations']
        grouped_annotations = {}

        for category in ['comment', 'content']:
            if category in annotations and annotations[category]:
                grouped = group_bboxes_by_overlap(
                    annotations[category],
                    overlap_threshold=0.2,
                    sort_right_to_left=True
                )
                # Sort the grouped bboxes: right to left, top to bottom
                grouped = sort_annotations(grouped)
                grouped_annotations[category] = grouped

        stage2_results.append({
            'filename': item['filename'],
            'annotations': grouped_annotations
        })

    return stage2_results


def visualize_bboxes(data: List[Dict], image_dir: str, output_dir: str, num_samples: int = 5):
    """
    Visualize bounding boxes on images.

    Args:
        data: List of items with annotations
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # Randomly select samples
    samples = random.sample(data, min(num_samples, len(data)))

    for idx, item in enumerate(samples):
        filename = item['filename']
        annotations = item['annotations']

        # Load image
        image_path = os.path.join(image_dir, Path(filename).name)
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Draw bboxes
        for category in ['comment', 'content']:
            if category not in annotations:
                continue

            # Use different colors for different categories
            color = (0, 255, 0) if category == 'comment' else (255, 0, 0)

            for ann in annotations[category]:
                if 'bbox' not in ann:
                    continue

                points = ann['bbox']
                bbox = polygon_to_bbox(points)
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Draw rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

                # Draw text
                text = ann.get('transcription', '')
                if text:
                    # Put text above the bbox
                    cv2.putText(image, text[:20], (x_min, max(y_min - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Save visualization
        output_filename = f"visualize_{idx:03d}_{Path(filename).stem}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image)
        print(f"Saved visualization: {output_path}")


def main():
    input_file = sys.argv[1] if len(
        sys.argv) > 1 else f'{get_image_dir()}/output_reanno.json'
    output_file = sys.argv[2] if len(
        sys.argv) > 2 else f'{get_image_dir()}/output_final.json'

    confidence_threshold = 0.6
    rollback_threshold = 0.4
    enable_preprocessing = True
    save_crops = True
    crops_dir = sys.argv[3] if len(
        sys.argv) > 3 else f'{get_image_dir()}/ocr_crops'

    # Add parameters for visualization
    visualize = True
    num_visualize_samples = 5

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    image_dir = Path(get_image_dir())

    print("Starting OCR re-recognition...")
    for idx, item in enumerate(data):
        filename = item['filename']
        annotations = item['annotations']

        image_path = image_dir / Path(filename).name
        if not image_path.exists():
            image_path = image_dir / filename

        updated_annotations = process_annotations(
            str(image_path),
            annotations,
            confidence_threshold=confidence_threshold,
            rollback_threshold=rollback_threshold,
            enable_preprocessing=enable_preprocessing,
            save_crops=save_crops,
            crops_dir=crops_dir
        )

        results.append({
            'filename': filename,
            'annotations': updated_annotations
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(data)} images...")

    print(f"\nApplying bounding box grouping...")
    # Apply two-stage bounding box grouping
    output_dir = str(image_dir)
    final_results = apply_bbox_grouping_stages(results, output_dir)

    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"Final results saved to: {output_file}")

    # Visualize if requested
    if visualize:
        print(
            f"\nGenerating visualizations for {num_visualize_samples} samples...")
        visualize_dir = os.path.join(output_dir, 'bbox_visualizations')
        visualize_bboxes(final_results, str(image_dir),
                         visualize_dir, num_visualize_samples)
        print(f"Visualizations saved to: {visualize_dir}")


if __name__ == "__main__":
    main()
