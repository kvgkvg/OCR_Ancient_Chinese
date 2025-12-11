from paddleocr import TextRecognition
import cv2
import json
import os
from typing import List, Dict, Tuple
from pathlib import Path
import copy
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_image_dir

rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")


def polygon_to_bbox(points: List[List[int]]) -> List[int]:
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def sort_annotations(annotations: List[Dict]) -> List[Dict]:
    """
    Sort annotations by reading order: top to bottom, right to left.

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

    # Sort by: top to bottom (ascending y), right to left (descending x)
    sorted_data = sorted(
        annotations_with_coords,
        key=lambda a: (a['y_min'], -a['x_min'])
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
            updated_annotations[category] = sort_annotations(updated_annotations[category])

    return updated_annotations


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else f'{get_image_dir()}/output_reanno.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else f'{get_image_dir()}/output_final.json'

    confidence_threshold = 0.6
    rollback_threshold = 0.4
    enable_preprocessing = True
    save_crops = True
    crops_dir = sys.argv[3] if len(sys.argv) > 3 else f'{get_image_dir()}/ocr_crops'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for idx, item in enumerate(data):
        filename = item['filename']
        annotations = item['annotations']

        updated_annotations = process_annotations(
            filename,
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

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
