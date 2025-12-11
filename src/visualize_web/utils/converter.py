import json
from pathlib import Path


def convert_quadrilateral_to_rectangle(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    return [
        [min(x_coords), min(y_coords)],
        [max(x_coords), max(y_coords)]
    ]


def categorize_box(bbox):
    y_min = bbox[0][1]
    y_max = bbox[1][1]

    if 0 < y_min < 900 and 0 < y_max < 900:
        return 'comment'
    return 'content'


def convert_output_processed_to_reanno(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for item in data:
        converted_annotations = {
            'comment': [],
            'content': []
        }

        for ann in item['annotations'].get('comment', []):
            converted_annotations['comment'].append({
                'transcription': ann['transcription'],
                'bbox': convert_quadrilateral_to_rectangle(ann['points']),
                'difficult': ann['difficult'],
                'confidence': ann['confidence']
            })

        for ann in item['annotations'].get('content', []):
            converted_annotations['content'].append({
                'transcription': ann['transcription'],
                'bbox': convert_quadrilateral_to_rectangle(ann['points']),
                'difficult': ann['difficult'],
                'confidence': ann['confidence']
            })

        result.append({
            'filename': item['filename'],
            'annotations': converted_annotations
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
