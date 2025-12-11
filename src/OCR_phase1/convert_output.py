import json
from pathlib import Path
from typing import Dict, List, Tuple
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_image_dir


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def classify_by_overlap(annotations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    if not annotations:
        return [], []

    n = len(annotations)

    annotations_with_coords = []
    for annotation in annotations:
        y_coords = [point[1] for point in annotation['points']]
        x_coords = [point[0] for point in annotation['points']]

        y_min = min(y_coords)
        y_max = max(y_coords)
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_center = (y_min + y_max) / 2

        annotations_with_coords.append({
            'annotation': annotation,
            'y_center': y_center,
            'y_min': y_min,
            'y_max': y_max,
            'x_min': x_min,
            'x_max': x_max
        })

    uf = UnionFind(n)

    overlap_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            box_i = annotations_with_coords[i]
            box_j = annotations_with_coords[j]

            has_overlap = (
                box_i['y_max'] > box_j['y_min'] and
                box_j['y_max'] > box_i['y_min']
            )

            if has_overlap:
                uf.union(i, j)
                overlap_count += 1

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    if len(groups) == 1:
        group_list = [{'indices': list(groups.values())[0], 'y_center_avg': 0}]
    elif len(groups) > 2:
        group_list = []
        for group_indices in groups.values():
            y_centers = [annotations_with_coords[i]['y_center']
                         for i in group_indices]
            avg_y = sum(y_centers) / len(y_centers)
            group_list.append({
                'indices': group_indices,
                'y_center_avg': avg_y
            })

        group_list.sort(key=lambda g: g['y_center_avg'])

        max_gap = 0
        split_index = 1

        for i in range(len(group_list) - 1):
            gap = group_list[i + 1]['y_center_avg'] - \
                group_list[i]['y_center_avg']
            if gap > max_gap:
                max_gap = gap
                split_index = i + 1

        comment_indices = []
        content_indices = []

        for i, group in enumerate(group_list):
            if i < split_index:
                comment_indices.extend(group['indices'])
            else:
                content_indices.extend(group['indices'])

        group_list = [
            {'indices': comment_indices, 'y_center_avg': sum(
                [annotations_with_coords[i]['y_center'] for i in comment_indices]) / len(comment_indices)},
            {'indices': content_indices, 'y_center_avg': sum(
                [annotations_with_coords[i]['y_center'] for i in content_indices]) / len(content_indices)}
        ]

        group_list.sort(key=lambda g: g['y_center_avg'])

    else:
        group_list = []
        for group_indices in groups.values():
            y_centers = [annotations_with_coords[i]['y_center']
                         for i in group_indices]
            avg_y = sum(y_centers) / len(y_centers)
            group_list.append({
                'indices': group_indices,
                'y_center_avg': avg_y
            })

        group_list.sort(key=lambda g: g['y_center_avg'])

    if len(group_list) >= 2:
        comment_indices = group_list[0]['indices']
        content_indices = group_list[1]['indices']
    elif len(group_list) == 1:
        comment_indices = []
        content_indices = group_list[0]['indices']
    else:
        comment_indices = []
        content_indices = []

    comment_data = [annotations_with_coords[i] for i in comment_indices]
    content_data = [annotations_with_coords[i] for i in content_indices]

    comment_data_sorted = sorted(
        comment_data, key=lambda a: (-a['x_min'], a['y_min']))
    content_data_sorted = sorted(
        content_data, key=lambda a: (-a['x_min'], a['y_min']))

    comment_result = [
        {
            'transcription': item['annotation']['transcription'],
            'points': item['annotation']['points'],
            'difficult': item['annotation']['difficult'],
            'confidence': item['annotation']['confidence']
        }
        for item in comment_data_sorted
    ]

    content_result = [
        {
            'transcription': item['annotation']['transcription'],
            'points': item['annotation']['points'],
            'difficult': item['annotation']['difficult'],
            'confidence': item['annotation']['confidence']
        }
        for item in content_data_sorted
    ]

    return comment_result, content_result


def process_json_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for idx, item in enumerate(data):
        filename = item['filename']
        annotations = item['annotations']

        comment_annotations, content_annotations = classify_by_overlap(
            annotations)

        result.append({
            'filename': filename,
            'annotations': {
                'comment': comment_annotations,
                'content': content_annotations
            }
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def convert_cache_to_json(cache_file: str, output_file: str = None,
                          format_type: str = 'dict') -> None:
    if output_file is None:
        output_file = str(Path(cache_file).with_suffix('.json'))

    data = {}
    data_list = []

    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue

            filename = parts[0]
            annotations_str = parts[1]

            try:
                annotations = json.loads(annotations_str)

                if format_type == 'dict':
                    data[filename] = annotations
                else:
                    data_list.append({
                        'filename': filename,
                        'annotations': annotations
                    })

            except json.JSONDecodeError as e:
                continue

    output_data = data if format_type == 'dict' else data_list

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    images_dir = sys.argv[1] if len(sys.argv) > 1 else get_image_dir()
    images_dir = Path(images_dir)

    convert_cache_to_json(images_dir / "Cache.cach",
                          images_dir / "output.json", format_type='list')
    process_json_file(images_dir / "output.json",
                      images_dir / "output_processed.json")
