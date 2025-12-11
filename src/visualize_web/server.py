from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.converter import convert_output_processed_to_reanno, categorize_box

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_image_dir

app = Flask(__name__)

current_image_dir = None


@app.route('/')
def index():
    global current_image_dir
    if current_image_dir is None:
        current_image_dir = Path(get_image_dir()).resolve()
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/images', methods=['GET'])
def get_images():
    global current_image_dir

    try:
        dir_path = request.args.get('dir')
        if not dir_path:
            dir_path = get_image_dir()

        dir_path = Path(dir_path).resolve()
        if not dir_path.exists():
            return jsonify({'error': f'Directory not found: {dir_path}'}), 404

        current_image_dir = dir_path

        png_files = sorted([f.name for f in dir_path.glob('*.png')])

        if not png_files:
            return jsonify({'error': 'No PNG files found in directory'}), 404

        reanno_file = dir_path / 'output_reanno.json'
        processed_file = dir_path / 'output_processed.json'

        if reanno_file.exists():
            with open(reanno_file, 'r', encoding='utf-8') as f:
                annotations_data = json.load(f)
        elif processed_file.exists():
            annotations_data = convert_output_processed_to_reanno(processed_file, reanno_file)
        else:
            annotations_data = []

        annotations = {}
        for item in annotations_data:
            relative_path = item['filename']
            filename = Path(relative_path).name

            flat_annotations = (
                item['annotations'].get('comment', []) +
                item['annotations'].get('content', [])
            )
            annotations[filename] = flat_annotations

        return jsonify({
            'images': png_files,
            'annotations': annotations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<filename>')
def get_image(filename):
    global current_image_dir

    try:
        if current_image_dir is None:
            return jsonify({'error': 'No directory loaded'}), 400

        image_path = current_image_dir / filename

        if not image_path.exists() or not image_path.is_file():
            return jsonify({'error': 'Image not found'}), 404

        return send_file(str(image_path), mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations', methods=['POST'])
def save_annotations():
    global current_image_dir

    try:
        if current_image_dir is None:
            return jsonify({'error': 'No directory loaded'}), 400

        data = request.json
        filename = data['filename']
        annotations = data['annotations']

        dir_name = current_image_dir.name
        relative_filename = f"{dir_name}/{filename}"

        reanno_file = current_image_dir / 'output_reanno.json'

        if reanno_file.exists():
            with open(reanno_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        else:
            all_data = []

        categorized_annotations = {
            'comment': [],
            'content': []
        }

        for ann in annotations:
            category = categorize_box(ann['bbox'])
            categorized_annotations[category].append(ann)

        found = False
        for item in all_data:
            if item['filename'] == relative_filename:
                item['annotations'] = categorized_annotations
                found = True
                break

        if not found:
            all_data.append({
                'filename': relative_filename,
                'annotations': categorized_annotations
            })

        with open(reanno_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("Starting server...")
    print(f"Default IMAGE_DIR: {get_image_dir()}")
    print(f"Server will be available at http://localhost:{args.port}")
    print("To use a different IMAGE_DIR, set the IMAGE_DIR environment variable before starting the server")

    app.run(host=args.host, port=args.port, debug=args.debug)
