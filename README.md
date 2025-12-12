# OCR Ancient Chinese

End-to-end OCR system for Ancient Chinese texts with human-in-the-loop annotation and confidence-aware post-processing.

## Features

- **Automatic OCR** using PPOCRLabel with PaddleOCR backend
- **Smart classification** of annotations into comments vs main content
- **Web-based annotation tool** for adding missing bounding boxes
- **Confidence-aware re-recognition** with automatic rollback mechanism
- **Image preprocessing** (contrast enhancement, denoising, sharpening)
- **Quality assurance** with visualization tools

## Project Structure

```
OCR_Ancient_Chinese/
├── config.py                    # Central configuration
├── set_config.py                # Configuration utility
├── requirements.txt             # Main dependencies
├── src/
│   ├── OCR_phase1/             # Core OCR pipeline
│   │   ├── PPOCRLabel.py       # Custom PPOCRLabel GUI
│   │   ├── convert_output.py   # Convert & classify annotations
│   │   ├── post_ocr.py         # Re-recognize uncertain text
│   │   └── vsl.py              # Visualization (optional)
│   └── visualize_web/          # Web annotation interface
│       ├── server.py           # Flask backend
│       └── templates/          # Web UI
└── PPOCRLabel/                 # Cloned PPOCRLabel repo (step 3)
```

## Setup and Execution Guide

### 1. Configure Image Directory

Set `IMAGE_DIR` to point to your folder containing images for OCR:
```bash
export IMAGE_DIR=/path/to/your/images
```

And set environment variable directly:
```bash
python set_config.py /path/to/your/images
```



The `IMAGE_DIR` should contain PNG images that you want to perform OCR on.

### 2. Install Dependencies

Install main project dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `numpy>=1.21.0` - Numerical computing
- `opencv-python>=4.5.0` - Image processing
- `Flask>=2.3.0` - Web server
- `paddleocr` - OCR engine (for post-processing)

### 3. Setup PPOCRLabel

Clone the PPOCRLabel repository:

```bash
git clone https://github.com/PFCCLab/PPOCRLabel.git
```

Then follow their installation instructions...

**Important:** Replace the default `PPOCRLabel.py` with the custom version:

```bash
cp ../src/OCR_phase1/PPOCRLabel.py PPOCRLabel.py
```

Return to project root:

```bash
cd ..
```

### 4. Run Auto-Recognize with PPOCRLabel

Launch the PPOCRLabel GUI:

```bash
cd PPOCRLabel
python PPOCRLabel.py
```

In the GUI:
1. Click **Open Dir** and select your `IMAGE_DIR` folder
2. Click **Auto Recognize** to run OCR on all images
3. Wait for recognition to complete
4. **Close the GUI** - this saves results to `Cache.cach`

The GUI will generate a `Cache.cach` file in your `IMAGE_DIR`.

### 5. Convert PPOCRLabel Output

Convert and classify the OCR results:

```bash
python src/OCR_phase1/convert_output.py
```

This script:
- Reads `IMAGE_DIR/Cache.cach` (PPOCRLabel output)
- Converts to JSON format
- Classifies bounding boxes into **comment** and **content** categories using overlap analysis
- Outputs to `IMAGE_DIR/output_processed.json`

### 6. Add Missing Bounding Boxes (Web Interface)

Start the web annotation tool:

```bash
cd src/visualize_web
python server.py
```

The server will start on `http://localhost:5000`.

Open your browser and navigate to `http://localhost:5000`:
1. The web interface will automatically load images from `IMAGE_DIR`
2. Existing annotations from step 5 will be displayed
3. **Draw missing bounding boxes** by clicking and dragging on the canvas
4. Use the controls to:
   - Resize boxes (drag corners/edges)
   - Move boxes (drag the box)
   - Delete boxes (select and press Delete key)
   - Navigate between images (Previous/Next buttons or arrow keys)
5. Click **Save** after adding annotations to each image
6. Close the browser when done

The web tool saves annotations to `IMAGE_DIR/output_reanno.json`.

### 7. Post-OCR Processing

Run confidence-aware re-recognition:

```bash
python src/OCR_phase1/post_ocr.py
```

This script:
- Reads `IMAGE_DIR/output_reanno.json`
- Identifies low-confidence annotations (confidence < 0.6)
- Extracts and preprocesses text regions (contrast enhancement, denoising, sharpening)
- Re-runs OCR using PaddleOCR `PP-OCRv5_server_rec` model
- Compares original vs new confidence scores
- **Rolls back** if confidence drops > 0.4 (keeps original)
- **Accepts** new result if confidence improves
- Saves processing crops to `IMAGE_DIR/ocr_crops/{image_name}/`
- Outputs final results to `IMAGE_DIR/output_final.json`

Statistics are printed at the end showing:
- Total annotations processed
- High-confidence skipped
- Re-recognized count
- Accepted improvements
- Rolled back (kept original)

### 8. Visualization (Optional)

Generate annotated images for quality assurance:

```bash
python src/OCR_phase1/vsl.py
```

This creates visualization images in `IMAGE_DIR/visualizations/` with:
- Red boxes for comments
- Green boxes for content
- Optional text labels

## Data Flow

```
Cache.cach (PPOCRLabel)
    ↓ convert_output.py
output_processed.json (classified: comment/content)
    ↓ Web annotation tool
output_reanno.json (+ manually added boxes)
    ↓ post_ocr.py
output_final.json (refined text + confidence)
    ↓ vsl.py (optional)
visualizations/ (annotated images)
```

## Output Files

After completing all steps, your `IMAGE_DIR` will contain:

- `output_processed.json` - Initial classified annotations
- `output_reanno.json` - After web annotation
- `output_final.json` - **Final OCR results** with refined confidence
- `ocr_crops/` - Saved crop images for inspection
- `visualizations/` - Annotated images (if step 8 run)

## Configuration

Edit [config.py](config.py) to customize:

- `IMAGE_DIR` - Main working directory (default: `test_image`)
- `IMAGES_SOURCE_DIR` - Source images in subdirectories (default: `images`)
- `IMAGES_WITH_ID_DIR` - Flat directory with renamed images (default: `images_with_id`)

## Troubleshooting

**Issue:** `IMAGE_DIR not set`
**Solution:** Run `python set_config.py /path/to/images` or set environment variable

**Issue:** Web interface shows "No PNG files found"
**Solution:** Ensure `IMAGE_DIR` contains `.png` files (not `.jpg` or other formats)

**Issue:** `post_ocr.py` fails with model download error
**Solution:** PaddleOCR will auto-download models on first run - ensure internet connection

**Issue:** PPOCRLabel GUI doesn't start
**Solution:** Check that you installed PPOCRLabel requirements: `cd PPOCRLabel && pip install -r requirements.txt`

## Additional Tools

The project includes additional utilities in `src/OCR_phase1/`:

- `pdf2img.py` - Convert PDF documents to PNG images
- `extract_image.py` - Organize images from subdirectories with systematic naming
- `verify_extraction.py` - Validate image extraction and splitting

See individual script help: `python <script.py> --help`
