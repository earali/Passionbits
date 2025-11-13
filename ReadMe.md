# Video Feature Extraction Tool

This tool analyzes a local video and extracts:

* Shot cut detection (hard cuts) via histogram differences
* Motion analysis (average optical flow magnitude)
* Text detection via Tesseract OCR (text presence ratio + keywords)
* Object vs person dominance using YOLO (if model files supplied)

## Setup

1. Python 3.8+
2. Install Python deps: python -m pip install opencv-python-headless numpy pytesseract imutils tqdm
3. Install Tesseract OCR (system-level):

* Windows: download installer from Tesseract project and add to PATH

4. (Optional) Download YOLO model files:

* yolov3.cfg, yolov3.weights, coco.names and place them in yolo/ dir.

## Usage

Basic: python VideoFeatureExtractor.py --video video/vd5.mp4 --out features.json

With YOLO: python VideoFeatureExtractor.py --video video/vd5.mp4 --yolo\_cfg yolo/yolov3.cfg --yolo\_weights yolo/yolov3.weights --yolo\_names yolo/coco.names --out features.json

