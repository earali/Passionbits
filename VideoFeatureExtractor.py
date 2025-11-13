
import cv2
import numpy
import pytesseract
import argparse
import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
import re

# -------------------------
# Utility helpers
# -------------------------
def SampleFrames(cap, rate):
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % rate == 0:
            yield idx, frame
        idx += 1

def EnsureGray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def CleanText(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [t for t in s.split() if len(t) > 1]
    stopwords = set([
        "the","and","that","with","this","from","have","were","your","for","are",
        "you","not","but","was","what","when","where","which","them","then","than"
    ])
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

# -------------------------
# Shot Cut Detection
# -------------------------
def DetectShotCuts(video_path, sample_rate=1, hist_threshold=0.5, resize_width=320):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    prev_hist = None
    cuts = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx, frame in SampleFrames(cap, sample_rate):
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [50,60], [0,180,0,256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - sim
            if diff > hist_threshold:
                cuts.append(idx)
        prev_hist = hist
    cap.release()
    return {"num_cuts": len(cuts), "cut_frames": cuts}

# -------------------------
# Motion Analysis (Optical Flow)
# -------------------------
def MotionAnalysis(video_path, sample_rate=1, resize_width=320):
   
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    pairs = []
    prev_gray = None
    frame_idx = 0
    magnitudes = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                h, w = gray.shape
                if w > resize_width:
                    scale = resize_width / w
                    gray_r = cv2.resize(gray, (int(w*scale), int(h*scale)))
                    prev_gray_r = cv2.resize(prev_gray, (int(w*scale), int(h*scale)))
                else:
                    gray_r = gray
                    prev_gray_r = prev_gray
                flow = cv2.calcOpticalFlowFarneback(prev_gray_r, gray_r,
                                                    None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                mean_mag = numpy.mean(mag)
                magnitudes.append({"frame": idx, "mean_magnitude": float(mean_mag)})
            prev_gray = gray
        idx += 1
    cap.release()
    avg_motion = float(numpy.mean([m["mean_magnitude"] for m in magnitudes]) if magnitudes else 0.0)
    return {"average_motion": avg_motion, "per_sample": magnitudes}

# -------------------------
# Text Detection (OCR)
# -------------------------
def OcrTextDetection(video_path, sample_rate=30, resize_width=800, ocr_conf_thresh=50):
   
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)
    extracted_texts = []
    sampled = 0
    frames_with_text = 0
    for idx, frame in SampleFrames(cap, sample_rate):
        sampled += 1
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        try:
            text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
        except Exception as e:
            text = ""
        text = text.strip()
        if len(text) > 0:
            frames_with_text += 1
            extracted_texts.append(text)
    cap.release()
    text_present_ratio = frames_with_text / sampled if sampled else 0.0
    # simple keyword extraction: token frequency
    counter = Counter()
    for t in extracted_texts:
        tokens = CleanText(t)
        counter.update(tokens)
    top_keywords = [w for w, _ in counter.most_common(20)]
    return {"sampled_frames": sampled,
            "frames_with_text": frames_with_text,
            "text_present_ratio": float(text_present_ratio),
            "top_keywords": top_keywords,
            "raw_text_samples_count": len(extracted_texts)}

# -------------------------
# Object vs Person 
# -------------------------
def LoadYolo(net_cfg_path, net_weights_path, names_path):
    net = cv2.dnn.readNetFromDarknet(net_cfg_path, net_weights_path)
    with open(names_path, 'r') as f:
        class_names = [c.strip() for c in f.readlines()]
    return net, class_names

def YoloDetectPersons(video_path, net, class_names, sample_rate=30, conf_threshold=0.5, nms_threshold=0.4, input_size=416):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)
    person_count = 0
    object_count = 0
    frames_sampled = 0
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    for idx, frame in SampleFrames(cap, sample_rate):
        frames_sampled += 1
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []; confidences = []; classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = numpy.argmax(scores)
                confidence = float(scores[classID])
                if confidence > conf_threshold:
                    box = detection[0:4] * numpy.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(idxs) > 0:
            for i in idxs.flatten():
                cname = class_names[classIDs[i]]
                if cname == 'person':
                    person_count += 1
                else:
                    object_count += 1
    cap.release()
    ratio_person_to_object = person_count / (object_count + 1e-6) if (person_count + object_count) else 0.0
    return {"frames_sampled": frames_sampled,
            "person_count": int(person_count),
            "object_count": int(object_count),
            "person_to_object_ratio": float(ratio_person_to_object)}

# -------------------------
# Main orchestration
# -------------------------
def ExtractFeatures(video_path, args):
    features = {}
    # Shot cuts
    features["shot_cuts"] = DetectShotCuts(video_path,
                                             sample_rate=args.cut_sample_rate,
                                             hist_threshold=args.cut_hist_thresh,
                                             resize_width=args.cut_resize_width)
    # Motion
    features["motion"] = MotionAnalysis(video_path,
                                         sample_rate=args.motion_sample_rate,
                                         resize_width=args.motion_resize_width)
    # OCR
    features["ocr"] = OcrTextDetection(video_path,
                                         sample_rate=args.ocr_sample_rate,
                                         resize_width=args.ocr_resize_width)
    # YOLO optional
    if args.yolo_cfg and args.yolo_weights and args.yolo_names:
        net, class_names = LoadYolo(args.yolo_cfg, args.yolo_weights, args.yolo_names)
        features["yolo"] = YoloDetectPersons(video_path, net, class_names, sample_rate=args.yolo_sample_rate,
                                              conf_threshold=args.yolo_conf_thresh, nms_threshold=args.yolo_nms_thresh,
                                              input_size=args.yolo_input_size)
    else:
        features["yolo"] = None
    return features

# -------------------------
# CLI
# -------------------------
def CLIArgs():
    parser = argparse.ArgumentParser(description="Video Feature Extraction Tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out", required=False, help="Path to save JSON output")
    # shot cut options
    parser.add_argument("--cut_sample_rate", type=int, default=1, help="Sample every N frames for cut detection (1=every frame)")
    parser.add_argument("--cut_hist_thresh", type=float, default=0.3, help="Histogram diff threshold (0-1) for cut detection")
    parser.add_argument("--cut_resize_width", type=int, default=320, help="Resize width for speed in cut detection")
    # motion options
    parser.add_argument("--motion_sample_rate", type=int, default=1, help="Sample rate for motion (frames)")
    parser.add_argument("--motion_resize_width", type=int, default=320, help="Resize width for motion")
    # ocr options
    parser.add_argument("--ocr_sample_rate", type=int, default=30, help="Sample every N frames for OCR")
    parser.add_argument("--ocr_resize_width", type=int, default=800, help="Resize width for OCR")
    # yolo options
    parser.add_argument("--yolo_cfg", default=None, help="Path to yolo cfg file (optional)")
    parser.add_argument("--yolo_weights", default=None, help="Path to yolo weights file (optional)")
    parser.add_argument("--yolo_names", default=None, help="Path to coco names file (optional)")
    parser.add_argument("--yolo_sample_rate", type=int, default=30, help="Frame sampling for YOLO (optional)")
    parser.add_argument("--yolo_conf_thresh", type=float, default=0.5)
    parser.add_argument("--yolo_nms_thresh", type=float, default=0.4)
    parser.add_argument("--yolo_input_size", type=int, default=416)
    args = parser.parse_args()

    assert os.path.exists(args.video), "Video not found"

    features = ExtractFeatures(args.video, args)
    print(json.dumps(features, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(features, f, indent=2)
        print("Saved output to", args.out)

if __name__ == "__main__":
    CLIArgs()
