import cv2
import os
import numpy as np
import pytesseract
from imutils.video import VideoStream
import time
import argparse
from ultralytics import YOLO
import sys
import site

# --------- CONFIGURATION ---------
AGE_PROTO = r"C:\Users\Shubham\Desktop\case study\age_deploy.prototxt"
AGE_MODEL = r"C:\Users\Shubham\Desktop\case study\age_net.caffemodel"
GENDER_PROTO = r"C:\Users\Shubham\Desktop\case study\gender_deploy.prototxt"
GENDER_MODEL = r"C:\Users\Shubham\Desktop\case study\gender_net.caffemodel"
FACE_PROTO = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
OBJ_PROTO = r"C:\Users\Shubham\Desktop\case study\deploy.prototxt"
OBJ_MODEL = r"C:\Users\Shubham\Desktop\case study\mobilenet_iter_73000.caffemodel"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
OBJ_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

# --------- ARGUMENTS ---------
parser = argparse.ArgumentParser(description="Live Object, Face, Age, Gender Detection with OCR")
parser.add_argument('--rtsp', type=str, default="", help="RTSP stream URL (leave blank for webcam)")
parser.add_argument('--save', action='store_true', help="Save frames with detections")
parser.add_argument('--debug', action='store_true', help="Enable debug prints")
args = parser.parse_args()

# --------- CHECK MODEL FILES ---------
for file in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL, OBJ_PROTO, OBJ_MODEL]:
    if not os.path.exists(file):
        print(f"[ERROR] Model file not found: {file}")
        exit()

# --------- INITIALIZE MODELS ---------
face_cascade = cv2.CascadeClassifier(FACE_PROTO)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
obj_net = cv2.dnn.readNetFromCaffe(OBJ_PROTO, OBJ_MODEL)

# Load YOLOv8 model (pretrained on COCO)
yolo_model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy

# --------- TESSERACT CONFIG ---------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --------- VIDEO SOURCE ---------
if args.rtsp:
    cap = cv2.VideoCapture(args.rtsp)
    print(f"[INFO] Using RTSP stream: {args.rtsp}")
else:
    cap = VideoStream(src=0).start()
    print("[INFO] Using webcam")
    time.sleep(2.0)

 # ...existing code...

# --------- UTILITY FUNCTIONS ---------
def preprocess_face(frame, x, y, w, h):
    face_img = frame[y:y+h, x:x+w]
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img, (227, 227))
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    return blob

def object_detection(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    obj_net.setInput(blob)
    detections = obj_net.forward()
    results = []
    # MobileNet-SSD detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append((OBJ_CLASSES[idx], confidence, (startX, startY, endX, endY)))
    # YOLOv8 mobile phone detection
    yolo_results = yolo_model(frame)
    boxes = yolo_results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # COCO class 67 is 'cell phone'
        if cls_id == 67 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            results.append(("Mobile Phone", conf, (x1, y1, x2, y2)))
    return results

def ocr_text(roi):
    # Only run OCR if ROI is large enough, not empty, and has enough variance
    if roi is not None and roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
        # Skip if the image is almost a single color (low variance)
        if np.std(roi) < 10:
            return ""
        try:
            text = pytesseract.image_to_string(roi)
            return text.strip()
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            return ""
    return ""

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def yolo_mobile_detection(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # COCO class 67 is 'cell phone'
        if cls_id == 67 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Mobile Phone: {conf*100:.1f}%", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# --------- MAIN LOOP ---------
frame_count = 0
start_time = time.time()
save_dir = "detections"
if args.save and not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("[DEBUG] Python executable:", sys.executable)
print("[DEBUG] sys.path:", sys.path)

while True:
    frame = cap.read() if args.rtsp else cap.read()
    if frame is None:
        print("[ERROR] Failed to capture frame.")
        break

    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # Object Detection
    objects = object_detection(frame)
    for obj_class, conf, (x1, y1, x2, y2) in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{obj_class}: {conf*100:.1f}%"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        roi = frame[y1:y2, x1:x2]
        text = ocr_text(roi)
        if text:
            cv2.putText(frame, f"OCR: {text}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if args.debug:
                print(f"[DEBUG] OCR on {obj_class}: {text}")

    # Face, Age, Gender Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        blob = preprocess_face(frame, x, y, w, h)
        if blob is None:
            continue
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        gender_conf = gender_preds[0].max()
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]
        age_conf = age_preds[0].max()
        label = f"{gender} ({gender_conf*100:.1f}%), {age} ({age_conf*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        face_roi = frame[y:y+h, x:x+w]
        text = ocr_text(face_roi)
        # Mood detection using smile (OpenCV only)
        mood = "Neutral"
        if face_roi is not None and face_roi.size > 0:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22)
            if len(smiles) > 0:
                mood = "Happy"
            cv2.putText(frame, f"Mood: {mood}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame in a window
    draw_fps(frame, fps)
    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
if args.rtsp:
    cap.release()
else:
    cap.stop()
cv2.destroyAllWindows()
