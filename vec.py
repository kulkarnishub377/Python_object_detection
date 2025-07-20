
import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
import pytesseract

# --- Age & Gender Models ---
AGE_PROTO = 'age_deploy.prototxt'
AGE_MODEL = 'age_net.caffemodel'
GENDER_PROTO = 'gender_deploy.prototxt'
GENDER_MODEL = 'gender_net.caffemodel'
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# --- RTSP Video Streaming Module ---
class RTSPCameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                # Attempt reconnection
                self.cap.release()
                time.sleep(1)  # Wait before retry
                self.cap = cv2.VideoCapture(self.rtsp_url)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# --- YOLO Vehicle Detection Module ---


class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt', device='auto'):
        self.model = YOLO(model_path)
        # Focus only on common Indian vehicle types, license plates, and persons
        self.vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
        self.plate_classes = ['license plate', 'number plate', 'plate']
        self.person_classes = ['person']

    def detect(self, frame):
        results = self.model(frame)
        vehicle_detections = []
        plate_detections = []
        person_detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[class_id]
                if class_name in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_detections.append({
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2)
                    })
                elif class_name.lower() in self.plate_classes:
                    x1, y1, x2, y2 = map(int, box)
                    plate_detections.append({
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2)
                    })
                elif class_name in self.person_classes:
                    x1, y1, x2, y2 = map(int, box)
                    person_detections.append({
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2)
                    })
        return vehicle_detections, plate_detections, person_detections

# --- License Plate Localization & OCR Module ---

import os
import csv

class LicensePlateOCR:
    def __init__(self, tesseract_cmd=None):
        # Tesseract config for a single line, uppercase, typical Indian plate chars
        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng'
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Check if tesseract is installed
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError("Tesseract OCR is not installed or not found. Please install it and add to PATH, or specify --tesseract argument.")

    def localize_plate(self, vehicle_img):
        # Heuristic: Indian plate is usually at the horizontal center, lower 1/3rd
        h, w = vehicle_img.shape[:2]
        y_start = int(h * 0.6)
        x_start = int(w * 0.17)
        x_end = int(w * 0.83)
        if y_start >= h or x_end >= w or x_start < 0:
            return None
        plate_region = vehicle_img[y_start:h, x_start:x_end]
        return plate_region

    def preprocess_plate(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def read_plate_text(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return ""
        processed = self.preprocess_plate(plate_img)
        text = pytesseract.image_to_string(processed, config=self.tesseract_config)
        # Clean OCR output: only valid chars, uppercase
        text = ''.join(filter(str.isalnum, text)).upper()
        return text

# --- Main System Integration ---

class VehicleRecognitionSystem:
    def __init__(self, rtsp_url, yolo_model_path='yolov8n.pt', tesseract_cmd=None, save_dir='detections', log_csv='detections_log.csv'):
        self.rtsp_stream = RTSPCameraStream(rtsp_url)
        self.detector = VehicleDetector(yolo_model_path)
        self.ocr = LicensePlateOCR(tesseract_cmd)
        self.running = True
        self.save_dir = save_dir
        self.log_csv = log_csv
        os.makedirs(self.save_dir, exist_ok=True)
        # Prepare CSV log file
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'class_name', 'confidence', 'plate_text', 'vehicle_img', 'plate_img', 'speed', 'age', 'gender'])
        # For speed estimation
        self.prev_positions = {}  # plate_text: (center_x, center_y, timestamp)
        # Load age/gender models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        self.gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

    def draw_detection(self, frame, detection, plate_text, fps=None):
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        # Color by class
        colors = {
            'car': (0, 255, 0),
            'bus': (0, 0, 255),
            'truck': (255, 0, 0),
            'motorcycle': (0, 255, 255)
        }
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        plate_label = f"Plate: {plate_text}" if plate_text else "Plate: N/A"
        cv2.putText(frame, plate_label, (x1, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)


    def predict_age_gender(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        return age, gender

    def process_frame(self, frame, fps=None):
        vehicle_detections, plate_detections, person_detections = self.detector.detect(frame)
        # For each vehicle, try to find a plate inside it
        for det in vehicle_detections:
            x1, y1, x2, y2 = det['bbox']
            vehicle_img = frame[y1:y2, x1:x2]
            # Find plate inside vehicle using YOLO plate detections
            plate_img = None
            for plate_det in plate_detections:
                px1, py1, px2, py2 = plate_det['bbox']
                # Check if plate bbox is inside vehicle bbox
                if px1 >= x1 and py1 >= y1 and px2 <= x2 and py2 <= y2:
                    plate_img = frame[py1:py2, px1:px2]
                    break
            # If no plate detected by YOLO, use heuristic
            if plate_img is None:
                plate_img = self.ocr.localize_plate(vehicle_img)
            plate_text = self.ocr.read_plate_text(plate_img)
            # --- Speed estimation ---
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            timestamp = time.time()
            speed = 0.0
            if plate_text:
                prev = self.prev_positions.get(plate_text)
                if prev:
                    prev_x, prev_y, prev_time = prev
                    dist = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                    dt = timestamp - prev_time
                    if dt > 0:
                        speed = dist / dt  # pixels per second
                self.prev_positions[plate_text] = (center_x, center_y, timestamp)
            self.draw_detection(frame, det, plate_text, fps)
            # Save images
            ts_str = time.strftime('%Y%m%d_%H%M%S')
            veh_path = os.path.join(self.save_dir, f'vehicle_{ts_str}_{det["class_name"]}.jpg')
            plate_path = os.path.join(self.save_dir, f'plate_{ts_str}.jpg')
            cv2.imwrite(veh_path, vehicle_img)
            if plate_img is not None:
                cv2.imwrite(plate_path, plate_img)
            # Log to CSV
            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ts_str, det['class_name'], det['confidence'], plate_text, veh_path, plate_path, speed, '', ''])
        # --- Person detection ---
        for person_det in person_detections:
            x1, y1, x2, y2 = person_det['bbox']
            person_img = frame[y1:y2, x1:x2]
            # Face detection inside person bbox
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            age, gender = '', ''
            for (fx, fy, fw, fh) in faces:
                face_crop = person_img[fy:fy+fh, fx:fx+fw]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    age, gender = self.predict_age_gender(face_crop)
                    # Draw age/gender on frame
                    cv2.putText(frame, f"{gender}, {age}", (x1, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                    break
            # Draw bounding box
            color = (0, 128, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {person_det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Save image
            ts_str = time.strftime('%Y%m%d_%H%M%S')
            person_path = os.path.join(self.save_dir, f'person_{ts_str}.jpg')
            cv2.imwrite(person_path, person_img)
            # Log to CSV
            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ts_str, 'person', person_det['confidence'], '', '', person_path, '', age, gender])
        return frame

    def run(self, display=True):
        print("Press ESC in the window to exit.")
        prev_time = time.time()
        fps = None
        while self.running:
            frame = self.rtsp_stream.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            curr_time = time.time()
            if fps is None:
                fps = 0.0
            else:
                fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            processed_frame = self.process_frame(frame, fps)
            if display:
                cv2.imshow('Indian Vehicle Detection + OCR', processed_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
        self.shutdown()

    def shutdown(self):
        self.running = False
        self.rtsp_stream.stop()
        cv2.destroyAllWindows()

# --- Script Entrypoint ---

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Real-time Indian Vehicle Detection and Plate OCR')
    parser.add_argument('--rtsp', type=str, default='', help='RTSP URL for video stream (leave blank for webcam)')
    parser.add_argument('--yolo', type=str, default='yolov8n.pt', help='Path to YOLO model (vehicle classes required)')
    parser.add_argument('--tesseract', type=str, default='', help='Path to tesseract.exe (if not in PATH)')
    parser.add_argument('--save_dir', type=str, default='detections', help='Directory to save detected vehicle and plate images')
    parser.add_argument('--log_csv', type=str, default='detections_log.csv', help='CSV file to log detections')
    args = parser.parse_args()
    # Use webcam if --rtsp is not provided
    video_source = args.rtsp if args.rtsp else 0
    tesseract_cmd = args.tesseract if args.tesseract else None
    system = VehicleRecognitionSystem(rtsp_url=video_source, yolo_model_path=args.yolo, tesseract_cmd=tesseract_cmd, save_dir=args.save_dir, log_csv=args.log_csv)
    try:
        system.run(display=True)
    except KeyboardInterrupt:
        system.shutdown()
        print('Stopped by user')
