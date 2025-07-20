
# 🚀 Case Study: Real-Time Object, Face, Age, Gender Detection with OCR

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Enabled-orange?logo=ultralytics)

---

## 📝 Introduction
This project leverages OpenCV, deep learning models, and Tesseract OCR to perform **real-time detection** of objects, faces, age, gender, mood (smile), and text from a webcam or RTSP stream. It integrates YOLOv8 for mobile phone detection and provides options to save annotated frames and debug output.

---

## ✨ Features
- **Object Detection:** MobileNet-SSD & YOLOv8 (COCO)
- **Face Detection:** Haar Cascade
- **Age & Gender Prediction:** Caffe models
- **Mood Detection:** Smile estimation
- **OCR:** Text extraction using Tesseract
- **Save Detections:** Store frames with annotations
- **Debug Mode:** Verbose output for troubleshooting

---

## 📦 Requirements
- Python 3.7+
- OpenCV
- imutils
- numpy
- pytesseract
- ultralytics (YOLO)

---

## 📁 Model Files
Place the following files in the project root directory:

- `age_deploy.prototxt`, `age_net.caffemodel`  
  <sub>For age prediction</sub>
- `gender_deploy.prototxt`, `gender_net.caffemodel`  
  <sub>For gender prediction</sub>
- `deploy.prototxt`, `mobilenet_iter_73000.caffemodel`  
  <sub>For MobileNet-SSD object detection</sub>
- `yolov8n.pt`  
  <sub>YOLOv8 model for mobile phone detection</sub>
- `emotion_model.hdf5` *(optional)*

---

## ⚙️ Installation
1. **Install Python dependencies:**
   ```powershell
   pip install opencv-python imutils numpy pytesseract ultralytics
   ```
2. **Install Tesseract OCR:**
   - Download from [Tesseract at UB Mannheim](https://github.com/tesseract-ocr/tesseract)
   - Set the path in `opencv.py`:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```

---

## 🚦 Usage
Run the main script from the project directory:

```powershell
python opencv.py [--rtsp <RTSP_URL>] [--save] [--debug]
```

**Arguments:**
- `--rtsp` : Use RTSP stream instead of webcam
- `--save` : Save frames with detections to `detections/`
- `--debug` : Print debug information

---

## 🎯 Output
- Detected objects, faces, age, gender, mood, and OCR text are displayed on the video stream.
- Saved frames are stored in the `detections/` folder if `--save` is used.

---

## 🛠️ Troubleshooting & Notes
- Ensure all model files are present in the project directory.
- For best results, use a well-lit environment.
- If Tesseract OCR is not found, check the path in `opencv.py`.
- For RTSP streams, ensure network connectivity and correct URL.

---


---

## �️ Project Structure
```
case study/
├── opencv.py                # Main script
├── README.md                # Documentation
├── detections/              # Saved frames (if --save is used)
├── age_deploy.prototxt      # Age model prototxt
├── age_net.caffemodel       # Age model weights
├── gender_deploy.prototxt   # Gender model prototxt
├── gender_net.caffemodel    # Gender model weights
├── deploy.prototxt          # MobileNet-SSD prototxt
├── mobilenet_iter_73000.caffemodel # MobileNet-SSD weights
├── yolov8n.pt               # YOLOv8 model
├── emotion_model.hdf5       # (Optional) Emotion model
└── ...                      # Other files
```

---

## ⚡ How It Works
1. **Video Capture:** Uses webcam or RTSP stream as input.
2. **Object Detection:** MobileNet-SSD and YOLOv8 models detect objects and mobile phones in real-time.
3. **Face Detection:** Haar Cascade locates faces in the frame.
4. **Age & Gender Prediction:** Caffe models estimate age and gender for detected faces.
5. **Mood Detection:** Smile detection estimates mood (happy/neutral).
6. **OCR:** Tesseract extracts text from detected regions.
7. **Annotation & Saving:** Results are displayed on the video stream and optionally saved.

---

## 🛠️ Customization
- **Change Models:** Replace model files in the root directory for different detection tasks.
- **Adjust Detection Thresholds:** Modify confidence thresholds in `opencv.py` for more/less sensitivity.
- **Add More Features:** Integrate additional models (e.g., emotion recognition) as needed.
- **Change Output Format:** Edit saving logic in `opencv.py` to customize output images or logs.

---

## ❓ FAQ
**Q: Can I use a different camera?**  
A: Yes, use the `--rtsp` argument for IP cameras or edit the source in `opencv.py`.

**Q: How do I add new object classes?**  
A: Update the model and class list in `opencv.py`.

**Q: Why is OCR not working?**  
A: Ensure Tesseract is installed and the path is set correctly in `opencv.py`.

**Q: How do I improve detection accuracy?**  
A: Use higher quality models, better lighting, and adjust detection parameters.

---

## 📬 Contact & Contributions
- For questions or suggestions, open an issue or pull request on this repository.
- Contributions are welcome! Please follow standard Python and documentation practices.

---

## 📜 License
This project is for educational purposes only.
