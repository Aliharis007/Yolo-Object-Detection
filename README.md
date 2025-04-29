# 🧠 YOLOv8 Object Detection Project

A YOLOv8-based object detection system featuring image recognition, custom training, and real-time webcam inference using OpenCV and Ultralytics.

This repository contains my implementation of object detection using **YOLOv8 (You Only Look Once)**. The project includes three main components:

- **Image-based object detection**
- **Custom model training using a Roboflow dataset**
- **Real-time object detection using webcam**

---

## 📁 Folder Structure

```
Yolo-Object-Detection/
│
├── images/                 # Folder for sample images
├── videos/                 # Folder for videos (optional)
├── annotations/            # Folder for annotation files (YOLO format)
├── best.pt                 # Fine-tuned YOLOv8 model
├── detect_image.py         # Image-based object detection script
├── detect_realtime.py      # Real-time webcam detection script
├── train_model.py          # Script for training the custom YOLOv8 model
└── README.md               # Project documentation
```

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **Ultralytics YOLOv8**
- **PyTorch**
- **Roboflow** (for dataset)

---

## 📸 Task 1: Image-Based Object Detection

- **Objective:** Use a pre-trained YOLOv8 model to detect objects in images.
- **Approach:**
  - Loaded the `yolov8s.pt` model.
  - Processed sample images with OpenCV.
  - Detected objects and drew bounding boxes with class labels and confidence scores.

### ✅ Challenges:
- Low confidence detections handled with thresholding.
- Label visibility improved with font scaling.

### 📌 **Observation:** 
The pre-trained YOLOv8 model showed fast inference (~325ms) and demonstrated good generalization on unseen images.

---

## 📦 Task 2: Custom YOLOv8 Training

- **Objective:** Train YOLOv8 model on a custom dataset.
- **Approach:**
  - Downloaded a public dataset from **Roboflow** in YOLO format.
  - Used the `train_model.py` script for training the YOLOv8 model.

### ✅ Challenges:
- Fixed annotation formatting issues during data preprocessing.
- Stabilized training by increasing batch size and training for at least 10 epochs.

### 📈 **Model Performance:**
- **mAP50**: 0.81
- **mAP50-95**: 0.64
- **Precision**: 0.82
- **Recall**: 0.73
- **Fitness Score**: 0.66

---

## 🎥 Task 3: Real-Time Detection via Webcam

- **Objective:** Perform real-time object detection using the webcam.
- **Approach:**
  - Loaded the fine-tuned `best.pt` model.
  - Processed video frames from the webcam and displayed detections in real-time.

### ✅ Challenges:
- Dropped FPS fixed by switching to a lightweight YOLOv8n model.
- Added adaptive brightness correction for low-light conditions.

### 📌 **Observation:**
The fine-tuned YOLOv8 model outperformed the default model on custom classes, achieving smoother real-time detection.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ultralytics opencv-python torch
```

### 3. Run Object Detection

- **Image-based Detection:**

```bash
python detect_image.py
```

- **Real-Time Webcam Detection:**

```bash
python detect_realtime.py
```

---

## 🧑‍💻 Author

Muhammad Ali Haris

---
