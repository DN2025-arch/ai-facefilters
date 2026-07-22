# VisionID - Real-Time OpenCV Face Recognition System

A modern desktop application built in **Python** using **OpenCV**, **Scikit-Learn**, **NumPy**, and **Tkinter** for real-time face detection, user face enrollment, offline image/video analysis, attendance logging, and metrics analytics.

---

## 🌟 Key Features

1. **🎥 Real-Time Webcam Recognition**
   - High-FPS video stream processing using threaded OpenCV background capture (`CameraStream`).
   - Color-coded bounding box overlays with corner accents and confidence percentage tags (Green for Known, Crimson for Unknown alerts).
   - Live FPS monitor and quick snapshot capture tool.

2. **👤 Face Enrollment & Profile Database**
   - Interactive wizard to enroll new individuals via live webcam camera snapshots or external image uploads.
   - Profile metadata storage (Name, Role/Department, Registered Date, Notes, Face sample count).
   - Single-click model training and real-time database roster management.

3. **📁 Offline Image & Video Inspector**
   - Upload static images (`.jpg`, `.png`) to inspect face detections and view prediction summaries.
   - Process video files (`.mp4`, `.avi`) frame-by-frame and export annotated video outputs.

4. **📋 Attendance Logging & Export**
   - Automated event logger tracking timestamp, person name, confidence score, and status.
   - Styled treeview table with status indicators.
   - One-click CSV report exporter.

5. **⚙️ Custom Settings & Controls**
   - Adjust recognition confidence threshold slider dynamically.
   - Adjust face detection sensitivity.

---

## 🏗 System Architecture

```
face_recognition_app/
│
├── main.py                     # Application entry point
├── test_app.py                 # Automated verification test suite
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
│
├── core/                       # Core AI Engine Subsystem
│   ├── face_detector.py        # OpenCV Haar Cascade detection & quality checks
│   ├── face_recognizer.py      # LBP/HOG feature extraction + PCA + SVM classifier
│   ├── camera_stream.py        # Threaded async webcam frame capture
│   ├── database_manager.py     # JSON metadata, face sample storage, event logger
│   └── video_processor.py      # Frame annotation & offline file processor
│
├── gui/                        # Modern Dark GUI Layout
│   ├── app_window.py           # Main window, sidebar navigation, tab views
│   └── components.py           # Stat Cards, Canvas Display, Treeview Table
│
└── data/                       # Persistent Data Storage
    ├── enrolled_faces/         # Enrolled user image sample folders
    ├── model/                  # Trained classifier state (`recognizer_model.pkl`)
    ├── logs/                   # Recognition event logs & CSVs
    └── snapshots/              # Captured live stream snapshots
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.8+** installed on your system.

### 2. Installation
Open a terminal in the `face_recognition_app` directory and install required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Application
Launch the desktop GUI:
```bash
python main.py
```

### 4. Running Verification Tests
Execute automated unit and integration tests:
```bash
python test_app.py
```

---

## 📖 Quick Start Guide

1. **Start Live Feed**: Navigate to the **Live Stream** tab and click **▶ Start Webcam**.
2. **Register a Person**:
   - Go to **Enrollment & DB**.
   - Enter Full Name (e.g. `Jane Doe`), Role, and click **Capture From Camera ROI** or **Upload Sample Image File**.
   - Click **⚡ Train / Retrain Model** to fit the classifier on the new face data.
3. **View Attendance Logs**: Check the **Attendance Logs** tab to view real-time recognition history and click **📥 Export CSV Report** to download records.
