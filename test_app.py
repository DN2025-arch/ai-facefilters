import os
import sys
import cv2
import numpy as np

# Ensure path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from face_recognition_app.core.face_detector import FaceDetector
    from face_recognition_app.core.face_recognizer import FaceRecognizer
    from face_recognition_app.core.database_manager import DatabaseManager
    from face_recognition_app.core.video_processor import VideoProcessor
except ModuleNotFoundError:
    from core.face_detector import FaceDetector
    from core.face_recognizer import FaceRecognizer
    from core.database_manager import DatabaseManager
    from core.video_processor import VideoProcessor

def create_synthetic_face(bg_color=(200, 200, 200), eye_color=(50, 50, 50), mouth_color=(30, 30, 150)) -> np.ndarray:
    """Generates a synthetic 200x200 image containing facial elements for automated testing."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (240, 240, 240) # light background

    # Face ellipse
    cv2.ellipse(img, (150, 150), (60, 85), 0, 0, 360, bg_color, -1)
    cv2.ellipse(img, (150, 150), (60, 85), 0, 0, 360, (100, 100, 100), 2)

    # Eyes
    cv2.circle(img, (130, 130), 10, eye_color, -1)
    cv2.circle(img, (170, 130), 10, eye_color, -1)
    cv2.circle(img, (130, 130), 4, (255, 255, 255), -1)
    cv2.circle(img, (170, 130), 4, (255, 255, 255), -1)

    # Nose line
    cv2.line(img, (150, 140), (147, 165), (80, 80, 80), 2)
    cv2.line(img, (147, 165), (155, 165), (80, 80, 80), 2)

    # Mouth
    cv2.ellipse(img, (150, 185), (22, 12), 0, 0, 180, mouth_color, -1)

    return img

def run_tests():
    print("==================================================")
    print("    Running VisionID App Core Automated Tests     ")
    print("==================================================")

    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data_env")
    
    # 1. Test Detector
    print("[1/4] Testing FaceDetector...")
    detector = FaceDetector()
    synthetic_img = create_synthetic_face()
    quality = detector.check_quality(synthetic_img)
    print(f"   -> Image Quality Check: Good={quality['is_good_quality']}, Blur Score={quality['blur_score']:.1f}")

    # Crop ROI test
    bbox = (90, 65, 120, 170)
    cropped = detector.crop_face(synthetic_img, bbox, target_size=(100, 100))
    assert cropped.shape == (100, 100, 3), f"Expected (100, 100, 3) cropped ROI, got {cropped.shape}"
    print("   -> Face ROI cropping successfully verified.")

    # 2. Test Database Manager
    print("\n[2/4] Testing DatabaseManager...")
    db = DatabaseManager(test_data_dir)
    user1 = db.add_user("Alice Smith", role="Engineer", notes="Test User 1")
    user2 = db.add_user("Bob Johnson", role="Manager", notes="Test User 2")

    face_sample1 = create_synthetic_face(bg_color=(220, 180, 170), eye_color=(20, 20, 80))
    face_sample2 = create_synthetic_face(bg_color=(180, 220, 170), eye_color=(80, 20, 20))

    db.save_face_sample("Alice Smith", detector.crop_face(face_sample1, bbox))
    db.save_face_sample("Alice Smith", detector.crop_face(face_sample1, bbox))
    db.save_face_sample("Bob Johnson", detector.crop_face(face_sample2, bbox))
    db.save_face_sample("Bob Johnson", detector.crop_face(face_sample2, bbox))

    all_users = db.get_all_users()
    print(f"   -> Registered Users in DB: {[u['name'] for u in all_users]}")
    assert len(all_users) >= 2, "Failed to register users"

    # Test Logging & CSV export
    db.log_event("Alice Smith", 0.92, cropped)
    db.log_event("Unknown", 0.40, cropped)
    csv_file = os.path.join(test_data_dir, "test_attendance.csv")
    db.export_logs_to_csv(csv_file)
    assert os.path.exists(csv_file), "CSV export failed"
    print("   -> Event logging & CSV export verified.")

    # 3. Test Recognizer & Training
    print("\n[3/4] Testing FaceRecognizer & Classifier Training...")
    recognizer = FaceRecognizer(confidence_threshold=0.50)
    images, labels = db.get_all_training_data()
    print(f"   -> Training dataset size: {len(images)} samples across labels {set(labels)}")

    trained = recognizer.train(images, labels)
    assert trained, "Recognizer training failed"
    print("   -> Recognizer model trained successfully.")

    # Test Prediction
    test_roi_alice = detector.crop_face(face_sample1, bbox)
    pred_label, conf = recognizer.predict(test_roi_alice)
    print(f"   -> Prediction for Alice sample: Name='{pred_label}', Confidence={conf*100:.1f}%")

    model_save_path = os.path.join(test_data_dir, "test_model.pkl")
    save_ok = recognizer.save_model(model_save_path)
    assert save_ok and os.path.exists(model_save_path), "Model saving failed"

    load_rec = FaceRecognizer()
    load_ok = load_rec.load_model(model_save_path)
    assert load_ok and load_rec.is_trained, "Model loading failed"
    print("   -> Model persistence (save/load .pkl) verified.")

    # 4. Test VideoProcessor
    print("\n[4/4] Testing VideoProcessor frame annotations...")
    vp = VideoProcessor(detector, recognizer)
    annotated_frame, dets = vp.process_frame(synthetic_img)
    assert annotated_frame is not None and annotated_frame.shape == synthetic_img.shape
    print(f"   -> Frame processed & annotated. Frame size: {annotated_frame.shape}")

    # Clean up test dir
    import shutil
    try:
        shutil.rmtree(test_data_dir)
    except Exception:
        pass

    print("\n==================================================")
    print("     ALL CORE MODULE VERIFICATION TESTS PASSED!   ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
