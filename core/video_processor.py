import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Callable, Optional
try:
    from face_recognition_app.core.face_detector import FaceDetector
    from face_recognition_app.core.face_recognizer import FaceRecognizer
except ModuleNotFoundError:
    from core.face_detector import FaceDetector
    from core.face_recognizer import FaceRecognizer

class VideoProcessor:
    """
    Offline Inspector engine to detect and recognize faces in static images
    and video files, annotating output frames with clean visual overlays.
    """
    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer):
        self.detector = detector
        self.recognizer = recognizer

    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detects faces in a frame, predicts labels, draws bounding boxes & text badges.
        Returns (annotated_frame, list_of_detection_info).
        """
        if frame is None or frame.size == 0:
            return frame, []

        annotated = frame.copy()
        bboxes = self.detector.detect_faces(frame)
        detections = []

        for (x, y, w, h) in bboxes:
            face_roi = self.detector.crop_face(frame, (x, y, w, h))
            name, confidence = self.recognizer.predict(face_roi)

            # Color scheme: Emerald Green for Known, Vibrant Crimson for Unknown
            is_known = (name != "Unknown")
            bbox_color = (46, 204, 113) if is_known else (41, 128, 185) if confidence > 0.3 else (70, 70, 220) # BGR
            text_bg_color = bbox_color

            # Draw bounding box with rounded corner accents
            thick = max(2, int(w / 100))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), bbox_color, thick)

            # Draw corner accents
            line_len = int(min(w, h) * 0.2)
            cv2.line(annotated, (x, y), (x + line_len, y), bbox_color, thick + 1)
            cv2.line(annotated, (x, y), (x, y + line_len), bbox_color, thick + 1)
            cv2.line(annotated, (x + w, y), (x + w - line_len, y), bbox_color, thick + 1)
            cv2.line(annotated, (x + w, y), (x + w, y + line_len), bbox_color, thick + 1)
            cv2.line(annotated, (x, y + h), (x + line_len, y + h), bbox_color, thick + 1)
            cv2.line(annotated, (x, y + h), (x, y + h - line_len), bbox_color, thick + 1)
            cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), bbox_color, thick + 1)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), bbox_color, thick + 1)

            # Draw text label background pill
            conf_str = f"{int(confidence * 100)}%" if is_known else "ALERT"
            label_text = f"{name} ({conf_str})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.45, min(0.75, w / 220.0))
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            # Position label pill above or inside bbox
            lbl_y = y - 10 if y - 10 > text_h + 10 else y + text_h + 10
            cv2.rectangle(annotated, (x, lbl_y - text_h - 6), (x + text_w + 12, lbl_y + 4), text_bg_color, -1)
            cv2.putText(annotated, label_text, (x + 6, lbl_y - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "bbox": (x, y, w, h),
                "name": name,
                "confidence": confidence,
                "face_roi": face_roi
            })

        return annotated, detections

    def process_image(self, image_path: str) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Loads static image from disk and processes face detection & recognition."""
        img = cv2.imread(image_path)
        if img is None:
            return None, []
        return self.process_frame(img)

    def process_video_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> bool:
        """Processes video file frame by frame and saves annotated output video."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            annotated, _ = self.process_frame(frame)
            out.write(annotated)

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(min(1.0, frame_idx / total_frames))

        cap.release()
        out.release()
        return True
