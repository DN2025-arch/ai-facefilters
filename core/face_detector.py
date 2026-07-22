import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional

class FaceDetector:
    """
    OpenCV Face Detection module using Haar Cascades with adaptive preprocessing,
    quality assessment, and face ROI alignment.
    """
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (40, 40)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load primary and fallback face cascade models
        cascade_dir = cv2.data.haarcascades
        primary_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
        secondary_path = os.path.join(cascade_dir, 'haarcascade_frontalface_alt2.xml')
        eye_path = os.path.join(cascade_dir, 'haarcascade_eye.xml')

        self.primary_cascade = cv2.CascadeClassifier(primary_path)
        self.secondary_cascade = cv2.CascadeClassifier(secondary_path) if os.path.exists(secondary_path) else None
        self.eye_cascade = cv2.CascadeClassifier(eye_path) if os.path.exists(eye_path) else None

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for light normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """Converts BGR image to grayscale and applies CLAHE contrast normalization."""
        if len(bgr_image.shape) == 3:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_image.copy()
            
        normalized_gray = self.clahe.apply(gray)
        return normalized_gray

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects face bounding boxes in an image.
        Returns a list of tuples: (x, y, w, h).
        """
        if image is None or image.size == 0:
            return []

        gray = self.preprocess_image(image)

        # Primary detection
        faces = self.primary_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        # Fallback to secondary cascade if no faces detected
        if len(faces) == 0 and self.secondary_cascade is not None:
            faces = self.secondary_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=self.min_size
            )

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Crops face region from image using bounding box (x, y, w, h)
        and resizes it to target_size.
        """
        h_img, w_img = image.shape[:2]
        x, y, w, h = bbox

        # Ensure coordinates are within frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        face_roi = image[y1:y2, x1:x2]

        if face_roi.size == 0:
            # Fallback blank image if ROI invalid
            return np.zeros((target_size[1], target_size[0], 3 if len(image.shape) == 3 else 1), dtype=np.uint8)

        # Apply slight padding to capture full chin and forehead
        resized_face = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        return resized_face

    def check_quality(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Checks face image quality:
        - blur score (Laplacian variance)
        - brightness/contrast status
        - size evaluation
        """
        if len(face_bgr.shape) == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr

        # Calculate blur via Laplacian variance
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)

        is_blurry = blur_var < 80.0
        is_too_dark = mean_brightness < 40.0
        is_too_bright = mean_brightness > 220.0

        is_good_quality = (not is_blurry) and (not is_too_dark) and (not is_too_bright)

        return {
            "is_good_quality": is_good_quality,
            "blur_score": float(blur_var),
            "is_blurry": is_blurry,
            "brightness": float(mean_brightness),
            "is_too_dark": is_too_dark,
            "is_too_bright": is_too_bright
        }
