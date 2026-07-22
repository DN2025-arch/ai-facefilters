import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FaceRecognizer:
    """
    Robust Face Recognition classifier combining LBP spatial texture histograms,
    multi-scale pixel intensity features, PCA dimensionality reduction,
    and Support Vector Machine (SVM) / k-NN classification with confidence thresholding.
    """
    def __init__(self, confidence_threshold: float = 0.55, classifier_type: str = 'svm'):
        self.confidence_threshold = confidence_threshold
        self.classifier_type = classifier_type
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, svd_solver='full')
        self.is_pca_fitted = False
        
        if classifier_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=3, weights='distance')
        else:
            self.model = SVC(C=10.0, kernel='rbf', probability=True, class_weight='balanced')
            
        self.is_trained = False
        self.classes_: List[str] = []
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def _compute_lbp_pixel(self, img: np.ndarray, r: int, c: int) -> int:
        """Computes 8-neighbor Local Binary Pattern byte for pixel (r, c)."""
        center = img[r, c]
        val = 0
        val |= (1 if img[r - 1, c - 1] >= center else 0) << 7
        val |= (1 if img[r - 1, c]     >= center else 0) << 6
        val |= (1 if img[r - 1, c + 1] >= center else 0) << 5
        val |= (1 if img[r,     c + 1] >= center else 0) << 4
        val |= (1 if img[r + 1, c + 1] >= center else 0) << 3
        val |= (1 if img[r + 1, c]     >= center else 0) << 2
        val |= (1 if img[r + 1, c - 1] >= center else 0) << 1
        val |= (1 if img[r,     c - 1] >= center else 0) << 0
        return val

    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extracts feature vector from a face image ROI (expected 100x100 grayscale or BGR).
        Features combined:
        1. Multi-grid Local Binary Pattern (LBP) histograms (4x4 spatial regions)
        2. Downsampled intensity features (32x32)
        3. Mean & std gradient magnitudes
        """
        # Ensure grayscale & 100x100 standard ROI
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()

        if gray.shape != (100, 100):
            gray = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)

        # Equalize lighting
        gray = self.clahe.apply(gray)

        # 1. Spatial Grid LBP Histograms (4x4 grid = 16 sub-regions)
        h, w = gray.shape
        grid_r, grid_c = 4, 4
        sub_h, sub_w = h // grid_r, w // grid_c
        
        # Fast Vectorized LBP Approximation using OpenCV Bitwise Comparisons
        padded = np.pad(gray, 1, mode='edge')
        center = gray.astype(np.int16)
        
        lbp_img = np.zeros(gray.shape, dtype=np.uint8)
        neighbors = [
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, 2:],  padded[2:, 2:],   padded[2:, 1:-1],
            padded[2:, :-2],   padded[1:-1, :-2]
        ]
        
        for i, nbr in enumerate(neighbors):
            lbp_img |= ((nbr.astype(np.int16) >= center).astype(np.uint8) << i)

        # Extract normalized 16-bin histogram per sub-region (16 * 16 = 256 features)
        lbp_features = []
        for r in range(grid_r):
            for c in range(grid_c):
                sub_roi = lbp_img[r*sub_h : (r+1)*sub_h, c*sub_w : (c+1)*sub_w]
                hist, _ = np.histogram(sub_roi.ravel(), bins=16, range=(0, 256))
                hist_norm = hist.astype(np.float32) / (np.sum(hist) + 1e-6)
                lbp_features.extend(hist_norm)

        # 2. Downsampled Intensity (32x32 = 1024 features)
        downsampled = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).ravel().astype(np.float32) / 255.0

        # 3. Sobel Gradient Magnitude Summary (Sobel X & Y)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobelx, sobely)
        grad_small = cv2.resize(grad_mag, (16, 16), interpolation=cv2.INTER_AREA).ravel()
        grad_small = grad_small / (np.max(grad_small) + 1e-6)

        # Concatenate feature vector
        feature_vec = np.hstack([np.array(lbp_features, dtype=np.float32), downsampled, grad_small])
        return feature_vec

    def train(self, face_images: List[np.ndarray], labels: List[str]) -> bool:
        """
        Trains the classifier model given a list of face ROI images and corresponding string labels.
        Returns True if successful, False if insufficient data.
        """
        if len(face_images) < 1 or len(set(labels)) < 1:
            self.is_trained = False
            return False

        # Extract features for all sample images
        X_features = [self.extract_features(img) for img in face_images]
        X_features = np.array(X_features, dtype=np.float32)
        y_labels = np.array(labels)

        # Fit Scaler
        X_scaled = self.scaler.fit_transform(X_features)

        # Fit PCA if enough samples
        n_samples, n_features = X_scaled.shape
        if n_samples >= 3:
            n_comp = min(n_samples - 1, n_features, 50)
            self.pca = PCA(n_components=n_comp, svd_solver='full')
            X_reduced = self.pca.fit_transform(X_scaled)
            self.is_pca_fitted = True
        else:
            X_reduced = X_scaled
            self.is_pca_fitted = False

        # Train Classifier Model
        self.model.fit(X_reduced, y_labels)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True
        return True

    def predict(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predicts person name label and confidence score (0.0 to 1.0) for a face ROI.
        If confidence < confidence_threshold or model not trained, returns ("Unknown", score).
        """
        if not self.is_trained or len(self.classes_) == 0:
            return ("Unknown", 0.0)

        feat = self.extract_features(face_image).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)

        if self.is_pca_fitted:
            feat_reduced = self.pca.transform(feat_scaled)
        else:
            feat_reduced = feat_scaled

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_reduced)[0]
            max_idx = np.argmax(probs)
            pred_label = self.classes_[max_idx]
            confidence = float(probs[max_idx])
        else:
            pred_label = self.model.predict(feat_reduced)[0]
            confidence = 0.85

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return ("Unknown", confidence)

        return (pred_label, confidence)

    def save_model(self, filepath: str) -> bool:
        """Saves trained model state to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            state = {
                "scaler": self.scaler,
                "pca": self.pca,
                "is_pca_fitted": self.is_pca_fitted,
                "model": self.model,
                "classes_": self.classes_,
                "is_trained": self.is_trained,
                "confidence_threshold": self.confidence_threshold,
                "classifier_type": self.classifier_type
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            print(f"Error saving face recognizer model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Loads trained model state from file."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.scaler = state["scaler"]
            self.pca = state["pca"]
            self.is_pca_fitted = state["is_pca_fitted"]
            self.model = state["model"]
            self.classes_ = state["classes_"]
            self.is_trained = state["is_trained"]
            self.confidence_threshold = state.get("confidence_threshold", self.confidence_threshold)
            self.classifier_type = state.get("classifier_type", self.classifier_type)
            return True
        except Exception as e:
            print(f"Error loading face recognizer model: {e}")
            return False
