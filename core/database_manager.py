import os
import json
import csv
import cv2
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

class DatabaseManager:
    """
    Manages enrolled user datasets, metadata JSON persistence,
    face ROI sample storage, recognition event logs, and CSV exporting.
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.faces_dir = os.path.join(base_dir, "enrolled_faces")
        self.model_dir = os.path.join(base_dir, "model")
        self.logs_dir = os.path.join(base_dir, "logs")
        self.snapshots_dir = os.path.join(base_dir, "snapshots")

        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.metadata_file = os.path.join(self.base_dir, "metadata.json")
        self.logs_file = os.path.join(self.logs_dir, "recognition_events.json")

        self.users: Dict[str, Dict[str, Any]] = self._load_json(self.metadata_file, {})
        self.logs: List[Dict[str, Any]] = self._load_json(self.logs_file, [])

    def _load_json(self, filepath: str, default: Any) -> Any:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading JSON from {filepath}: {e}")
        return default

    def _save_json(self, filepath: str, data: Any):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")

    def add_user(self, name: str, role: str = "Member", notes: str = "") -> str:
        """Registers a new user or updates profile."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("User name cannot be empty")

        user_dir = os.path.join(self.faces_dir, clean_name)
        os.makedirs(user_dir, exist_ok=True)

        if clean_name not in self.users:
            self.users[clean_name] = {
                "name": clean_name,
                "role": role,
                "notes": notes,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": 0
            }
        else:
            self.users[clean_name]["role"] = role
            self.users[clean_name]["notes"] = notes

        self._save_json(self.metadata_file, self.users)
        return clean_name

    def save_face_sample(self, name: str, face_img: np.ndarray) -> bool:
        """Saves a cropped face ROI sample image for specified user."""
        if name not in self.users:
            self.add_user(name)

        user_dir = os.path.join(self.faces_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        idx = self.users[name].get("sample_count", 0) + 1
        filename = f"sample_{idx:03d}_{int(time.time())}.jpg"
        filepath = os.path.join(user_dir, filename)

        # Standardize face ROI to 100x100 for storage efficiency
        if face_img.shape[:2] != (100, 100):
            face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)

        success = cv2.imwrite(filepath, face_img)
        if success:
            self.users[name]["sample_count"] = idx
            self._save_json(self.metadata_file, self.users)
        return success

    def get_user_samples(self, name: str) -> List[np.ndarray]:
        """Loads all face image samples for a specified user."""
        user_dir = os.path.join(self.faces_dir, name)
        if not os.path.exists(user_dir):
            return []

        samples = []
        for filename in os.listdir(user_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(user_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    samples.append(img)
        return samples

    def get_all_training_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Loads all face images and corresponding name labels across all enrolled users
        for model training.
        """
        images = []
        labels = []

        if not os.path.exists(self.faces_dir):
            return images, labels

        for user_name in os.listdir(self.faces_dir):
            user_dir = os.path.join(self.faces_dir, user_name)
            if os.path.isdir(user_dir):
                for fname in os.listdir(user_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(user_dir, fname)
                        img = cv2.imread(path)
                        if img is not None:
                            images.append(img)
                            labels.append(user_name)
        return images, labels

    def delete_user(self, name: str) -> bool:
        """Deletes user profile and associated image samples."""
        if name in self.users:
            del self.users[name]
            self._save_json(self.metadata_file, self.users)

        user_dir = os.path.join(self.faces_dir, name)
        if os.path.exists(user_dir):
            for fname in os.listdir(user_dir):
                try:
                    os.remove(os.path.join(user_dir, fname))
                except Exception:
                    pass
            try:
                os.rmdir(user_dir)
            except Exception:
                pass
        return True

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Returns list of all enrolled user dicts."""
        # Ensure sample counts are accurate with disk
        for user_name in list(self.users.keys()):
            user_dir = os.path.join(self.faces_dir, user_name)
            if os.path.exists(user_dir):
                files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.users[user_name]["sample_count"] = len(files)
            else:
                self.users[user_name]["sample_count"] = 0

        self._save_json(self.metadata_file, self.users)
        return list(self.users.values())

    def log_event(self, name: str, confidence: float, face_crop: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Logs a face recognition event."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        snapshot_rel_path = ""
        if face_crop is not None:
            snap_name = f"snap_{int(time.time()*1000)}.jpg"
            snap_path = os.path.join(self.snapshots_dir, snap_name)
            cv2.imwrite(snap_path, face_crop)
            snapshot_rel_path = snap_name

        event = {
            "id": len(self.logs) + 1,
            "timestamp": now_str,
            "name": name,
            "confidence": round(confidence * 100, 1),
            "status": "Known" if name != "Unknown" else "Unknown Alert",
            "snapshot": snapshot_rel_path
        }
        
        self.logs.insert(0, event) # Newest first
        if len(self.logs) > 500:   # Keep last 500 logs
            self.logs = self.logs[:500]

        self._save_json(self.logs_file, self.logs)
        return event

    def get_logs(self) -> List[Dict[str, Any]]:
        """Returns recognition events log history."""
        return self.logs

    def clear_logs(self):
        self.logs = []
        self._save_json(self.logs_file, self.logs)

    def export_logs_to_csv(self, csv_filepath: str) -> bool:
        """Exports recognition logs to CSV file."""
        try:
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Timestamp", "Name", "Confidence (%)", "Status"])
                for log in self.logs:
                    writer.writerow([log["id"], log["timestamp"], log["name"], log["confidence"], log["status"]])
            return True
        except Exception as e:
            print(f"Error exporting logs to CSV: {e}")
            return False
