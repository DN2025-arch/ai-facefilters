import cv2
import time
import threading
from typing import Tuple, Optional, List
import numpy as np

class CameraStream:
    """
    Threaded OpenCV Video Capture pipeline for smooth, high-FPS video streaming.
    Frame reading runs asynchronously in a dedicated background thread to prevent GUI lagging.
    """
    def __init__(self, src: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.src = src
        self.resolution = resolution
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # FPS calculation variables
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.error_msg: Optional[str] = None

    def start(self) -> bool:
        """Starts the video stream capture thread."""
        if self.is_running:
            return True

        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
        if not self.cap.isOpened():
            # Try default backend if DSHOW fails
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            self.error_msg = f"Cannot open camera device index {self.src}"
            return False

        # Set resolution
        w, h = self.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Read initial frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.error_msg = "Failed to grab initial frame from camera"
            self.cap.release()
            return False

        self.frame = frame
        self.is_running = True
        self.error_msg = None
        self.start_time = time.time()
        self.frame_count = 0

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return True

    def _update_loop(self):
        """Continuously reads frames from camera in background thread."""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    
                    # Update FPS every 10 frames
                    if self.frame_count % 10 == 0:
                        now = time.time()
                        elapsed = now - self.start_time
                        if elapsed > 0:
                            self.fps = round(self.frame_count / elapsed, 1)
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (success, frame) from current buffer."""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def get_fps(self) -> float:
        """Returns calculated frames per second."""
        return self.fps

    def stop(self):
        """Stops video stream thread and releases camera."""
        self.is_running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    @staticmethod
    def list_available_cameras(max_check: int = 4) -> List[int]:
        """Scans system for available camera index devices."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available if available else [0]
