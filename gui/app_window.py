import os
import time
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Dict, Any

try:
    from face_recognition_app.core.face_detector import FaceDetector
    from face_recognition_app.core.face_recognizer import FaceRecognizer
    from face_recognition_app.core.camera_stream import CameraStream
    from face_recognition_app.core.database_manager import DatabaseManager
    from face_recognition_app.core.video_processor import VideoProcessor
    from face_recognition_app.gui.components import (
        StatCard, VideoDisplayCanvas, LogTableWidget,
        BG_DARK, BG_CARD, BG_SIDEBAR, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, TEXT_MAIN, TEXT_MUTED, BORDER_COLOR
    )
except ModuleNotFoundError:
    from core.face_detector import FaceDetector
    from core.face_recognizer import FaceRecognizer
    from core.camera_stream import CameraStream
    from core.database_manager import DatabaseManager
    from core.video_processor import VideoProcessor
    from gui.components import (
        StatCard, VideoDisplayCanvas, LogTableWidget,
        BG_DARK, BG_CARD, BG_SIDEBAR, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, TEXT_MAIN, TEXT_MUTED, BORDER_COLOR
    )

class AppWindow(tk.Tk):
    """
    Main Application GUI Window for OpenCV Face Recognition Application.
    """
    def __init__(self, base_data_dir: str):
        super().__init__()
        self.title("VisionID - OpenCV Face Recognition System")
        self.geometry("1280x820")
        self.minsize(1024, 700)
        self.configure(bg=BG_DARK)

        self.base_data_dir = base_data_dir

        # Initialize Subsystems
        self.detector = FaceDetector()
        # Use slightly more permissive default confidence to reduce false 'Unknown'
        self.recognizer = FaceRecognizer(confidence_threshold=0.45)
        self.db = DatabaseManager(self.base_data_dir)
        self.video_processor = VideoProcessor(self.detector, self.recognizer)
        self.camera_stream = CameraStream(src=0)

        # Build header first so status labels exist for model training feedback
        self._build_header()

        # Load existing trained model if available, else auto-train from DB samples
        model_path = os.path.join(self.base_data_dir, "model", "recognizer_model.pkl")
        if os.path.exists(model_path):
            try:
                self.recognizer.load_model(model_path)
            except Exception:
                # If loading fails, mark untrained and allow retrain
                self.recognizer.is_trained = False
                self.lbl_model_status.config(text="○ Model Untrained", fg=ACCENT_RED)
        else:
            self._retrain_model_from_db(silent=True)

        # State Variables
        self.current_tab = "live"
        self.is_streaming = False
        self.draw_landmarks = True
        self.last_log_time: Dict[str, float] = {}

        # UI Layout Construction
        self._build_main_layout()

        # Start GUI Update Loop for video feed
        self.after(20, self._update_video_loop)

        # Handle window close gracefully
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_header(self):
        """Top Header Bar with Title, Status Badges, and Stats."""
        header = tk.Frame(self, bg=BG_SIDEBAR, height=56, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header.pack(side="top", fill="x")

        # Brand / Logo
        lbl_logo = tk.Label(header, text="👁️ VisionID", font=("Segoe UI", 14, "bold"), bg=BG_SIDEBAR, fg=TEXT_MAIN)
        lbl_logo.pack(side="left", padx=18, pady=10)

        lbl_sub = tk.Label(header, text="|  OpenCV Real-Time AI Engine", font=("Segoe UI", 10), bg=BG_SIDEBAR, fg=TEXT_MUTED)
        lbl_sub.pack(side="left", pady=10)

        # Status Indicators Right
        self.lbl_model_status = tk.Label(
            header,
            text="● Model Trained" if self.recognizer.is_trained else "○ Model Untrained",
            font=("Segoe UI", 10, "bold"),
            bg=BG_SIDEBAR,
            fg=ACCENT_GREEN if self.recognizer.is_trained else ACCENT_RED
        )
        self.lbl_model_status.pack(side="right", padx=18)

        self.lbl_fps = tk.Label(header, text="FPS: 0.0", font=("Segoe UI", 10), bg=BG_SIDEBAR, fg=TEXT_MUTED)
        self.lbl_fps.pack(side="right", padx=12)

    def _build_main_layout(self):
        """Builds Left Navigation Sidebar and Right Content Container."""
        container = tk.Frame(self, bg=BG_DARK)
        container.pack(fill="both", expand=True)

        # Sidebar Frame
        sidebar = tk.Frame(container, bg=BG_SIDEBAR, width=200, highlightbackground=BORDER_COLOR, highlightthickness=1)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Sidebar Nav Buttons
        nav_items = [
            ("live", "🎥  Live Stream"),
            ("enrollment", "👤  Enrollment & DB"),
            ("inspector", "📁  File Inspector"),
            ("logs", "📋  Attendance Logs"),
            ("settings", "⚙️  Settings")
        ]

        self.nav_buttons = {}
        for key, text in nav_items:
            btn = tk.Button(
                sidebar,
                text=text,
                font=("Segoe UI", 10, "bold"),
                anchor="w",
                padx=16,
                pady=12,
                bg=BG_SIDEBAR,
                fg=TEXT_MUTED,
                activebackground=BG_CARD,
                activeforeground=TEXT_MAIN,
                bd=0,
                cursor="hand2",
                command=lambda k=key: self.switch_tab(k)
            )
            btn.pack(fill="x", pady=2)
            self.nav_buttons[key] = btn

        # Content Area Stack
        self.content_area = tk.Frame(container, bg=BG_DARK, padx=16, pady=16)
        self.content_area.pack(side="right", fill="both", expand=True)

        # Panels
        self.panels = {
            "live": self._create_live_panel(),
            "enrollment": self._create_enrollment_panel(),
            "inspector": self._create_inspector_panel(),
            "logs": self._create_logs_panel(),
            "settings": self._create_settings_panel()
        }

        self.switch_tab("live")

    def switch_tab(self, tab_key: str):
        """Switches active view tab."""
        self.current_tab = tab_key
        for k, btn in self.nav_buttons.items():
            if k == tab_key:
                btn.config(bg=BG_CARD, fg=ACCENT_BLUE)
            else:
                btn.config(bg=BG_SIDEBAR, fg=TEXT_MUTED)

        for k, panel in self.panels.items():
            if k == tab_key:
                panel.pack(fill="both", expand=True)
            else:
                panel.pack_forget()

        if tab_key == "logs":
            self.refresh_logs_table()
        elif tab_key == "enrollment":
            self.refresh_users_list()

    # -------------------------------------------------------------------------
    # TAB 1: LIVE STREAM
    # -------------------------------------------------------------------------
    def _create_live_panel(self) -> tk.Frame:
        panel = tk.Frame(self.content_area, bg=BG_DARK)

        # Left Column: Video Feed & Controls
        left_col = tk.Frame(panel, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.live_canvas = VideoDisplayCanvas(left_col, width=640, height=480)
        self.live_canvas.pack(fill="both", expand=True)

        # Control Bar below video
        ctrl_bar = tk.Frame(left_col, bg=BG_CARD, pady=10, padx=12, highlightbackground=BORDER_COLOR, highlightthickness=1)
        ctrl_bar.pack(fill="x", pady=(10, 0))

        self.btn_toggle_stream = tk.Button(
            ctrl_bar, text="▶ Start Webcam", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_BLUE, fg="#FFFFFF", activebackground="#2563EB", bd=0, padx=16, pady=6,
            cursor="hand2", command=self.toggle_camera_stream
        )
        self.btn_toggle_stream.pack(side="left")

        btn_snap = tk.Button(
            ctrl_bar, text="📸 Snapshot", font=("Segoe UI", 10, "bold"),
            bg="#374151", fg=TEXT_MAIN, bd=0, padx=14, pady=6, cursor="hand2",
            command=self.take_live_snapshot
        )
        btn_snap.pack(side="left", padx=8)

        # Right Column: Metric Cards & Live Activity Ticker
        right_col = tk.Frame(panel, bg=BG_DARK, width=320)
        right_col.pack(side="right", fill="y")
        right_col.pack_propagate(False)

        # Metrics Stack
        self.card_detections = StatCard(right_col, "Total Detections", "0", icon_str="👁️", accent_color=ACCENT_BLUE)
        self.card_detections.pack(fill="x", pady=(0, 10))

        self.card_known = StatCard(right_col, "Known People Today", "0", icon_str="✅", accent_color=ACCENT_GREEN)
        self.card_known.pack(fill="x", pady=(0, 10))

        self.card_alerts = StatCard(right_col, "Unknown Alerts", "0", icon_str="⚠️", accent_color=ACCENT_RED)
        self.card_alerts.pack(fill="x", pady=(0, 10))

        # Recent Activity Box
        act_box = tk.LabelFrame(right_col, text="Live Activity Feed", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN, padx=10, pady=10)
        act_box.pack(fill="both", expand=True)

        self.txt_activity = tk.Text(act_box, bg="#11131C", fg="#D1D5DB", font=("Consolas", 9), bd=0, wrap="word")
        self.txt_activity.pack(fill="both", expand=True)
        self.txt_activity.insert("end", "System ready. Click 'Start Webcam' to begin live detection.\n")

        return panel

    def toggle_camera_stream(self):
        """Starts or stops webcam video stream."""
        if not self.is_streaming:
            success = self.camera_stream.start()
            if success:
                self.is_streaming = True
                self.btn_toggle_stream.config(text="⏹ Stop Stream", bg=ACCENT_RED)
                self._log_activity("Webcam stream started.")
            else:
                err = self.camera_stream.error_msg or "Failed to connect to camera."
                messagebox.showerror("Camera Error", err)
        else:
            self.camera_stream.stop()
            self.is_streaming = False
            self.btn_toggle_stream.config(text="▶ Start Webcam", bg=ACCENT_BLUE)
            self.live_canvas.update_image(None)
            self._log_activity("Webcam stream stopped.")

    def take_live_snapshot(self):
        """Saves current webcam frame to snapshots folder."""
        ret, frame = self.camera_stream.read()
        if ret and frame is not None:
            snap_path = os.path.join(self.db.snapshots_dir, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, frame)
            messagebox.showinfo("Snapshot Saved", f"Saved image to:\n{snap_path}")
        else:
            messagebox.showwarning("Warning", "No active camera frame to capture.")

    def _log_activity(self, text: str):
        t_str = time.strftime("[%H:%M:%S] ")
        self.txt_activity.insert("end", t_str + text + "\n")
        self.txt_activity.see("end")

    # -------------------------------------------------------------------------
    # TAB 2: ENROLLMENT & DATABASE MANAGEMENT
    # -------------------------------------------------------------------------
    def _create_enrollment_panel(self) -> tk.Frame:
        panel = tk.Frame(self.content_area, bg=BG_DARK)

        # Left Column: User Registration Form
        left_box = tk.LabelFrame(panel, text="Register New Person", font=("Segoe UI", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN, padx=14, pady=14, width=340)
        left_box.pack(side="left", fill="y", padx=(0, 12))
        left_box.pack_propagate(False)

        tk.Label(left_box, text="Full Name:", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", pady=(4, 2))
        self.ent_name = tk.Entry(left_box, font=("Segoe UI", 10), bg="#11131C", fg=TEXT_MAIN, insertbackground=TEXT_MAIN, bd=1)
        self.ent_name.pack(fill="x", pady=(0, 10))

        tk.Label(left_box, text="Role / Department:", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", pady=(4, 2))
        self.ent_role = tk.Entry(left_box, font=("Segoe UI", 10), bg="#11131C", fg=TEXT_MAIN, insertbackground=TEXT_MAIN, bd=1)
        self.ent_role.insert(0, "Employee")
        self.ent_role.pack(fill="x", pady=(0, 10))

        tk.Label(left_box, text="Notes:", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", pady=(4, 2))
        self.ent_notes = tk.Entry(left_box, font=("Segoe UI", 10), bg="#11131C", fg=TEXT_MAIN, insertbackground=TEXT_MAIN, bd=1)
        self.ent_notes.pack(fill="x", pady=(0, 16))

        # Sample Capture Actions
        btn_snap_sample = tk.Button(
            left_box, text="📷 Capture From Camera ROI", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_BLUE, fg="#FFFFFF", bd=0, pady=8, cursor="hand2",
            command=self.capture_sample_from_camera
        )
        btn_snap_sample.pack(fill="x", pady=(0, 8))

        btn_upload_sample = tk.Button(
            left_box, text="📁 Upload Sample Image File", font=("Segoe UI", 10, "bold"),
            bg="#374151", fg=TEXT_MAIN, bd=0, pady=8, cursor="hand2",
            command=self.upload_sample_image
        )
        btn_upload_sample.pack(fill="x", pady=(0, 16))

        # Train Model Action Box
        tk.Label(left_box, text="Model Training", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", pady=(10, 4))
        btn_retrain = tk.Button(
            left_box, text="⚡ Train / Retrain Model", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_GREEN, fg="#FFFFFF", bd=0, pady=10, cursor="hand2",
            command=lambda: self._retrain_model_from_db(silent=False)
        )
        btn_retrain.pack(fill="x")

        # Right Column: Registered Users Table
        right_box = tk.LabelFrame(panel, text="Enrolled Database Roster", font=("Segoe UI", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN, padx=14, pady=14)
        right_box.pack(side="right", fill="both", expand=True)

        cols = ("name", "role", "samples", "created")
        self.tree_users = ttk.Treeview(right_box, columns=cols, show="headings", style="Dark.Treeview")
        self.tree_users.heading("name", text="Name")
        self.tree_users.heading("role", text="Role")
        self.tree_users.heading("samples", text="Samples Count")
        self.tree_users.heading("created", text="Registered Date")

        self.tree_users.column("name", width=140)
        self.tree_users.column("role", width=110)
        self.tree_users.column("samples", width=100, anchor="center")
        self.tree_users.column("created", width=150, anchor="center")

        self.tree_users.pack(fill="both", expand=True, pady=(0, 10))

        btn_del_user = tk.Button(
            right_box, text="🗑 Delete Selected Person", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_RED, fg="#FFFFFF", bd=0, pady=6, padx=12, cursor="hand2",
            command=self.delete_selected_user
        )
        btn_del_user.pack(anchor="e")

        return panel

    def capture_sample_from_camera(self):
        """Captures face sample for user from active camera stream."""
        name = self.ent_name.get().strip()
        if not name:
            messagebox.showwarning("Validation Error", "Please enter Full Name first.")
            return

        if not self.is_streaming:
            messagebox.showwarning("Camera Stopped", "Please start webcam stream first in Live Stream tab.")
            return

        ret, frame = self.camera_stream.read()
        if not ret or frame is None:
            messagebox.showerror("Error", "Could not read frame from camera.")
            return

        bboxes = self.detector.detect_faces(frame)
        if len(bboxes) == 0:
            messagebox.showwarning("No Face Detected", "No face detected in camera frame. Please position face clearly.")
            return

        # Take largest face detected
        bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        face_roi = self.detector.crop_face(frame, bboxes[0])

        role = self.ent_role.get().strip() or "Employee"
        notes = self.ent_notes.get().strip()
        self.db.add_user(name, role, notes)
        
        success = self.db.save_face_sample(name, face_roi)
        if success:
            messagebox.showinfo("Success", f"Captured face sample for '{name}'.")
            self.refresh_users_list()
        else:
            messagebox.showerror("Error", "Failed to save face sample.")

    def upload_sample_image(self):
        """Uploads an image file to extract face and save to database."""
        name = self.ent_name.get().strip()
        if not name:
            messagebox.showwarning("Validation Error", "Please enter Full Name first.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Unable to load selected image.")
            return

        bboxes = self.detector.detect_faces(img)
        if len(bboxes) == 0:
            messagebox.showwarning("No Face Found", "No face detected in selected image.")
            return

        role = self.ent_role.get().strip() or "Employee"
        notes = self.ent_notes.get().strip()
        self.db.add_user(name, role, notes)

        saved_count = 0
        for bbox in bboxes:
            face_roi = self.detector.crop_face(img, bbox)
            if self.db.save_face_sample(name, face_roi):
                saved_count += 1

        messagebox.showinfo("Upload Complete", f"Added {saved_count} face sample(s) for '{name}'.")
        self.refresh_users_list()

    def refresh_users_list(self):
        """Refreshes enrolled user roster table."""
        for item in self.tree_users.get_children():
            self.tree_users.delete(item)

        users = self.db.get_all_users()
        for u in users:
            self.tree_users.insert("", "end", values=(u["name"], u["role"], u["sample_count"], u["created_at"]))

    def delete_selected_user(self):
        """Deletes selected user from database."""
        selected = self.tree_users.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select a person from the roster to delete.")
            return

        values = self.tree_users.item(selected[0], "values")
        user_name = values[0]

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete profile for '{user_name}'?"):
            self.db.delete_user(user_name)
            self.refresh_users_list()
            self._retrain_model_from_db(silent=True)
            messagebox.showinfo("Deleted", f"User '{user_name}' deleted.")

    def _retrain_model_from_db(self, silent: bool = False):
        """Trains recognizer on all enrolled database samples."""
        images, labels = self.db.get_all_training_data()
        print(f"[DEBUG] Retrain requested — found {len(images)} images for {len(set(labels))} labels")
        if len(images) == 0:
            self.recognizer.is_trained = False
            self.lbl_model_status.config(text="○ Model Untrained", fg=ACCENT_RED)
            if not silent:
                messagebox.showwarning("No Data", "No enrolled face samples found in database. Add faces first.")
            return

        try:
            success = self.recognizer.train(images, labels)
        except Exception as e:
            print(f"[ERROR] Training exception: {e}")
            self.recognizer.is_trained = False
            self.lbl_model_status.config(text="○ Model Untrained", fg=ACCENT_RED)
            if not silent:
                messagebox.showerror("Training Error", f"Model training failed: {e}")
            return

        if success:
            model_path = os.path.join(self.base_data_dir, "model", "recognizer_model.pkl")
            saved = self.recognizer.save_model(model_path)
            self.lbl_model_status.config(text="● Model Trained", fg=ACCENT_GREEN)
            if not silent:
                messagebox.showinfo("Model Trained", f"Successfully trained recognition model on {len(images)} samples across {len(set(labels))} people. Saved: {saved}")
        else:
            self.lbl_model_status.config(text="○ Model Untrained", fg=ACCENT_RED)
            if not silent:
                messagebox.showerror("Training Error", "Model training failed.")

    # -------------------------------------------------------------------------
    # TAB 3: FILE INSPECTOR
    # -------------------------------------------------------------------------
    def _create_inspector_panel(self) -> tk.Frame:
        panel = tk.Frame(self.content_area, bg=BG_DARK)

        # Top Control Bar
        bar = tk.Frame(panel, bg=BG_CARD, padx=12, pady=10, highlightbackground=BORDER_COLOR, highlightthickness=1)
        bar.pack(fill="x", pady=(0, 12))

        btn_img = tk.Button(
            bar, text="🖼 Open Image File", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_BLUE, fg="#FFFFFF", bd=0, padx=14, pady=6, cursor="hand2",
            command=self.inspect_image_file
        )
        btn_img.pack(side="left", padx=(0, 8))

        btn_vid = tk.Button(
            bar, text="🎞 Process Video File", font=("Segoe UI", 10, "bold"),
            bg="#374151", fg=TEXT_MAIN, bd=0, padx=14, pady=6, cursor="hand2",
            command=self.inspect_video_file
        )
        btn_vid.pack(side="left")

        self.lbl_inspector_status = tk.Label(bar, text="Select an image or video file to process.", font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_MUTED)
        self.lbl_inspector_status.pack(side="right", padx=10)

        # Canvas Display
        self.inspector_canvas = VideoDisplayCanvas(panel, width=800, height=500)
        self.inspector_canvas.pack(fill="both", expand=True)

        return panel

    def inspect_image_file(self):
        """Processes static image file."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        annotated, detections = self.video_processor.process_image(file_path)
        if annotated is not None:
            self.inspector_canvas.update_image(annotated)
            names = [d["name"] for d in detections]
            self.lbl_inspector_status.config(text=f"Detected {len(detections)} face(s): {', '.join(names) if names else 'None'}")
        else:
            messagebox.showerror("Error", "Could not load image file.")

    def inspect_video_file(self):
        """Processes video file frame by frame and saves output."""
        input_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if not input_path:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not output_path:
            return

        self.lbl_inspector_status.config(text="Processing video file... Please wait.")
        self.update()

        success = self.video_processor.process_video_file(input_path, output_path)
        if success:
            self.lbl_inspector_status.config(text=f"Finished processing! Exported to: {os.path.basename(output_path)}")
            messagebox.showinfo("Video Processing Complete", f"Exported annotated video to:\n{output_path}")
        else:
            messagebox.showerror("Error", "Failed to process video file.")

    # -------------------------------------------------------------------------
    # TAB 4: ATTENDANCE & LOGS
    # -------------------------------------------------------------------------
    def _create_logs_panel(self) -> tk.Frame:
        panel = tk.Frame(self.content_area, bg=BG_DARK)

        # Header Action Bar
        bar = tk.Frame(panel, bg=BG_CARD, padx=12, pady=10, highlightbackground=BORDER_COLOR, highlightthickness=1)
        bar.pack(fill="x", pady=(0, 12))

        btn_refresh = tk.Button(
            bar, text="🔄 Refresh Logs", font=("Segoe UI", 10, "bold"),
            bg="#374151", fg=TEXT_MAIN, bd=0, padx=12, pady=5, cursor="hand2",
            command=self.refresh_logs_table
        )
        btn_refresh.pack(side="left", padx=(0, 8))

        btn_csv = tk.Button(
            bar, text="📥 Export CSV Report", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_GREEN, fg="#FFFFFF", bd=0, padx=14, pady=5, cursor="hand2",
            command=self.export_csv_logs
        )
        btn_csv.pack(side="left")

        btn_clear = tk.Button(
            bar, text="🗑 Clear History", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_RED, fg="#FFFFFF", bd=0, padx=12, pady=5, cursor="hand2",
            command=self.clear_logs_history
        )
        btn_clear.pack(side="right")

        # Table
        self.log_table = LogTableWidget(panel)
        self.log_table.pack(fill="both", expand=True)

        return panel

    def refresh_logs_table(self):
        """Refreshes attendance logs table."""
        logs = self.db.get_logs()
        self.log_table.populate(logs)

    def export_csv_logs(self):
        """Exports logs to user selected CSV file."""
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not csv_path:
            return

        if self.db.export_logs_to_csv(csv_path):
            messagebox.showinfo("Export Successful", f"Saved attendance logs to:\n{csv_path}")
        else:
            messagebox.showerror("Export Failed", "Could not export CSV file.")

    def clear_logs_history(self):
        """Clears log history."""
        if messagebox.askyesno("Confirm", "Clear all recognition event logs?"):
            self.db.clear_logs()
            self.refresh_logs_table()

    # -------------------------------------------------------------------------
    # TAB 5: SETTINGS
    # -------------------------------------------------------------------------
    def _create_settings_panel(self) -> tk.Frame:
        panel = tk.LabelFrame(self.content_area, text="System Configuration", font=("Segoe UI", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN, padx=16, pady=16)

        # Slider: Confidence Threshold
        tk.Label(panel, text="Recognition Confidence Threshold:", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", pady=(8, 2))
        self.slider_conf = tk.Scale(
            panel, from_=0.30, to=0.90, resolution=0.05, orient="horizontal",
            bg=BG_CARD, fg=TEXT_MAIN, highlightthickness=0, troughcolor="#11131C",
            command=self.on_confidence_change
        )
        self.slider_conf.set(self.recognizer.confidence_threshold)
        self.slider_conf.pack(fill="x", pady=(0, 16))

        # Slider: Detection Scale Factor
        tk.Label(panel, text="Face Detection Min Neighbors:", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", pady=(8, 2))
        self.slider_neighbors = tk.Scale(
            panel, from_=3, to=10, resolution=1, orient="horizontal",
            bg=BG_CARD, fg=TEXT_MAIN, highlightthickness=0, troughcolor="#11131C",
            command=self.on_neighbors_change
        )
        self.slider_neighbors.set(self.detector.min_neighbors)
        self.slider_neighbors.pack(fill="x", pady=(0, 16))

        return panel

    def on_confidence_change(self, val):
        conf = float(val)
        self.recognizer.confidence_threshold = conf

    def on_neighbors_change(self, val):
        self.detector.min_neighbors = int(float(val))

    # -------------------------------------------------------------------------
    # MAIN VIDEO STREAM LOOP & EVENT LOGGING
    # -------------------------------------------------------------------------
    def _update_video_loop(self):
        """Periodic loop to grab frame from CameraStream and update Live Stream canvas."""
        if self.is_streaming:
            ret, frame = self.camera_stream.read()
            if ret and frame is not None:
                annotated, detections = self.video_processor.process_frame(frame)
                self.live_canvas.update_image(annotated)
                self.lbl_fps.config(text=f"FPS: {self.camera_stream.get_fps()}")

                # Process detections for card stats & event logging
                now = time.time()
                total_dets = len(detections)
                known_cnt = 0
                unknown_cnt = 0

                for det in detections:
                    name = det["name"]
                    conf = det["confidence"]
                    crop = det["face_roi"]

                    if name != "Unknown":
                        known_cnt += 1
                    else:
                        unknown_cnt += 1

                    # Log event max once every 5 seconds per person
                    last_seen = self.last_log_time.get(name, 0.0)
                    if now - last_seen > 5.0:
                        self.last_log_time[name] = now
                        self.db.log_event(name, conf, crop)
                        self._log_activity(f"Recognized: {name} ({int(conf*100)}%)")

                # Update Stat Cards
                self.card_detections.update_value(str(total_dets))
                self.card_known.update_value(str(known_cnt))
                self.card_alerts.update_value(str(unknown_cnt))

        # Schedule next loop iteration (approx 30 FPS UI refresh)
        self.after(30, self._update_video_loop)

    def on_close(self):
        """Clean shutdown."""
        if self.is_streaming:
            self.camera_stream.stop()
        self.destroy()
