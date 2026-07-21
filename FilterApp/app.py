"""
Python OpenCV Face Filter GUI Application
Built with OpenCV, PIL, and Tkinter.
"""

import os
import time
import datetime
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from filters import FaceFilterManager


class FaceFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity Python OpenCV Face Filter Studio")
        self.root.geometry("1080x720")
        self.root.minsize(900, 600)

        # Style configuration (Modern Dark Theme)
        self.bg_dark = "#181825"
        self.panel_bg = "#1e1e2e"
        self.accent_color = "#89b4fa"
        self.button_bg = "#313244"
        self.text_color = "#cdd6f4"
        self.active_bg = "#45475a"

        self.root.configure(bg=self.bg_dark)

        # Initialize Filter Engine
        self.filter_manager = FaceFilterManager()

        # Webcam Setup
        self.cap = self._init_camera()
        self.is_running = True
        self.flash_effect_counter = 0

        # Create GUI Layout
        self._build_ui()
        self._bind_keyboard_shortcuts()

        # Start Video Capture Loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def _init_camera(self):
        """Attempt to open default webcam (0) or fallback camera (1)."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            messagebox.showwarning(
                "Camera Warning",
                "Could not connect to webcam. Please ensure a camera is connected and accessible."
            )
        return cap

    def _build_ui(self):
        """Construct application layout with video view and sidebar controls."""
        # Top Title Bar
        header_frame = tk.Frame(self.root, bg=self.panel_bg, height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="✨ ANTIGRAVITY FACE FILTER STUDIO",
            font=("Segoe UI", 16, "bold"),
            fg=self.accent_color,
            bg=self.panel_bg,
            pady=10
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # Main Content Container (Video left, Controls right)
        main_container = tk.Frame(self.root, bg=self.bg_dark)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Video Canvas Container
        self.video_frame = tk.Frame(main_container, bg="#000000", bd=2, relief=tk.RIDGE)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Control Sidebar Panel
        sidebar = tk.Frame(main_container, bg=self.panel_bg, width=320, bd=1, relief=tk.SOLID)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        sidebar.pack_propagate(False)

        # --- Section 1: Filter Selection ---
        section_filter = tk.LabelFrame(
            sidebar,
            text=" 🎨 Select Face Filter ",
            font=("Segoe UI", 11, "bold"),
            fg=self.accent_color,
            bg=self.panel_bg,
            padx=10, pady=10
        )
        section_filter.pack(fill=tk.X, padx=15, pady=10)

        self.filter_buttons = {}
        for idx, filter_name in enumerate(self.filter_manager.filter_names):
            btn = tk.Button(
                section_filter,
                text=f"{idx+1}. {filter_name}",
                font=("Segoe UI", 9),
                anchor="w",
                bg=self.active_bg if filter_name == self.filter_manager.current_filter else self.button_bg,
                fg=self.text_color,
                activebackground=self.accent_color,
                activeforeground="#000000",
                bd=0,
                padx=10,
                pady=5,
                command=lambda f=filter_name: self.change_filter(f)
            )
            btn.pack(fill=tk.X, pady=2)
            self.filter_buttons[filter_name] = btn

        # --- Section 2: Adjustments ---
        section_adj = tk.LabelFrame(
            sidebar,
            text=" ⚙️ Filter & Image Controls ",
            font=("Segoe UI", 11, "bold"),
            fg=self.accent_color,
            bg=self.panel_bg,
            padx=10, pady=10
        )
        section_adj.pack(fill=tk.X, padx=15, pady=10)

        # Filter Scale Slider
        tk.Label(section_adj, text="Filter Scale Size:", fg=self.text_color, bg=self.panel_bg, font=("Segoe UI", 9)).pack(anchor="w")
        self.scale_slider = tk.Scale(
            section_adj, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg=self.panel_bg, fg=self.text_color, highlightthickness=0,
            command=self._on_scale_change
        )
        self.scale_slider.set(1.0)
        self.scale_slider.pack(fill=tk.X, pady=(0, 8))

        # Brightness Slider
        tk.Label(section_adj, text="Brightness:", fg=self.text_color, bg=self.panel_bg, font=("Segoe UI", 9)).pack(anchor="w")
        self.bright_slider = tk.Scale(
            section_adj, from_=-100, to=100, orient=tk.HORIZONTAL,
            bg=self.panel_bg, fg=self.text_color, highlightthickness=0,
            command=self._on_brightness_change
        )
        self.bright_slider.set(0)
        self.bright_slider.pack(fill=tk.X, pady=(0, 8))

        # Contrast Slider
        tk.Label(section_adj, text="Contrast:", fg=self.text_color, bg=self.panel_bg, font=("Segoe UI", 9)).pack(anchor="w")
        self.contrast_slider = tk.Scale(
            section_adj, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg=self.panel_bg, fg=self.text_color, highlightthickness=0,
            command=self._on_contrast_change
        )
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill=tk.X, pady=(0, 8))

        # Reset Adjustments Button
        reset_btn = tk.Button(
            section_adj,
            text="🔄 Reset Adjustments",
            font=("Segoe UI", 9),
            bg=self.button_bg,
            fg=self.text_color,
            bd=0, pady=4,
            command=self.reset_adjustments
        )
        reset_btn.pack(fill=tk.X, pady=5)

        # --- Section 3: Actions ---
        section_actions = tk.Frame(sidebar, bg=self.panel_bg)
        section_actions.pack(fill=tk.X, padx=15, pady=15)

        snap_btn = tk.Button(
            section_actions,
            text="📷 TAKE SNAPSHOT (Space)",
            font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1", # Catppuccin Green
            fg="#11111b",
            activebackground="#94e2d5",
            bd=0, pady=10,
            cursor="hand2",
            command=self.take_snapshot
        )
        snap_btn.pack(fill=tk.X, pady=5)

        # Footer Status Bar
        self.status_label = tk.Label(
            self.root,
            text="Ready. Press Space to take a photo.",
            bd=1, relief=tk.SUNKEN, anchor="w",
            font=("Segoe UI", 9),
            bg=self.panel_bg, fg=self.accent_color,
            padx=15, pady=5
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keyboard_shortcuts(self):
        """Bind keyboard keys for quick navigation."""
        self.root.bind("<space>", lambda e: self.take_snapshot())
        self.root.bind("<Escape>", lambda e: self.on_closing())
        self.root.bind("q", lambda e: self.on_closing())

        # Number keys 1-9 for filters
        for i in range(1, min(10, len(self.filter_manager.filter_names) + 1)):
            self.root.bind(str(i), lambda e, idx=i-1: self.change_filter(self.filter_manager.filter_names[idx]))

    def change_filter(self, filter_name):
        """Switch active filter and highlight selected button."""
        self.filter_manager.set_filter(filter_name)
        for name, btn in self.filter_buttons.items():
            if name == filter_name:
                btn.configure(bg=self.active_bg, fg=self.accent_color, font=("Segoe UI", 9, "bold"))
            else:
                btn.configure(bg=self.button_bg, fg=self.text_color, font=("Segoe UI", 9))
        self.status_label.config(text=f"Filter changed to: {filter_name}")

    def _on_scale_change(self, val):
        self.filter_manager.scale_factor = float(val)

    def _on_brightness_change(self, val):
        self.filter_manager.brightness = int(val)

    def _on_contrast_change(self, val):
        self.filter_manager.contrast = float(val)

    def reset_adjustments(self):
        self.scale_slider.set(1.0)
        self.bright_slider.set(0)
        self.contrast_slider.set(1.0)
        self.filter_manager.scale_factor = 1.0
        self.filter_manager.brightness = 0
        self.filter_manager.contrast = 1.0
        self.status_label.config(text="Adjustments reset to default.")

    def take_snapshot(self):
        """Capture current processed frame and save to disk with camera shutter effect."""
        if hasattr(self, 'current_processed_frame') and self.current_processed_frame is not None:
            snapshots_dir = "snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(snapshots_dir, f"snapshot_{timestamp}.png")

            # Trigger flash effect
            self.flash_effect_counter = 4

            # Save image
            cv2.imwrite(filename, self.current_processed_frame)
            self.status_label.config(text=f"📸 Snapshot saved: {filename}")

    def update_frame(self):
        """Video processing loop called repeatedly by Tkinter mainloop."""
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)

            # Apply active face filter
            processed_frame = self.filter_manager.apply_filter(frame)
            self.current_processed_frame = processed_frame.copy()

            # Handle photo shutter flash effect
            if self.flash_effect_counter > 0:
                processed_frame = np.full(processed_frame.shape, 255, dtype=np.uint8)
                self.flash_effect_counter -= 1

            # Convert BGR to RGB for PIL / Tkinter
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            # Resize nicely to fit canvas
            canvas_w = max(400, self.video_frame.winfo_width())
            canvas_h = max(300, self.video_frame.winfo_height())
            img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            # Display camera error placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "No Webcam Stream Available", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            rgb_frame = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Schedule next frame refresh (~30 FPS)
        self.root.after(30, self.update_frame)

    def on_closing(self):
        """Clean shutdown handler."""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceFilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
