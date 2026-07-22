import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Tuple, Callable

# Color Palette Definitions (Sleek Dark Theme)
BG_DARK = "#12141C"
BG_CARD = "#1E2230"
BG_SIDEBAR = "#181B26"
ACCENT_BLUE = "#3B82F6"
ACCENT_GREEN = "#10B981"
ACCENT_RED = "#EF4444"
TEXT_MAIN = "#F3F4F6"
TEXT_MUTED = "#9CA3AF"
BORDER_COLOR = "#2D3348"

class StatCard(tk.Frame):
    """Sleek metric card showing key statistics."""
    def __init__(self, parent, title: str, initial_val: str, icon_str: str = "📊", accent_color: str = ACCENT_BLUE, **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=16, pady=14, **kwargs)
        
        header_frame = tk.Frame(self, bg=BG_CARD)
        header_frame.pack(fill="x")
        
        lbl_icon = tk.Label(header_frame, text=icon_str, font=("Segoe UI Emoji", 14), bg=BG_CARD, fg=TEXT_MAIN)
        lbl_icon.pack(side="left")
        
        lbl_title = tk.Label(header_frame, text=title.upper(), font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_MUTED)
        lbl_title.pack(side="left", padx=8)
        
        self.lbl_value = tk.Label(self, text=initial_val, font=("Segoe UI", 20, "bold"), bg=BG_CARD, fg=accent_color)
        self.lbl_value.pack(anchor="w", pady=(8, 0))

    def update_value(self, new_val: str):
        self.lbl_value.config(text=str(new_val))


class VideoDisplayCanvas(tk.Frame):
    """High-performance frame displaying video feeds or static images with aspect ratio scaling."""
    def __init__(self, parent, width: int = 640, height: int = 480, **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, **kwargs)
        self.target_w = width
        self.target_h = height

        self.canvas = tk.Canvas(self, bg="#0A0C10", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._image_tk = None

    def update_image(self, bgr_image: Optional[np.ndarray]):
        """Displays OpenCV BGR numpy array image onto canvas."""
        if bgr_image is None or bgr_image.size == 0:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 300,
                self.canvas.winfo_height() // 2 or 200,
                text="📷 No Signal / Stream Inactive",
                fill=TEXT_MUTED,
                font=("Segoe UI", 13, "bold")
            )
            return

        cw = self.canvas.winfo_width() or self.target_w
        ch = self.canvas.winfo_height() or self.target_h

        if cw <= 1 or ch <= 1:
            cw, ch = self.target_w, self.target_h

        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Calculate scale ratio to fit canvas
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        if nw > 0 and nh > 0:
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(resized)
            self._image_tk = ImageTk.PhotoImage(image=pil_img)

            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._image_tk)


class LogTableWidget(tk.Frame):
    """Custom styled table view for attendance and recognition logs."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, **kwargs)

        # Style Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.Treeview",
            background=BG_CARD,
            foreground=TEXT_MAIN,
            fieldbackground=BG_CARD,
            rowheight=30,
            font=("Segoe UI", 10)
        )
        style.configure(
            "Dark.Treeview.Heading",
            background="#161923",
            foreground=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map("Dark.Treeview", background=[("selected", ACCENT_BLUE)])

        columns = ("id", "time", "name", "conf", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", style="Dark.Treeview", selectmode="browse")

        self.tree.heading("id", text="#")
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("name", text="Recognized Person")
        self.tree.heading("conf", text="Confidence")
        self.tree.heading("status", text="Status")

        self.tree.column("id", width=50, anchor="center")
        self.tree.column("time", width=160, anchor="center")
        self.tree.column("name", width=180, anchor="w")
        self.tree.column("conf", width=100, anchor="center")
        self.tree.column("status", width=130, anchor="center")

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scrollbar.pack(side="right", fill="y", pady=4, padx=(0, 4))

        # Define row tags for status colors
        self.tree.tag_configure("Known", foreground="#34D399")
        self.tree.tag_configure("Unknown Alert", foreground="#F87171")

    def populate(self, logs_data):
        """Clears and inserts log items."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for item in logs_data:
            tag = item.get("status", "Known")
            conf_display = f"{item['confidence']}%"
            self.tree.insert("", "end", values=(
                item["id"],
                item["timestamp"],
                item["name"],
                conf_display,
                item["status"]
            ), tags=(tag,))
