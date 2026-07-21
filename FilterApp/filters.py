"""
Face Filter Engine using OpenCV and NumPy
Provides real-time face detection, AR vector overlays, and artistic visual effects.
"""

import cv2
import numpy as np
import math
import os


class FaceFilterManager:
    def __init__(self):
        # Load OpenCV pre-trained Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # Available filters list
        self.filter_names = [
            "Normal (No Filter)",
            "Sunglasses & Mustache",
            "Cyberpunk Visor",
            "Dog Ears & Nose",
            "Bunny Ears & Nose",
            "Neon Crown",
            "Cartoon Comic",
            "Thermal Vision",
            "Cyber Neon Glow",
            "Vintage Sepia",
            "Pixelate Anonymizer"
        ]

        self.current_filter = "Sunglasses & Mustache"
        self.scale_factor = 1.0
        self.brightness = 0
        self.contrast = 1.0

    def set_filter(self, filter_name):
        """Set active filter name."""
        if filter_name in self.filter_names:
            self.current_filter = filter_name

    def adjust_image(self, frame):
        """Apply user-selected brightness and contrast adjustments."""
        if self.brightness != 0 or self.contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        return frame

    def apply_filter(self, frame):
        """Process video frame and apply selected filter."""
        frame = self.adjust_image(frame)
        h, w = frame.shape[:2]

        if self.current_filter == "Normal (No Filter)":
            return frame
        elif self.current_filter == "Cartoon Comic":
            return self._apply_cartoon(frame)
        elif self.current_filter == "Thermal Vision":
            return self._apply_thermal(frame)
        elif self.current_filter == "Cyber Neon Glow":
            return self._apply_neon_glow(frame)
        elif self.current_filter == "Vintage Sepia":
            return self._apply_sepia(frame)

        # Face-detection based filters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        output_frame = frame.copy()

        for (x, y, fw, fh) in faces:
            if self.current_filter == "Sunglasses & Mustache":
                output_frame = self._draw_sunglasses_and_mustache(output_frame, gray, x, y, fw, fh)
            elif self.current_filter == "Cyberpunk Visor":
                output_frame = self._draw_cyberpunk_visor(output_frame, x, y, fw, fh)
            elif self.current_filter == "Dog Ears & Nose":
                output_frame = self._draw_dog_ears_nose(output_frame, x, y, fw, fh)
            elif self.current_filter == "Bunny Ears & Nose":
                output_frame = self._draw_bunny_ears_nose(output_frame, x, y, fw, fh)
            elif self.current_filter == "Neon Crown":
                output_frame = self._draw_neon_crown(output_frame, x, y, fw, fh)
            elif self.current_filter == "Pixelate Anonymizer":
                output_frame = self._apply_pixelate(output_frame, x, y, fw, fh)

        return output_frame

    # --- AR Vector Overlay Filters ---

    def _draw_sunglasses_and_mustache(self, img, gray_face, x, y, fw, fh):
        """Draw sleek dark sunglasses over eyes and classic mustache below nose."""
        roi_gray = gray_face[y:y + fh, x:x + fw]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Estimate eye level if eyes detected, else use face proportions
        if len(eyes) >= 2:
            # Sort by X coordinate to get left & right eye
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            eye_center_y = y + int((ey1 + ey2) / 2 + eh1 / 2)
        else:
            eye_center_y = y + int(fh * 0.35)

        # Draw Sunglasses
        glass_w = int(fw * 0.85 * self.scale_factor)
        glass_h = int(fh * 0.22 * self.scale_factor)
        gx = x + int((fw - glass_w) / 2)
        gy = eye_center_y - int(glass_h / 2)

        if gy > 0 and gx > 0 and (gx + glass_w) < img.shape[1] and (gy + glass_h) < img.shape[0]:
            overlay = img.copy()
            # Left lens & right lens rounded rectangles / ellipses
            lens_w = int(glass_w * 0.42)
            lens_gap = int(glass_w * 0.16)

            left_lens_x = gx + 5
            right_lens_x = gx + lens_w + lens_gap - 5

            # Outer frame
            cv2.rectangle(overlay, (gx, gy), (gx + glass_w, gy + glass_h), (10, 10, 10), -1)
            # Metallic highlights on lenses
            cv2.ellipse(overlay, (left_lens_x + lens_w // 2, gy + glass_h // 2),
                        (lens_w // 2 - 4, glass_h // 2 - 4), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(overlay, (right_lens_x + lens_w // 2, gy + glass_h // 2),
                        (lens_w // 2 - 4, glass_h // 2 - 4), 0, 0, 360, (50, 50, 50), -1)

            # Lens shine lines
            cv2.line(overlay, (left_lens_x + 10, gy + 8), (left_lens_x + lens_w - 15, gy + glass_h - 8), (200, 200, 200), 2)
            cv2.line(overlay, (right_lens_x + 10, gy + 8), (right_lens_x + lens_w - 15, gy + glass_h - 8), (200, 200, 200), 2)

            # Bridge
            cv2.rectangle(overlay, (gx + lens_w, gy + 5), (gx + lens_w + lens_gap, gy + 12), (0, 215, 255), -1)

            # Blend overlay with image for semi-transparency effect
            img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

        # Draw Mustache
        stache_w = int(fw * 0.55 * self.scale_factor)
        stache_h = int(fh * 0.18 * self.scale_factor)
        sx = x + int((fw - stache_w) / 2)
        sy = y + int(fh * 0.68)

        if sy > 0 and sx > 0 and (sx + stache_w) < img.shape[1] and (sy + stache_h) < img.shape[0]:
            center_x = sx + stache_w // 2
            # Curved mustache shape using filled polygons/ellipses
            pts_left = np.array([
                [center_x, sy + stache_h // 2],
                [sx + int(stache_w * 0.25), sy],
                [sx, sy + int(stache_h * 0.4)],
                [sx + int(stache_w * 0.1), sy + stache_h],
                [center_x - 5, sy + int(stache_h * 0.7)]
            ], np.int32)

            pts_right = np.array([
                [center_x, sy + stache_h // 2],
                [sx + int(stache_w * 0.75), sy],
                [sx + stache_w, sy + int(stache_h * 0.4)],
                [sx + int(stache_w * 0.9), sy + stache_h],
                [center_x + 5, sy + int(stache_h * 0.7)]
            ], np.int32)

            cv2.fillPoly(img, [pts_left], (20, 20, 20))
            cv2.fillPoly(img, [pts_right], (20, 20, 20))
            cv2.polylines(img, [pts_left], True, (0, 0, 0), 2)
            cv2.polylines(img, [pts_right], True, (0, 0, 0), 2)

        return img

    def _draw_cyberpunk_visor(self, img, x, y, fw, fh):
        """Draw futuristic glowing cyan & magenta HUD visor over eyes."""
        visor_w = int(fw * 1.1 * self.scale_factor)
        visor_h = int(fh * 0.3 * self.scale_factor)
        vx = x + int((fw - visor_w) / 2)
        vy = y + int(fh * 0.25)

        overlay = img.copy()

        # Hexagon/Angular Visor Shape
        pts = np.array([
            [vx, vy + int(visor_h * 0.3)],
            [vx + int(visor_w * 0.15), vy],
            [vx + int(visor_w * 0.85), vy],
            [vx + visor_w, vy + int(visor_h * 0.3)],
            [vx + int(visor_w * 0.9), vy + visor_h],
            [vx + int(visor_w * 0.1), vy + visor_h]
        ], np.int32)

        # Fill with translucent neon cyan
        cv2.fillPoly(overlay, [pts], (255, 255, 0)) # Cyan in BGR
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

        # Glowing Neon Border (Magenta & Cyan)
        cv2.polylines(img, [pts], True, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Tech Target Reticles & Grid lines
        center_x = vx + visor_w // 2
        center_y = vy + visor_h // 2
        cv2.circle(img, (center_x - int(visor_w * 0.22), center_y), 14, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (center_x + int(visor_w * 0.22), center_y), 14, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (vx + 10, vy + visor_h // 2), (vx + visor_w - 10, vy + visor_h // 2), (0, 255, 255), 1)

        # HUD Text
        cv2.putText(img, "SYS.OK // 100%", (vx + 15, vy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return img

    def _draw_dog_ears_nose(self, img, x, y, fw, fh):
        """Draw puppy ears atop head and dog nose button."""
        # Dog ears positions above face
        ear_w = int(fw * 0.38 * self.scale_factor)
        ear_h = int(fh * 0.55 * self.scale_factor)

        # Left Ear
        left_ear_pts = np.array([
            [x - int(ear_w * 0.2), y + int(fh * 0.1)],
            [x - int(ear_w * 0.5), y - int(ear_h * 0.7)],
            [x + int(ear_w * 0.5), y - int(ear_h * 0.3)]
        ], np.int32)

        # Right Ear
        right_ear_pts = np.array([
            [x + fw + int(ear_w * 0.2), y + int(fh * 0.1)],
            [x + fw + int(ear_w * 0.5), y - int(ear_h * 0.7)],
            [x + fw - int(ear_w * 0.5), y - int(ear_h * 0.3)]
        ], np.int32)

        # Brown ears with pink inner ear
        cv2.fillPoly(img, [left_ear_pts], (42, 75, 124)) # Brown BGR
        cv2.fillPoly(img, [right_ear_pts], (42, 75, 124))
        cv2.polylines(img, [left_ear_pts], True, (25, 45, 80), 3)
        cv2.polylines(img, [right_ear_pts], True, (25, 45, 80), 3)

        # Inner Pink Ear
        left_inner = np.array([
            [x - int(ear_w * 0.1), y],
            [x - int(ear_w * 0.35), y - int(ear_h * 0.5)],
            [x + int(ear_w * 0.3), y - int(ear_h * 0.25)]
        ], np.int32)

        right_inner = np.array([
            [x + fw + int(ear_w * 0.1), y],
            [x + fw + int(ear_w * 0.35), y - int(ear_h * 0.5)],
            [x + fw - int(ear_w * 0.3), y - int(ear_h * 0.25)]
        ], np.int32)

        cv2.fillPoly(img, [left_inner], (180, 150, 240)) # Pink BGR
        cv2.fillPoly(img, [right_inner], (180, 150, 240))

        # Dog Nose
        nose_center = (x + fw // 2, y + int(fh * 0.62))
        nose_radius = int(fw * 0.09 * self.scale_factor)
        cv2.circle(img, nose_center, nose_radius, (30, 30, 30), -1, cv2.LINE_AA)
        # Highlight on nose
        cv2.circle(img, (nose_center[0] - nose_radius // 3, nose_center[1] - nose_radius // 3),
                   max(2, nose_radius // 4), (240, 240, 240), -1, cv2.LINE_AA)

        # Cute tongue sticking out
        tongue_w = int(fw * 0.14)
        tongue_h = int(fh * 0.18)
        tx = nose_center[0] - tongue_w // 2
        ty = nose_center[1] + nose_radius + 2
        cv2.ellipse(img, (nose_center[0], ty + tongue_h // 2), (tongue_w // 2, tongue_h // 2),
                    0, 0, 180, (140, 120, 255), -1, cv2.LINE_AA)

        return img

    def _draw_bunny_ears_nose(self, img, x, y, fw, fh):
        """Draw tall fluffy white bunny ears with pink inner ears, pink nose, and cute whiskers."""
        ear_w = int(fw * 0.32 * self.scale_factor)
        ear_h = int(fh * 0.9 * self.scale_factor)

        # Base Y level above face
        base_y = y - int(fh * 0.05)

        # --- Left Bunny Ear (Upright) ---
        left_center_x = x + int(fw * 0.28)
        left_center_y = max(ear_h // 2, base_y - ear_h // 2)

        # Outer White Fluffy Ear
        cv2.ellipse(img, (left_center_x, left_center_y), (ear_w // 2, ear_h // 2),
                    -6, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.ellipse(img, (left_center_x, left_center_y), (ear_w // 2, ear_h // 2),
                    -6, 0, 360, (180, 180, 180), 2, cv2.LINE_AA)

        # Inner Pink Ear
        cv2.ellipse(img, (left_center_x, left_center_y + 8),
                    (max(2, int(ear_w * 0.28)), max(5, int(ear_h * 0.38))),
                    -6, 0, 360, (200, 170, 255), -1, cv2.LINE_AA)

        # --- Right Bunny Ear (Cute angle) ---
        right_center_x = x + int(fw * 0.72)
        right_center_y = max(ear_h // 2, base_y - ear_h // 2)

        # Outer White Fluffy Ear
        cv2.ellipse(img, (right_center_x, right_center_y), (ear_w // 2, ear_h // 2),
                    8, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.ellipse(img, (right_center_x, right_center_y), (ear_w // 2, ear_h // 2),
                    8, 0, 360, (180, 180, 180), 2, cv2.LINE_AA)

        # Inner Pink Ear
        cv2.ellipse(img, (right_center_x, right_center_y + 8),
                    (max(2, int(ear_w * 0.28)), max(5, int(ear_h * 0.38))),
                    8, 0, 360, (200, 170, 255), -1, cv2.LINE_AA)

        # --- Bunny Nose (Small Pink Triangle) ---
        nose_center_x = x + fw // 2
        nose_y = y + int(fh * 0.62)
        nw = int(fw * 0.11 * self.scale_factor)
        nh = int(fh * 0.08 * self.scale_factor)

        nose_pts = np.array([
            [nose_center_x - nw // 2, nose_y - nh // 2],
            [nose_center_x + nw // 2, nose_y - nh // 2],
            [nose_center_x, nose_y + nh // 2]
        ], np.int32)
        cv2.fillPoly(img, [nose_pts], (180, 150, 255), cv2.LINE_AA)
        cv2.polylines(img, [nose_pts], True, (140, 100, 220), 1, cv2.LINE_AA)

        # --- Whiskers (3 lines on each side) ---
        w_len = int(fw * 0.32)
        # Left whiskers
        cv2.line(img, (nose_center_x - nw // 2, nose_y), (nose_center_x - nw // 2 - w_len, nose_y - 12), (60, 60, 60), 2, cv2.LINE_AA)
        cv2.line(img, (nose_center_x - nw // 2, nose_y + 4), (nose_center_x - nw // 2 - w_len, nose_y + 6), (60, 60, 60), 2, cv2.LINE_AA)
        cv2.line(img, (nose_center_x - nw // 2, nose_y + 8), (nose_center_x - nw // 2 - w_len + 5, nose_y + 22), (60, 60, 60), 2, cv2.LINE_AA)

        # Right whiskers
        cv2.line(img, (nose_center_x + nw // 2, nose_y), (nose_center_x + nw // 2 + w_len, nose_y - 12), (60, 60, 60), 2, cv2.LINE_AA)
        cv2.line(img, (nose_center_x + nw // 2, nose_y + 4), (nose_center_x + nw // 2 + w_len, nose_y + 6), (60, 60, 60), 2, cv2.LINE_AA)
        cv2.line(img, (nose_center_x + nw // 2, nose_y + 8), (nose_center_x + nw // 2 + w_len - 5, nose_y + 22), (60, 60, 60), 2, cv2.LINE_AA)

        return img

    def _draw_neon_crown(self, img, x, y, fw, fh):
        """Draw a glowing golden crown hovering over the head."""
        crown_w = int(fw * 0.9 * self.scale_factor)
        crown_h = int(fh * 0.45 * self.scale_factor)
        cx = x + int((fw - crown_w) / 2)
        cy = y - int(crown_h * 0.85)

        if cy > 0:
            pts = np.array([
                [cx, cy + crown_h],
                [cx, cy + int(crown_h * 0.3)],
                [cx + int(crown_w * 0.25), cy + int(crown_h * 0.6)],
                [cx + crown_w // 2, cy],
                [cx + int(crown_w * 0.75), cy + int(crown_h * 0.6)],
                [cx + crown_w, cy + int(crown_h * 0.3)],
                [cx + crown_w, cy + crown_h]
            ], np.int32)

            # Crown Base
            cv2.fillPoly(img, [pts], (0, 215, 255)) # Gold
            cv2.polylines(img, [pts], True, (0, 140, 255), 3, cv2.LINE_AA)

            # Jewels on Crown tips
            cv2.circle(img, (cx, cy + int(crown_h * 0.3)), 7, (255, 0, 128), -1, cv2.LINE_AA)
            cv2.circle(img, (cx + crown_w // 2, cy), 9, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx + crown_w, cy + int(crown_h * 0.3)), 7, (255, 0, 128), -1, cv2.LINE_AA)

            # Glowing sparkles around crown
            cv2.circle(img, (cx - 15, cy + 10), 3, (255, 255, 255), -1)
            cv2.circle(img, (cx + crown_w + 15, cy + 10), 3, (255, 255, 255), -1)

        return img

    # --- Artistic Image Processing Filters ---

    def _apply_cartoon(self, frame):
        """Bilateral filter + Adaptive Thresholding for Pop-Art Comic effect."""
        # Color quantization / smoothing with bilateral filter
        color = frame
        for _ in range(2):
            color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

        # Convert to grayscale and detect heavy edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=9, C=2
        )

        # Convert edges to 3-channel
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Combine smoothed color frame with black outlines
        cartoon = cv2.bitwise_and(color, edges_bgr)
        return cartoon

    def _apply_thermal(self, frame):
        """Sci-Fi Thermal / Heat vision effect using color mapping."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Contrast stretch
        gray = cv2.equalizeHist(gray)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return thermal

    def _apply_neon_glow(self, frame):
        """Cyberpunk Neon Edge Glow effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # Create neon colored channels
        h, w = frame.shape[:2]
        neon = np.zeros((h, w, 3), dtype=np.uint8)
        neon[edges_dilated > 0] = (255, 0, 255) # Pink edges

        # Add original frame dimly in background
        dim_frame = cv2.addWeighted(frame, 0.3, neon, 0.7, 0)
        return dim_frame

    def _apply_sepia(self, frame):
        """Retro Vintage Sepia transformation."""
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia_frame = cv2.transform(frame, sepia_matrix)
        sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
        return sepia_frame

    def _apply_pixelate(self, img, x, y, fw, fh):
        """Anonymize/Blur detected face with mosaic pixelation."""
        # Clamp bounding box to frame
        h, w = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            return img

        # Scale down and back up with nearest neighbor interpolation
        blocks = 12
        rh, rw = max(1, (y2 - y1) // blocks), max(1, (x2 - x1) // blocks)
        small = cv2.resize(face_roi, (rw, rh), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        img[y1:y2, x1:x2] = pixelated
        return img
