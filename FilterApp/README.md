# 🎭 Python OpenCV Face Filter Studio

A user-friendly desktop application built in Python using OpenCV, NumPy, PIL, and Tkinter. It applies real-time face tracking AR filters and stylized artistic effects to live webcam video feeds.

---

## 🌟 Key Features

- **Real-Time AR Face Detection**: Automatically detects face and eye positions using built-in OpenCV Haar Cascades.
- **11 Visual Filters**:
  1. **Sunglasses & Mustache**: Cool dark shades with metallic highlights and a classic mustache.
  2. **Cyberpunk Visor**: Glowing cyan & magenta HUD visor with reticles and tech data.
  3. **Dog Ears & Nose**: Cute puppy ears, dog nose, and sticking-out tongue.
  4. **Bunny Ears & Nose**: Tall fluffy white bunny ears with pink inner ears, pink nose, and cute whiskers.
  5. **Neon Crown**: Glowing golden crown with gemstones hovering over your head.
  6. **Cartoon Comic**: Pop-art comic book edge lines and color quantization.
  7. **Thermal Vision**: Sci-fi heat signature color mapping.
  8. **Cyber Neon Glow**: Glowing pink neon edge detection on dark background.
  9. **Vintage Sepia**: Warm retro 80s film color transform.
  10. **Pixelate Anonymizer**: Blurs/pixelates face for privacy.
  11. **Normal Mode**: Standard unaltered camera feed.
- **User-Friendly GUI**: Sleek dark-mode interface built with Tkinter.
- **Fine Control Sliders**: Adjust filter scale, image brightness, and contrast in real time.
- **Photo Snapshots**: Click to take a photo with a realistic camera shutter flash effect, saved directly to the `snapshots/` folder.
- **Keyboard Shortcuts**: Quick hotkeys for hands-free operation while posing.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

Ensure Python 3.8+ is installed on your system, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Launch the desktop app:

```bash
python app.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `Space` | Take Snapshot (Save photo to `snapshots/`) |
| `1` - `9` | Quick-switch filters |
| `Q` or `Esc` | Quit application |

---

## 📁 File Structure

```text
├── app.py           # Tkinter GUI application & camera loop
├── filters.py       # OpenCV face detection & AR filter engine
├── requirements.txt # Project dependencies
├── README.md        # User documentation
└── snapshots/       # Captured photos (automatically created)
```
