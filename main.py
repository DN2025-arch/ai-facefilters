import os
import sys

# Add both application folder and parent folder to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from face_recognition_app.gui.app_window import AppWindow
except ModuleNotFoundError:
    from gui.app_window import AppWindow

def main():
    """Main entry point for VisionID Face Recognition Application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(os.path.join(data_dir, "enrolled_faces"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "snapshots"), exist_ok=True)

    print("Launching VisionID Face Recognition App...")
    app = AppWindow(base_data_dir=data_dir)
    app.mainloop()

if __name__ == "__main__":
    main()
