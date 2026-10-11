"""
video_recorder.py: Manages video recording of Tkinter GUI simulation widgets using PIL & OpenCV.
"""

import os
import datetime
import numpy as np
import cv2
from PIL import ImageGrab
import tkinter as tk
import ctypes

class VideoRecorder:
    def __init__(self):
        self.is_recording = False
        self.writer = None
        self.output_path = None
        self.frame_count = 0
        self.fps = 20
        self.target_size = None
        self._ensure_dpi_awareness()

    def _ensure_dpi_awareness(self):
        """Enable Per-Monitor DPI awareness on Windows if available."""
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _get_dpi_scale(self, widget):
        """Compute DPI scaling factor between Tkinter coordinates and screen pixels."""
        try:
            # Physical screen width via Windows API
            hdc = ctypes.windll.user32.GetDC(0)
            phys_width = ctypes.windll.gdi32.GetDeviceCaps(hdc, 118) # DESKTOPHORZRES
            ctypes.windll.user32.ReleaseDC(0, hdc)
            tk_width = widget.winfo_screenwidth()
            if tk_width > 0 and phys_width > 0:
                return phys_width / tk_width
        except Exception:
            pass
        return 1.0

    def start_recording(self, widget, output_path=None, fps=20):
        if self.is_recording:
            return None

        # Create recordings folder if needed
        os.makedirs("recordings", exist_ok=True)

        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("recordings", f"sim_recording_{timestamp}.mp4")

        self.output_path = output_path
        self.fps = fps
        self.frame_count = 0
        self.writer = None
        self.target_size = None
        self.is_recording = True

        # Capture first frame to determine video dimensions
        self.capture_frame(widget)
        return self.output_path

    def capture_frame(self, widget):
        if not self.is_recording or widget is None:
            return

        try:
            widget.update_idletasks()
            scale = self._get_dpi_scale(widget)

            x = int(widget.winfo_rootx() * scale)
            y = int(widget.winfo_rooty() * scale)
            w = int(widget.winfo_width() * scale)
            h = int(widget.winfo_height() * scale)

            if w <= 0 or h <= 0:
                return

            # Ensure even dimensions for video codecs
            w = w if w % 2 == 0 else w - 1
            h = h if h % 2 == 0 else h - 1

            bbox = (x, y, x + w, y + h)
            screenshot = ImageGrab.grab(bbox=bbox, all_screens=True)

            if screenshot is None:
                return

            frame_rgb = np.array(screenshot)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Lazy initialize writer with exact frame shape
            if self.writer is None:
                img_h, img_w = frame_bgr.shape[:2]
                img_w = img_w if img_w % 2 == 0 else img_w - 1
                img_h = img_h if img_h % 2 == 0 else img_h - 1
                self.target_size = (img_w, img_h)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.target_size)

            # Resize if slightly mismatched
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.target_size:
                frame_bgr = cv2.resize(frame_bgr, self.target_size)

            self.writer.write(frame_bgr)
            self.frame_count += 1

        except Exception as e:
            print(f"[VideoRecorder] Frame capture error: {e}")

    def stop_recording(self):
        if not self.is_recording:
            return None, 0

        self.is_recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        saved_path = self.output_path
        total_frames = self.frame_count
        return saved_path, total_frames
