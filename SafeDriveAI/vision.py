import os
import cv2
import time
import math
import numpy as np
import threading
import collections
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

import config

# Absolute path to the model — works regardless of working directory
_THIS_DIR   = Path(__file__).parent.resolve()
_MODEL_PATH = str(_THIS_DIR / "face_landmarker.task")

# ── Face landmark indices ──────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308, 82,  87,  312, 317]


# ══════════════════════════════════════════════════════════════════════════════
# _FrameGrabber  — dedicated thread for latency-free frame acquisition
# ══════════════════════════════════════════════════════════════════════════════

class _FrameGrabber(threading.Thread):
    """
    Continuously reads frames from a cv2.VideoCapture in a background thread.

    Why: cv2.VideoCapture.read() blocks until the next frame arrives from the
    camera driver.  If the main vision thread calls read() directly it wastes
    processing time waiting.  This grabber keeps the buffer fresh so the
    vision thread always gets the *latest* frame instantly without blocking.
    """

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="FrameGrabber")
        self.cap          = cap
        self._lock        = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ret         = False
        self.running      = True
        self.grab_count   = 0

    # ── background loop ───────────────────────────────────────────────────────
    def run(self):
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame
                self.grab_count += 1

    # ── non-blocking read called by vision thread ─────────────────────────────
    def read(self):
        """Returns (bool, frame_copy).  Never blocks."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════════════════
# VisionMonitor
# ══════════════════════════════════════════════════════════════════════════════

class VisionMonitor:
    def __init__(self):
        self.status  = "NO_FACE"
        self.running = False
        self.thread  = None
        self.current_frame = None

        # ── EAR / drowsiness ──────────────────────────────────────────────────
        self.eyes_closed_start_time = None

        # ── PERCLOS (60-second sliding window) ───────────────────────────────
        self.perclos_history = collections.deque(maxlen=600)
        self.perclos_score   = 0.0

        # ── Blink tracking ───────────────────────────────────────────────────
        self.blink_timestamps = collections.deque()
        self.blink_rate       = 0
        self.eye_was_closed   = False

        # ── Yawn tracking ────────────────────────────────────────────────────
        self.yawn_start_time = None
        self.is_yawning      = False

        # ── Distraction tracking ─────────────────────────────────────────────
        self.distracted_start_time = None

        # ── Metrics exposed to dashboard ─────────────────────────────────────
        self.avg_ear    = 0.0
        self.head_angle = 0.0
        self.mar        = 0.0
        self.yaw_angle  = 0.0

        # ── Health data from sensors (updated by logic loop) ─────────────────
        self._health_lock = threading.Lock()
        self._health_data = {
            "hr": 0, "spo2": 0, "ecg_status": "NORMAL",
            "hrv": 0.0, "state": "NORMAL"
        }

        # ── Camera state ─────────────────────────────────────────────────────
        self.camera_connected     = False
        self.camera_resolution    = (0, 0)
        self.camera_source_label  = "Initializing…"
        self.fps                  = 0.0
        self._fps_times: collections.deque = collections.deque(maxlen=30)
        self._grabber: _FrameGrabber | None = None

        # ── MediaPipe FaceLandmarker ──────────────────────────────────────────
        print(f"  [Vision] Loading model from: {_MODEL_PATH}")
        print(f"  [Vision] Model exists: {Path(_MODEL_PATH).exists()}")
        try:
            base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("  [Vision] FaceLandmarker loaded OK")
        except Exception as e:
            print(f"  [Vision] FaceLandmarker load FAILED: {e}")
            self.detector = None

    # ── Public getters ────────────────────────────────────────────────────────

    def get_status(self):
        return self.status

    def get_perclos(self):
        return self.perclos_score

    def get_blink_rate(self):
        return self.blink_rate

    def get_fps(self):
        return self.fps

    def get_camera_info(self):
        return {
            "connected":  self.camera_connected,
            "resolution": self.camera_resolution,
            "source":     self.camera_source_label,
            "fps":        self.fps,
        }

    def update_health_data(self, sensor_data: dict, logic_state: str = "NORMAL"):
        """Called by the logic loop to push live sensor values into the HUD."""
        with self._health_lock:
            self._health_data["hr"]         = sensor_data.get("hr", 0)
            self._health_data["spo2"]       = sensor_data.get("spo2", 0)
            self._health_data["ecg_status"] = sensor_data.get("ecg_status", "NORMAL")
            self._health_data["hrv"]        = sensor_data.get("hrv", 0.0)
            self._health_data["state"]      = logic_state

    # ── Thread control ────────────────────────────────────────────────────────

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self._grabber:
            self._grabber.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    # ── Metric helpers ────────────────────────────────────────────────────────

    def _calculate_ear(self, landmarks, eye_indices):
        p = [landmarks[i] for i in eye_indices]

        def dist(a, b):
            return math.hypot(a.x - b.x, a.y - b.y)

        v1 = dist(p[1], p[5])
        v2 = dist(p[2], p[4])
        h  = dist(p[0], p[3])
        return (v1 + v2) / (2.0 * h) if h else 0.0

    def _calculate_mar(self, landmarks):
        p = [landmarks[i] for i in MOUTH]

        def dist(a, b):
            return math.hypot(a.x - b.x, a.y - b.y)

        v1 = dist(p[4], p[5])
        v2 = dist(p[6], p[7])
        v3 = dist(p[0], p[1])
        h  = dist(p[2], p[3])
        return (v1 + v2 + v3) / (3.0 * h) if h else 0.0

    def _calculate_head_angles(self, transformation_matrix):
        rmat = transformation_matrix[:3, :3]
        sy   = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy >= 1e-6:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
        return abs(math.degrees(x)), abs(math.degrees(y))

    # ── PERCLOS ───────────────────────────────────────────────────────────────

    def _update_perclos(self, eyes_closed):
        now = time.time()
        self.perclos_history.append((now, eyes_closed))
        cutoff = now - config.PERCLOS_WINDOW
        while self.perclos_history and self.perclos_history[0][0] < cutoff:
            self.perclos_history.popleft()
        if self.perclos_history:
            closed = sum(1 for _, c in self.perclos_history if c)
            self.perclos_score = closed / len(self.perclos_history)
        else:
            self.perclos_score = 0.0

    # ── Blink rate ────────────────────────────────────────────────────────────

    def _update_blink_rate(self, eyes_closed):
        now = time.time()
        if self.eye_was_closed and not eyes_closed:
            self.blink_timestamps.append(now)
        self.eye_was_closed = eyes_closed
        cutoff = now - config.BLINK_WINDOW
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()
        self.blink_rate = len(self.blink_timestamps)

    # ── Camera openers ────────────────────────────────────────────────────────

    def _open_webcam(self):
        """Opens local USB / built-in webcam with optimised settings."""
        idx         = getattr(config, 'WEBCAM_INDEX',   1)
        backend_str = getattr(config, 'WEBCAM_BACKEND', 'AUTO').upper()
        backend_map = {
            'AUTO':  cv2.CAP_ANY,
            'DSHOW': cv2.CAP_DSHOW,   # Windows DirectShow — lowest latency on Win
            'MSMF':  cv2.CAP_MSMF,    # Windows Media Foundation
            'V4L2':  cv2.CAP_V4L2,    # Linux V4L2
        }
        backend = backend_map.get(backend_str, cv2.CAP_ANY)

        try:
            cam = cv2.VideoCapture(idx, backend)
            if not cam.isOpened():
                raise RuntimeError(f"Could not open webcam index {idx}.")

            # Request resolution / FPS (camera driver may silently clamp values)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  getattr(config, 'WEBCAM_WIDTH',  640))
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(config, 'WEBCAM_HEIGHT', 480))
            cam.set(cv2.CAP_PROP_FPS,          getattr(config, 'WEBCAM_FPS',     30))
            cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # keep buffer minimal → lowest lag

            # Warm-up: discard initial dark / calibrating frames
            for _ in range(5):
                cam.read()

            # Read back what the hardware actually delivered
            aw  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah  = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = cam.get(cv2.CAP_PROP_FPS) or getattr(config, 'WEBCAM_FPS', 30)

            self.camera_resolution   = (aw, ah)
            self.camera_source_label = f"Webcam #{idx}  {aw}×{ah} @ {afps:.0f} fps"
            print(f"  [Camera] {self.camera_source_label}")
            return cam

        except Exception as e:
            print(f"  [Camera] Webcam #{idx} error: {e}")
            return None

    def _open_esp32(self):
        """Opens the MJPEG stream from an ESP32-CAM module."""
        url = f"http://{config.ESP32_CAM_IP}:81/stream"
        try:
            cam = cv2.VideoCapture(url)
            if not cam.isOpened():
                raise RuntimeError("Stream URL could not be opened.")
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(30):  # flush stale MJPEG frames
                cam.read()
            aw = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_resolution   = (aw, ah)
            self.camera_source_label = f"ESP32-CAM  {config.ESP32_CAM_IP}"
            print(f"  [Camera] {self.camera_source_label}")
            return cam
        except Exception as e:
            print(f"  [Camera] ESP32-CAM failed ({url}): {e}")
            return None

    def _open_camera(self):
        """Routes to the correct camera based on config.CAMERA_SOURCE."""
        source = getattr(config, 'CAMERA_SOURCE', 'ESP32').upper()
        if source == 'WEBCAM':
            return self._open_webcam()
        cam = self._open_esp32()
        if cam is None:
            print("  [Camera] ESP32 unavailable — falling back to webcam.")
            return self._open_webcam()
        return cam

    # ── HUD overlay renderer ──────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, landmarks=None) -> np.ndarray:
        """
        3-panel HUD matching the reference design:
          LEFT  — DRIVER STATUS (EAR, state, PERCLOS, blinks)
          CENTER — emergency banner (only when alert/warning)
          RIGHT  — HEALTH VITALS (HR, SpO2, ECG)
        """
        h, w = frame.shape[:2]
        F    = cv2.FONT_HERSHEY_DUPLEX
        FS   = cv2.FONT_HERSHEY_SIMPLEX

        ORANGE = (0, 165, 255)
        GREEN  = (0, 220, 60)
        RED    = (0, 50, 230)
        WHITE  = (255, 255, 255)
        YELLOW = (0, 220, 220)
        GREY   = (160, 160, 160)
        CYAN   = (255, 220, 0)

        # ── Pick status colour ────────────────────────────────────────────────
        EMRG = {"EYES_CLOSED", "MICROSLEEP", "DRIVER_ASLEEP",
                "CARDIAC_EMERGENCY", "ACCIDENT", "MEDICAL_SHOCK"}
        WARN = {"YAWNING", "EYES_CLOSING", "HEAD_DROOPING", "DISTRACTED"}

        with self._health_lock:
            hd = self._health_data.copy()

        logic_state = hd.get("state", self.status)
        if logic_state in EMRG or self.status in EMRG:
            border_col = RED
        elif logic_state in WARN or self.status in WARN:
            border_col = ORANGE
        else:
            border_col = GREEN

        # ── Coloured frame border ─────────────────────────────────────────────
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), border_col, 3)

        # ── Face landmarks ────────────────────────────────────────────────────
        if landmarks:
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, CYAN, -1)
            for idx in MOUTH:
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, ORANGE, -1)
            lp = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in LEFT_EYE])
            rp = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in RIGHT_EYE])
            cv2.polylines(frame, [lp], True, CYAN,   1)
            cv2.polylines(frame, [rp], True, CYAN,   1)

        # ═════════════════════════════════════════════════════════════════════
        # LEFT PANEL — DRIVER STATUS
        # ═════════════════════════════════════════════════════════════════════
        PW = 230   # panel width
        PH = 220   # panel height
        roi = frame[0:PH, 0:PW]
        dark = np.zeros_like(roi)
        frame[0:PH, 0:PW] = cv2.addWeighted(roi, 0.3, dark, 0.7, 0)
        # orange title
        cv2.putText(frame, "DRIVER STATUS", (10, 26), F, 0.62, ORANGE, 2)
        cv2.line(frame, (8, 32), (PW - 8, 32), ORANGE, 1)

        # EAR
        ear_col = RED if self.avg_ear < config.EAR_THRESHOLD and self.avg_ear > 0 else WHITE
        cv2.putText(frame, f"EAR  : {self.avg_ear:.3f}", (10, 58), FS, 0.55, ear_col, 1)

        # MAR
        mar_col = YELLOW if self.mar > config.MAR_THRESHOLD else WHITE
        cv2.putText(frame, f"MAR  : {self.mar:.3f}",    (10, 82), FS, 0.55, mar_col, 1)

        # PERCLOS
        cv2.putText(frame, f"PERCLOS: {self.perclos_score*100:.1f}%", (10, 106), FS, 0.52, GREY, 1)

        # Blink rate
        blink_col = YELLOW if self.blink_rate < 8 or self.blink_rate > 28 else WHITE
        cv2.putText(frame, f"BLINKS : {self.blink_rate}/min", (10, 128), FS, 0.52, blink_col, 1)

        # Pitch / Yaw
        cv2.putText(frame, f"PITCH  : {self.head_angle:.1f} deg", (10, 152), FS, 0.50, GREY, 1)
        cv2.putText(frame, f"YAW    : {self.yaw_angle:.1f} deg",  (10, 173), FS, 0.50, GREY, 1)

        # STATE label
        st_disp = self.status if self.status != "ALERT" else "NORMAL"
        st_col  = GREEN if st_disp == "NORMAL" else (RED if st_disp in EMRG else ORANGE)
        cv2.putText(frame, f"STATE: {st_disp}", (10, 202), F, 0.60, st_col, 2)

        # ═════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — HEALTH VITALS
        # ═════════════════════════════════════════════════════════════════════
        rx = w - PW
        roi_r = frame[0:PH, rx:w]
        dark_r = np.zeros_like(roi_r)
        frame[0:PH, rx:w] = cv2.addWeighted(roi_r, 0.3, dark_r, 0.7, 0)

        cv2.putText(frame, "HEALTH VITALS", (rx + 8, 26), F, 0.62, ORANGE, 2)
        cv2.line(frame, (rx + 8, 32), (w - 8, 32), ORANGE, 1)

        hr_val  = hd.get("hr", 0)
        spo2_val = hd.get("spo2", 0)
        ecg_val  = hd.get("ecg_status", "NORMAL")
        hrv_val  = hd.get("hrv", 0.0)

        hr_col = RED if hr_val > config.HR_MAX or (0 < hr_val < config.HR_MIN) else WHITE
        sp_col = RED if spo2_val > 0 and spo2_val < config.SPO2_MIN else WHITE
        ecg_col = RED if ecg_val != "NORMAL" else GREEN

        cv2.putText(frame, f"Heart Rate",          (rx + 8, 60),  FS, 0.50, GREY,    1)
        cv2.putText(frame, f": {hr_val:.0f} BPM",  (rx + 8, 80),  F,  0.62, hr_col,  2)

        cv2.putText(frame, f"SpO2",                (rx + 8, 112), FS, 0.50, GREY,    1)
        cv2.putText(frame, f": {spo2_val:.1f} %",  (rx + 8, 132), F,  0.62, sp_col,  2)

        cv2.putText(frame, f"ECG",                 (rx + 8, 162), FS, 0.50, GREY,    1)
        cv2.putText(frame, f": {ecg_val}",          (rx + 8, 182), F,  0.58, ecg_col, 2)

        cv2.putText(frame, f"HRV: {hrv_val:.1f}",  (rx + 8, 210), FS, 0.48, GREY,    1)

        # ═════════════════════════════════════════════════════════════════════
        # CENTER BANNER — only for WARN / EMRG states
        # ═════════════════════════════════════════════════════════════════════
        BANNER_MESSAGES = {
            "EYES_CLOSED":        ("EYES CLOSED — WAKE UP!",        "Drowsiness detected"),
            "EYES_CLOSING":       ("EYES CLOSING",                   "Keep your eyes on the road"),
            "MICROSLEEP":         ("MICROSLEEP DETECTED!",           "Critical — pull over now"),
            "DRIVER_ASLEEP":      ("DRIVER UNRESPONSIVE!",           "Calling emergency contact"),
            "YAWNING":            ("YAWNING DETECTED",               "Fatigue warning"),
            "HEAD_DROOPING":      ("HEAD DROOPING",                  "Posture alert"),
            "DISTRACTED":         ("DISTRACTED!",                    "Eyes on road"),
            "CARDIAC_EMERGENCY":  ("HEALTH EMERGENCY DETECTED",      "Cardiac event — SMS sent"),
            "ACCIDENT":           ("CRASH DETECTED!",                "Emergency services notified"),
            "MEDICAL_SHOCK":      ("HEALTH EMERGENCY DETECTED",      "Medical shock — SMS sent"),
        }

        banner_state = logic_state if logic_state in BANNER_MESSAGES else \
                       (self.status if self.status in BANNER_MESSAGES else None)

        if banner_state:
            title, subtitle = BANNER_MESSAGES[banner_state]
            is_emrg = banner_state in EMRG
            bg_col  = (0, 0, 180) if is_emrg else (0, 100, 180)
            bdr_col = RED if is_emrg else ORANGE

            # measure text
            (tw, th), _ = cv2.getTextSize(title, F, 0.75, 2)
            bx  = (w - tw) // 2 - 18
            bx2 = bx + tw + 36
            by1 = 12;  by2 = 72

            roi_b = frame[by1:by2, max(0,bx):min(w,bx2)]
            bg    = np.full_like(roi_b, 0)
            bg[:] = bg_col
            frame[by1:by2, max(0,bx):min(w,bx2)] = cv2.addWeighted(roi_b, 0.25, bg, 0.75, 0)
            cv2.rectangle(frame, (bx, by1), (bx2, by2), bdr_col, 2)

            cv2.putText(frame, title,    (bx + 10, by1 + 28), F,  0.72, WHITE,  2)
            cv2.putText(frame, subtitle, (bx + 10, by1 + 52), FS, 0.50, YELLOW, 1)

        # ── Bottom status bar ─────────────────────────────────────────────────
        bh = 32
        bar_roi = frame[h-bh:h, 0:w]
        solid   = np.full_like(bar_roi, 0)
        solid[:] = border_col
        frame[h-bh:h, 0:w] = cv2.addWeighted(bar_roi, 0.35, solid, 0.65, 0)
        st_label = self.status if self.status != "ALERT" else "NORMAL"
        cv2.putText(frame, f"  VitalDrive AI  |  STATUS: {st_label}",
                    (6, h - 8), FS, 0.55, WHITE, 2)
        fps_txt = f"{self.fps:.1f} FPS"
        (ftw, _), _ = cv2.getTextSize(fps_txt, FS, 0.52, 1)
        cv2.putText(frame, fps_txt, (w - ftw - 10, h - 8), FS, 0.52, (100, 255, 100), 1)

        return frame

    # ── Main processing loop ──────────────────────────────────────────────────

    def _run(self):
        if not self.detector:
            print("  [Vision] FaceLandmarker unavailable — thread exiting.")
            return

        cap              = self._open_camera()
        grabber          = None
        reconnect_delay  = 1.0   # seconds; grows with exponential back-off

        def _start_grabber(c):
            g = _FrameGrabber(c)
            g.start()
            return g

        if cap and cap.isOpened():
            grabber = _start_grabber(cap)
            self.camera_connected = True

        while self.running:

            # ── Reconnect if camera is lost ───────────────────────────────────
            if cap is None or not cap.isOpened():
                self.status           = "NO_FACE"
                self.camera_connected = False
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 10.0)  # back-off
                cap = self._open_camera()
                if cap and cap.isOpened():
                    grabber = _start_grabber(cap)
                    self.camera_connected = True
                    reconnect_delay = 1.0
                continue

            # ── Grab latest frame (non-blocking via _FrameGrabber) ────────────
            ret, frame = grabber.read() if grabber else (False, None)

            if not ret or frame is None:
                self.status           = "NO_FACE"
                self.camera_connected = False
                if grabber:
                    grabber.stop()
                cap.release()
                time.sleep(1)
                cap = self._open_camera()
                if cap and cap.isOpened():
                    grabber = _start_grabber(cap)
                    self.camera_connected = True
                continue

            # ── FPS tracking ─────────────────────────────────────────────────
            t0 = time.time()

            # ── MediaPipe inference ───────────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = self.detector.detect(mp_image)

            # Reset metrics each cycle
            self.avg_ear = self.head_angle = self.mar = self.yaw_angle = 0.0

            detected_landmarks = None

            if not result.face_landmarks:
                self.status = "NO_FACE"
                self.eyes_closed_start_time = None
                self.yawn_start_time        = None
                self.distracted_start_time  = None
                self._update_perclos(False)
                self._update_blink_rate(False)

            else:
                landmarks        = result.face_landmarks[0]
                transform_matrix = result.facial_transformation_matrixes[0]
                detected_landmarks = landmarks

                left_ear  = self._calculate_ear(landmarks, LEFT_EYE)
                right_ear = self._calculate_ear(landmarks, RIGHT_EYE)
                self.avg_ear    = (left_ear + right_ear) / 2.0
                self.head_angle, self.yaw_angle = self._calculate_head_angles(
                    transform_matrix)
                self.mar = self._calculate_mar(landmarks)

                eyes_closed = self.avg_ear < config.EAR_THRESHOLD

                self._update_perclos(eyes_closed)
                self._update_blink_rate(eyes_closed)

                # ── State machine (priority order) ───────────────────────────

                # 1. MICROSLEEP
                if self.perclos_score > config.PERCLOS_THRESHOLD and eyes_closed:
                    self.status = "MICROSLEEP"
                    if self.eyes_closed_start_time is None:
                        self.eyes_closed_start_time = time.time()

                # 2. EYES_CLOSED / EYES_CLOSING
                elif eyes_closed:
                    if self.eyes_closed_start_time is None:
                        self.eyes_closed_start_time = time.time()
                    dur = time.time() - self.eyes_closed_start_time
                    self.status = "EYES_CLOSED" if dur >= config.DROWSY_SECONDS \
                                  else "EYES_CLOSING"

                # 3. YAWNING
                elif self.mar > config.MAR_THRESHOLD:
                    if self.yawn_start_time is None:
                        self.yawn_start_time = time.time()
                    if (time.time() - self.yawn_start_time) >= config.YAWN_SECONDS:
                        self.status    = "YAWNING"
                        self.is_yawning = True
                    elif self.status != "YAWNING":
                        self.status = "ALERT"

                else:
                    self.yawn_start_time = None
                    self.is_yawning      = False

                    # 4. HEAD_DROOPING
                    if self.head_angle > config.HEAD_ANGLE_THRESHOLD:
                        self.status                = "HEAD_DROOPING"
                        self.eyes_closed_start_time = None
                        self.distracted_start_time  = None

                    # 5. DISTRACTED
                    elif self.yaw_angle > config.DISTRACTION_ANGLE:
                        if self.distracted_start_time is None:
                            self.distracted_start_time = time.time()
                        dur = time.time() - self.distracted_start_time
                        self.status = "DISTRACTED" \
                            if dur >= config.DISTRACTION_SECONDS else "ALERT"

                    # 6. All clear
                    else:
                        self.status                = "ALERT"
                        self.eyes_closed_start_time = None
                        self.distracted_start_time  = None

            # ── Draw HUD overlay ─────────────────────────────────────────────
            self._draw_overlay(frame, detected_landmarks)
            self.current_frame = frame

            # ── Update live FPS ───────────────────────────────────────────────
            elapsed = time.time() - t0
            self._fps_times.append(elapsed)
            avg = sum(self._fps_times) / len(self._fps_times) if self._fps_times else 0.01
            self.fps = 1.0 / avg if avg > 0 else 0.0

            # ── Adaptive sleep: aim for ~20 fps net, account for proc time ───
            sleep_time = max(0.0, 0.05 - elapsed)
            time.sleep(sleep_time)

        # ── Cleanup ───────────────────────────────────────────────────────────
        if grabber:
            grabber.stop()
        if cap:
            cap.release()
        print("  [Vision] Camera released — thread stopped.")
