"""
vision.py - SafeDrive AI
Handles face landmark detection using MediaPipe FaceLandmarker (Tasks API).
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import urllib.request
import os

# Download the face landmark model if it doesn't exist
MODEL_PATH = "face_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model (first time only)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete!")

# Eye landmark indices for 478-point MediaPipe face mesh (FaceLandmarker model)
# These are the standard indices used for EAR calculation
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

class FaceMeshDetector:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API).

    Usage:
        detector  = FaceMeshDetector()
        landmarks = detector.find_face_landmarks(frame)   # list or None
        coords    = detector.get_eye_coords(landmarks, LEFT_EYE_INDICES, w, h)
    """

    def __init__(self):
        download_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        face_landmarker_options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(face_landmarker_options)

    def find_face_landmarks(self, image_bgr):
        """Returns MediaPipe face landmarks list for the first face, or None."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def get_eye_coords(self, landmarks, eye_indices, img_w, img_h):
        """Convert normalized landmarks to pixel coordinates for given indices."""
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append((int(lm.x * img_w), int(lm.y * img_h)))
        return np.array(coords)
