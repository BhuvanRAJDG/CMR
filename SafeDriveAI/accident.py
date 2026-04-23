import time

class AccidentDetector:
    def __init__(self, fps=30, missing_face_threshold=5.0):
        self.frames_face_missing = 0
        self.max_missing_frames = int(fps * missing_face_threshold)
        self.is_accident = False

    def check_accident(self, face_detected, is_health_emergency):
        """
        Simple heuristic for accident:
        Driver's face disappears from camera view for > 5 seconds
        AND health sensors indicate a critical condition concurrently.
        """
        if not face_detected:
            self.frames_face_missing += 1
        else:
            self.frames_face_missing = max(0, self.frames_face_missing - 5) # recover

        # Check conditions
        if self.frames_face_missing > self.max_missing_frames and is_health_emergency:
            self.is_accident = True
        else:
            self.is_accident = False

        return self.is_accident
