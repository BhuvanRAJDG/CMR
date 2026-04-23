
class AttentionDetector:
    def __init__(self, fps=30, distracted_time_threshold=3.0):
        # Allow looking away for a few seconds before alerting
        self.frames_distracted = 0
        self.max_distracted_frames = int(fps * distracted_time_threshold)
        self.is_distracted = False
        self.yaw_history = []
        self.pitch_history = []

    def get_head_pose(self, face_landmarks, img_w, img_h):
        """
        Estimates head pose direction using nose, chin, and eye landmarks.
        Since we don't have a 3D model, we use 2D ratios as a simple heuristic.
        """
        # Key landmark indices for FaceLandmarker
        NOSE_TIP = 1
        CHIN = 152
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263

        nose = face_landmarks[NOSE_TIP]
        chin = face_landmarks[CHIN]
        left_eye = face_landmarks[LEFT_EYE_OUTER]
        right_eye = face_landmarks[RIGHT_EYE_OUTER]

        # Convert to pixel coordinates
        nose_x, nose_y = nose.x * img_w, nose.y * img_h
        chin_y = chin.y * img_h
        left_x = left_eye.x * img_w
        right_x = right_eye.x * img_w

        # Horizontal direction (Yaw)
        face_width = right_x - left_x
        if face_width == 0:
            return "FORWARD" # Prevent division by zero

        # Ratio of nose horizontal position relative to eyes
        # ~0.5 is forward, <0.4 is right (from camera perspective), >0.6 is left
        yaw_ratio = (nose_x - left_x) / face_width

        # Vertical direction (Pitch)
        # Ratio of nose vertical position relative to eyes and chin
        eye_y = (left_eye.y + right_eye.y) / 2 * img_h
        face_height = chin_y - eye_y
        if face_height == 0:
            return "FORWARD"
            
        pitch_ratio = (nose_y - eye_y) / face_height

        # Smooth the ratios for drastic accuracy
        self.yaw_history.append(yaw_ratio)
        self.pitch_history.append(pitch_ratio)
        if len(self.yaw_history) > 7:
            self.yaw_history.pop(0)
            self.pitch_history.pop(0)
            
        avg_yaw = sum(self.yaw_history) / len(self.yaw_history)
        avg_pitch = sum(self.pitch_history) / len(self.pitch_history)

        direction = "FORWARD"
        if avg_yaw < 0.35:
            direction = "LOOKING RIGHT"
        elif avg_yaw > 0.65:
            direction = "LOOKING LEFT"
        elif avg_pitch > 0.65:
            direction = "LOOKING DOWN"

        return direction

    def check_attention(self, direction):
        """Updates distraction state based on current direction."""
        if direction != "FORWARD":
            self.frames_distracted += 1
            if self.frames_distracted > self.max_distracted_frames:
                self.is_distracted = True
        else:
            self.frames_distracted = max(0, self.frames_distracted - 2) # Recover quickly when looking forward
            if self.frames_distracted == 0:
                self.is_distracted = False
                
        return self.is_distracted
