from scipy.spatial import distance as dist
import numpy as np

class YawnDetector:
    def __init__(self, mar_threshold=0.6, fps=30, yawn_time_threshold=1.0):
        self.mar_threshold = mar_threshold
        self.frames_yawning = 0
        self.max_yawning_frames = int(fps * yawn_time_threshold)
        self.is_yawning = False
        self.yawn_count = 0  # To track total yawns for fatigue score
        self.mar_history = [] # Smooth MAR readings

    def calculate_mar(self, face_landmarks, img_w, img_h):
        """
        Calculates Mouth Aspect Ratio (MAR) to detect yawning.
        """
        # Outer lip landmarks for FaceLandmarker
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291
        TOP_MOUTH = 0     # Upper lip center
        BOTTOM_MOUTH = 17 # Lower lip center

        def get_pt(idx):
            lm = face_landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        left_pt = get_pt(LEFT_MOUTH)
        right_pt = get_pt(RIGHT_MOUTH)
        top_pt = get_pt(TOP_MOUTH)
        bottom_pt = get_pt(BOTTOM_MOUTH)

        # Calculate distances
        horizontal_dist = dist.euclidean(left_pt, right_pt)
        vertical_dist = dist.euclidean(top_pt, bottom_pt)

        if horizontal_dist == 0:
            return 0.0

        mar = vertical_dist / horizontal_dist
        return mar

    def check_yawn(self, raw_mar):
        """Updates yawn state based on current MAR."""
        self.mar_history.append(raw_mar)
        if len(self.mar_history) > 5:
            self.mar_history.pop(0)

        mar = sum(self.mar_history) / len(self.mar_history)

        if mar > self.mar_threshold:
            self.frames_yawning += 1
            if self.frames_yawning >= self.max_yawning_frames:
                if not self.is_yawning:
                    self.yawn_count += 1  # Register one yawn event (count on first frame it becomes true)
                self.is_yawning = True
        else:
            self.frames_yawning = 0
            self.is_yawning = False
            
        return self.is_yawning, self.yawn_count
