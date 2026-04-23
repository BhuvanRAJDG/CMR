from scipy.spatial import distance as dist

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=20):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_counter = 0
        self.is_drowsy = False
        self.ear_history = [] # For moving average

    def calculate_ear(self, eye_landmarks):
        # MediaPipe Eye Landmarks:
        # P1=0, P2=1, P3=2, P4=3, P5=4, P6=5 (Relative to the passed array)
        # EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        
        # Vertical distances
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def check_drowsiness(self, left_ear, right_ear):
        raw_avg_ear = (left_ear + right_ear) / 2.0
        
        # Moving average filter for drastic accuracy stability
        self.ear_history.append(raw_avg_ear)
        if len(self.ear_history) > 5:
            self.ear_history.pop(0)
            
        avg_ear = sum(self.ear_history) / len(self.ear_history)
        
        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
            if self.frame_counter >= self.consecutive_frames:
                self.is_drowsy = True
        else:
            self.frame_counter = 0
            self.is_drowsy = False
            
        return self.is_drowsy, avg_ear

# Eye Landmark Indices for MediaPipe FaceMesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
