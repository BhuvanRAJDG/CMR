class FatigueScoring:
    def __init__(self):
        self.score = 0
        self.level = "SAFE"
        # Weights for different fatigue indicators
        self.weight_eye_closure = 2.0  # Points per frame of eye closure
        self.weight_yawn = 15.0        # Points per yawn event
        self.weight_distraction = 1.0  # Points per frame of distraction

    def update_score(self, is_drowsy, is_yawning, is_distracted):
        """
        Calculate a rolling fatigue score from 0 to 100.
        It naturally decreases over time if the driver is attentive.
        """
        # Increase score based on active events
        if is_drowsy:
            self.score += self.weight_eye_closure
        if is_yawning:
            # Yawn adds a burst of fatigue, but we throttle it slightly so it doesn't instantly max out
            self.score += (self.weight_yawn / 10.0) # Distributed over frames of the yawn
        if is_distracted:
            self.score += self.weight_distraction

        # Natural recovery (if driving safely, fatigue score slowly decreases)
        if not is_drowsy and not is_distracted and not is_yawning:
            self.score -= 0.5

        # Clamp score between 0 and 100 (keep as float for precision)
        self.score = max(0.0, min(100.0, self.score))

        # Determine level
        if self.score < 30:
            self.level = "SAFE"
        elif self.score < 70:
            self.level = "DROWSY"
        else:
            self.level = "CRITICAL FATIGUE"

        return self.score, self.level
