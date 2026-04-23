import random
import time

class HealthMonitor:
    def __init__(self):
        self.heart_rate = 75
        self.spo2 = 98
        self.ecg_status = "Normal"
        self.last_update = time.time()
        self.emergency_detected = False
        self.emergency_reason = ""
        
        # Internal target values for smooth signal tracking
        self.target_hr = 75.0
        self.target_spo2 = 98.0

    def update_sensors(self):
        # Implement a biomedical-like low-pass filter (Exponential Moving Average)
        # to simulate accurate, stable instrumentation readings that don't jitter wildly.
        
        # 1. Random walk the underlying targets slowly
        if random.random() < 0.1: # 10% chance per frame to gently shift baseline
            self.target_hr += random.uniform(-1.5, 1.5)
            self.target_spo2 += random.uniform(-0.5, 0.5)
        
        # Keep targets in realistic normal bounds (before emergencies)
        self.target_hr = max(55.0, min(100.0, self.target_hr))
        self.target_spo2 = max(94.0, min(100.0, self.target_spo2))
        
        # 2. Occasionally simulate an emergency spike/drop
        if random.random() < 0.005: 
            action = random.choice(["high_hr", "low_hr", "low_spo2"])
            if action == "high_hr":
                self.target_hr = random.uniform(130, 150)
            elif action == "low_hr":
                self.target_hr = random.uniform(35, 45)
            elif action == "low_spo2":
                self.target_spo2 = random.uniform(82, 88)

        # 3. Smooth mathematical approach (EMA) toward targets
        self.heart_rate += (self.target_hr - self.heart_rate) * 0.05
        self.spo2 += (self.target_spo2 - self.spo2) * 0.05

        # 4. Add imperceptible high-frequency sensor noise 
        self.display_hr = self.heart_rate + random.uniform(-0.5, 0.5)
        self.display_spo2 = self.spo2 + random.uniform(-0.1, 0.1)

        self.check_emergency()
        return self.get_data()

    def check_emergency(self):
        self.emergency_detected = False
        self.emergency_reason = ""
        
        if self.heart_rate > 120:
            self.emergency_detected = True
            self.emergency_reason = "High Heart Rate (Tachycardia)"
        elif self.heart_rate < 50:
            self.emergency_detected = True
            self.emergency_reason = "Low Heart Rate (Bradycardia)"
        
        if self.spo2 < 90:
            self.emergency_detected = True
            self.emergency_reason += " | Critical SpO2 Level" if self.emergency_reason else "Critical SpO2 Level"

    def get_data(self):
        return {
            "heart_rate": round(getattr(self, 'display_hr', self.heart_rate), 1),
            "spo2": round(getattr(self, 'display_spo2', self.spo2), 1),
            "ecg_status": "Normal" if not self.emergency_detected else "Abnormal",
            "emergency": self.emergency_detected,
            "reason": self.emergency_reason
        }

    def simulate_gps_alert(self):
        # Simulate sending GPS location
        lat = 37.7749 + random.uniform(-0.01, 0.01)
        lon = -122.4194 + random.uniform(-0.01, 0.01)
        return f"ALRT: Location sent to Emergency Services. Lat: {lat:.5f}, Lon: {lon:.5f}"

if __name__ == "__main__":
    monitor = HealthMonitor()
    for _ in range(5):
        print(monitor.update_sensors())
        time.sleep(1)
