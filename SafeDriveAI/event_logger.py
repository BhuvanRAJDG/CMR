import csv
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_dir="logs", filename="events.csv"):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create CSV and write header if it's newly created
        file_exists = os.path.isfile(self.filepath)
        if not file_exists:
            with open(self.filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "EventType", "HeartRate", "SpO2", "FatigueScore", "GPS_Lat", "GPS_Lon"])

    def log_event(self, event_type,  heart_rate, spo2, fatigue_score, gps_lat=0.0, gps_lon=0.0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event_type, heart_rate, spo2, fatigue_score, gps_lat, gps_lon])
            
        print(f"Logged Event: {event_type} at {timestamp}")
