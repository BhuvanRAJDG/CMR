import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, report_dir="reports"):
        self.report_dir = report_dir
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            
    def generate_report(self, event_reason, health_data, fatigue_score, gps_lat, gps_lon):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        filename = f"emergency_{timestamp}.txt"
        filepath = os.path.join(self.report_dir, filename)
        
        report_content = f"""=========================================
SafeDrive AI - Emergency Medical Report
=========================================
Timestamp:       {readable_time}
Primary Trigger: {event_reason}

--- Driver Vital Signs ---
Heart Rate:      {health_data['heart_rate']} bpm
SpO2 Level:      {health_data['spo2']} %
ECG Status:      {health_data['ecg_status']}

--- Driver State ---
Fatigue Score:   {fatigue_score} / 100

--- Location Data ---
GPS Latitude:    {gps_lat:.6f}
GPS Longitude:   {gps_lon:.6f}

Status: SYSTEM DISPATCHED REPORT TO EMERGENCY SERVICES
=========================================
"""
        with open(filepath, 'w') as f:
            f.write(report_content)
            
        return filepath
