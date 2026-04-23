"""
main.py - SafeDrive AI
Central integration point for all modules including the 8 new features:
head pose, yawn detection, fatigue scoring, voice alerts, event logging, 
accident detection, and auto-reporting.
"""
import cv2
import time
import random

# Core Vision and Health
from vision import FaceMeshDetector, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from drowsiness import DrowsinessDetector
from health_monitor import HealthMonitor

# New Hackathon Features
from attention import AttentionDetector
from yawn import YawnDetector
from fatigue import FatigueScoring
from voice_alert import VoiceAlertSystem
from event_logger import EventLogger
from accident import AccidentDetector
from report_generator import ReportGenerator

def main():
    print("SafeDrive AI: Initialising with Advanced Features...")
    cap = cv2.VideoCapture(0)
    # 720p is good for the big dashboard
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Instantiate modules
    detector        = FaceMeshDetector()
    drowsiness_chk  = DrowsinessDetector(ear_threshold=0.22, consecutive_frames=15)
    health_monitor  = HealthMonitor()
    
    attention_chk   = AttentionDetector(fps=30, distracted_time_threshold=3.0)
    yawn_chk        = YawnDetector(mar_threshold=0.6, fps=30, yawn_time_threshold=1.0)
    fatigue_sys     = FatigueScoring()
    voice           = VoiceAlertSystem()
    logger          = EventLogger()
    accident_chk    = AccidentDetector(fps=30, missing_face_threshold=5.0)
    reporter        = ReportGenerator()

    prev_time = time.time()
    
    # State flags
    accident_reported = False
    
    print("System Active – press 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed – exiting.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── 1. Update Health Sensors ────────────────────────────────
        hdata = health_monitor.update_sensors()
        
        # Simulated GPS for this iteration
        gps_lat = 37.7749 + random.uniform(-0.01, 0.01)
        gps_lon = -122.4194 + random.uniform(-0.01, 0.01)

        # ── 2. Vision Processing ────────────────────────────────────
        landmarks = detector.find_face_landmarks(frame)
        face_detected = landmarks is not None
        
        # State variables
        is_drowsy      = False
        avg_ear        = 0.0
        is_yawning     = False
        mar            = 0.0
        head_direction = "FORWARD"
        is_distracted  = False
        
        if face_detected:
            # Drowsiness (EAR)
            left_pts  = detector.get_eye_coords(landmarks, LEFT_EYE_INDICES, w, h)
            right_pts = detector.get_eye_coords(landmarks, RIGHT_EYE_INDICES, w, h)
            left_ear  = drowsiness_chk.calculate_ear(left_pts)
            right_ear = drowsiness_chk.calculate_ear(right_pts)
            is_drowsy, avg_ear = drowsiness_chk.check_drowsiness(left_ear, right_ear)
            
            # Yawning (MAR)
            mar = yawn_chk.calculate_mar(landmarks, w, h)
            is_yawning, yawn_count = yawn_chk.check_yawn(mar)
            
            # Head Pose Attention
            head_direction = attention_chk.get_head_pose(landmarks, w, h)
            is_distracted = attention_chk.check_attention(head_direction)
            
            # Draw facial dots for feedback
            for (x, y) in left_pts: cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_pts: cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        # ── 3. Fatigue Scoring & Alerts ──────────────────────────────
        f_score, f_level = fatigue_sys.update_score(is_drowsy, is_yawning, is_distracted)
        
        # Voice and Logging Logic
        if is_drowsy or f_level == "CRITICAL FATIGUE":
            voice.play_alert("drowsy")
            if is_drowsy and drowsiness_chk.frame_counter == drowsiness_chk.consecutive_frames:
                logger.log_event("DROWSY", hdata['heart_rate'], hdata['spo2'], f_score, gps_lat, gps_lon)
                
        if is_distracted:
            voice.play_alert("distracted")
            if attention_chk.frames_distracted == attention_chk.max_distracted_frames:
                logger.log_event("DISTRACTED", hdata['heart_rate'], hdata['spo2'], f_score, gps_lat, gps_lon)
                
        if hdata['emergency']:
            voice.play_alert("medical")
            if random.random() < 0.05: # occasional log
                logger.log_event("MEDICAL", hdata['heart_rate'], hdata['spo2'], f_score, gps_lat, gps_lon)
                
        # ── 4. Accident Detection ──────────────────────────────────
        is_accident = accident_chk.check_accident(face_detected, hdata['emergency'])
        if is_accident and not accident_reported:
            voice.play_alert("accident")
            logger.log_event("ACCIDENT", hdata['heart_rate'], hdata['spo2'], f_score, gps_lat, gps_lon)
            report_path = reporter.generate_report("ACCIDENT DETECTED", hdata, f_score, gps_lat, gps_lon)
            print(f"!!! EMERGENCY REPORT GENERATED: {report_path} !!!")
            accident_reported = True
        elif not is_accident:
            accident_reported = False

        # ── 5. UI Rendering ────────────────────────────────────────
        F = cv2.FONT_HERSHEY_DUPLEX
        WHITE, RED, GREEN, YELLOW = (255,255,255), (0,0,255), (0,220,0), (0,255,255)
        ORANGE, GREY = (0,165,255), (150,150,150)
        
        # Background panels
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 280), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w-350, 10), (w-10, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # --- LEFT PANEL: Driver State ---
        cv2.putText(frame, "DRIVER STATE", (20, 40), F, 0.7, ORANGE, 2)
        cv2.line(frame, (20, 50), (330, 50), WHITE, 1)
        
        cv2.putText(frame, f"Face: {'Detected' if face_detected else 'Missing'}", (20, 80), F, 0.6, GREEN if face_detected else RED, 1)
        cv2.putText(frame, f"Pose: {head_direction}", (20, 110), F, 0.6, YELLOW if is_distracted else WHITE, 1)
        cv2.putText(frame, f"EAR : {avg_ear:.3f} | MAR: {mar:.2f}", (20, 140), F, 0.6, WHITE, 1)
        
        # Fatigue Bar
        cv2.putText(frame, "FATIGUE SCORE:", (20, 180), F, 0.6, WHITE, 1)
        bar_w = 250
        cv2.rectangle(frame, (20, 195), (20 + bar_w, 215), GREY, 1)
        fill_w = int((f_score / 100.0) * bar_w)
        score_color = GREEN if f_score < 30 else (YELLOW if f_score < 70 else RED)
        if fill_w > 0:
            cv2.rectangle(frame, (21, 196), (20 + fill_w - 1, 214), score_color, -1)
        cv2.putText(frame, f"{f_score}/100", (20 + bar_w + 10, 210), F, 0.6, score_color, 1)
        
        cv2.putText(frame, f"Level: {f_level}", (20, 250), F, 0.65, score_color, 2)

        # --- RIGHT PANEL: Health & Safety ---
        cv2.putText(frame, "HEALTH VITALS", (w-330, 40), F, 0.7, ORANGE, 2)
        cv2.line(frame, (w-330, 50), (w-20, 50), WHITE, 1)
        
        hr_col = RED if hdata['emergency'] and (hdata['heart_rate']>120 or hdata['heart_rate']<50) else WHITE
        sp_col = RED if hdata['emergency'] and hdata['spo2']<90 else WHITE
        
        cv2.putText(frame, f"Heart Rate:  {hdata['heart_rate']} bpm", (w-330, 80), F, 0.6, hr_col, 1)
        cv2.putText(frame, f"SpO2 Level:  {hdata['spo2']} %", (w-330, 110), F, 0.6, sp_col, 1)
        cv2.putText(frame, f"ECG Status:  {hdata['ecg_status']}", (w-330, 140), F, 0.6, WHITE, 1)
        
        # Overall Status
        cv2.putText(frame, "SYSTEM STATUS:", (w-330, 190), F, 0.6, ORANGE, 1)
        if is_accident:
            sys_stat, sys_col = "ACCIDENT DETECTED", RED
        elif hdata['emergency']:
            sys_stat, sys_col = "MEDICAL EMERGENCY", RED
        elif f_level == "CRITICAL FATIGUE" or is_drowsy:
            sys_stat, sys_col = "DRIVER DROWSY", RED
        elif is_distracted:
            sys_stat, sys_col = "DRIVER DISTRACTED", YELLOW
        else:
            sys_stat, sys_col = "SAFE", GREEN
            
        cv2.putText(frame, sys_stat, (w-330, 230), F, 0.7, sys_col, 2)

        # ── 6. BIG ALERTS (Center Screen) ──────────────────────────
        center_x = w // 2
        center_y = h - 80
        if is_accident:
            cv2.rectangle(frame, (center_x-300, center_y-40), (center_x+300, center_y+20), RED, -1)
            cv2.putText(frame, "!!! CRASH DETECTED - CONTACTING SOS !!!", (center_x-280, center_y), F, 0.8, WHITE, 2)
        elif sys_col == RED:
            # Flashing effect
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(frame, (center_x-250, center_y-40), (center_x+250, center_y+20), RED, -1)
                cv2.putText(frame, f"WARNING: {sys_stat}", (center_x-200, center_y), F, 0.8, WHITE, 2)

        # --- Status Bar ---
        cv2.rectangle(frame, (0, h-30), (w, h), (20, 20, 20), -1)
        fps = 1.0 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()
        cv2.putText(frame, "SafeDrive AI Extended | Logging: Active | Voice: Active", (10, h-10), F, 0.45, GREY, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, h-10), F, 0.45, GREY, 1)

        cv2.imshow("SafeDrive AI - Advanced Dashboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("SafeDrive AI Shutdown.")

if __name__ == "__main__":
    main()
