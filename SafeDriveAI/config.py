# ═══════════════════════════════════════
# VitalDrive AI — Configuration
# ═══════════════════════════════════════

# Camera
# ── Source: "WEBCAM" = local USB/built-in webcam | "ESP32" = IP stream ──
CAMERA_SOURCE  = "WEBCAM"        # Forced webcam — set to "ESP32" for IP stream
ESP32_CAM_IP   = "192.168.1.100" # only used when CAMERA_SOURCE = "ESP32"

# Webcam settings (only used when CAMERA_SOURCE = "WEBCAM")
WEBCAM_INDEX   = 0     # 0 = default system camera; 1, 2 … for extra cameras
WEBCAM_WIDTH   = 640   # requested capture width  (camera may override)
WEBCAM_HEIGHT  = 480   # requested capture height
WEBCAM_FPS     = 30    # requested frame-rate
WEBCAM_BACKEND = "DSHOW" # DSHOW = Windows DirectShow (fastest on Windows laptops)

# Serial
SERIAL_PORT = "AUTO"  # auto detect COM port, or enter COMx
BAUD_RATE = 115200
SIMULATION_MODE = True  # True allows demo mode with fake sensor values if hardware missing

# ─── SafeDriveAI Integration ───────────────────────────────────────────────
# Fatigue scoring weights (from SafeDriveAI)
FATIGUE_EYE_WEIGHT        = 2.0   # points per drowsy frame
FATIGUE_YAWN_WEIGHT       = 1.5   # points per yawn frame
FATIGUE_DISTRACT_WEIGHT   = 1.0   # points per distracted frame
FATIGUE_RECOVERY_RATE     = 0.5   # points recovered per safe frame
FATIGUE_SAFE_THRESHOLD    = 30    # below = SAFE
FATIGUE_DROWSY_THRESHOLD  = 70    # below = DROWSY, above = CRITICAL

# Attention / head-pose thresholds (from SafeDriveAI AttentionDetector)
ATTENTION_YAW_LEFT_RATIO  = 0.65  # nose ratio > this = looking left
ATTENTION_YAW_RIGHT_RATIO = 0.35  # nose ratio < this = looking right
ATTENTION_PITCH_DOWN      = 0.65  # pitch ratio > this = looking down
ATTENTION_DISTRACT_SECS   = 3.0   # seconds looking away = distracted

# EAR smoothing (moving-average window from SafeDriveAI)
EAR_SMOOTH_FRAMES = 5

# Emergency thresholds (from sleep/comp.py)
EMERGENCY_HR_HIGH  = 120   # bpm above = possible cardiac
EMERGENCY_HR_LOW   = 40    # bpm below = critical low HR
EMERGENCY_SPO2_LOW = 90    # % below = low oxygen

# Vision thresholds
EAR_THRESHOLD = 0.25
DROWSY_SECONDS = 2.0
HEAD_ANGLE_THRESHOLD = 30

# Health thresholds
HR_MIN = 40
HR_MAX = 120
SPO2_MIN = 90
SPO2_CRITICAL = 85
ECG_IRREGULARITY_THRESHOLD = 0.7

# Alert timings
CANCEL_WINDOW_SECONDS = 10
ESCALATION_DELAY = 5

# ─── Twilio SMS Config (from sleep/comp.py) ───────────────────────────────
# Fill in your real Twilio credentials here, or set via the sidebar in the UI
TWILIO_SID    = "your_sid"          # Twilio Account SID
TWILIO_TOKEN  = "your_token"        # Twilio Auth Token
TWILIO_FROM   = "+1xxxxxxxxxx"      # Your Twilio phone number
EMERGENCY_CONTACT = "+91xxxxxxxxxx" # Family / emergency contact
HOSPITAL_CONTACT  = "+91xxxxxxxxxx" # Hospital / dispatch contact

# ═══════════════════════════════════════
# UPGRADED FEATURES
# ═══════════════════════════════════════

# Voice
VOICE_ENABLED = True

# PERCLOS (Percentage of Eye Closure)
PERCLOS_WINDOW = 60        # seconds to track PERCLOS
PERCLOS_THRESHOLD = 0.15   # 15% of time eyes closed = drowsy

# Blink rate
BLINK_WINDOW = 60          # seconds to track blink rate
BLINK_LOW = 8              # below = hyperfocused / micro-sleep risk
BLINK_HIGH = 30            # above = extreme fatigue

# Yawn detection
MAR_THRESHOLD = 0.6        # mouth aspect ratio threshold
YAWN_SECONDS = 2.0         # how long mouth open = yawn

# Distraction detection
DISTRACTION_ANGLE = 30     # degrees left/right for distraction
DISTRACTION_SECONDS = 2.0  # seconds looking away = distracted

# HRV (Heart Rate Variability)
HRV_HIGH_THRESHOLD = 15    # std dev > 15 = irregular heartbeat
HRV_LOW_THRESHOLD = 2      # std dev < 2 = stress/shock

# SpO2 Trend
SPO2_DROP_THRESHOLD = 3    # % drop in 30s = trending down

# Risk scoring
PRE_EMERGENCY_SCORE = 7    # risk score for warning SMS
