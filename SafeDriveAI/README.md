# SafeDrive AI (AI Driver Monitoring & Health Emergency Detection)

SafeDrive AI is a hackathon prototype designed to enhance driver safety by detecting drowsiness and monitoring vital health signs. The system uses computer vision for real-time face tracking and simulates health sensor data (ECG, Heart Rate, SpO2) to demonstrate emergency response capabilities.

## Features
- **Drowsiness Detection**: Uses Eye Aspect Ratio (EAR) via MediaPipe FaceMesh to detect if a driver's eyes are closed for too long.
- **Health Monitoring Simulation**: Real-time simulation of heart rate, SpO2, and ECG status.
- **Emergency Alerts**: Visual and console alerts for drowsiness and critical health conditions (Tachycardia, Bradycardia, Low SpO2).
- **GPS Simulation**: Automatically simulates sending a GPS location to emergency services when a health crisis is detected.
- **Integrated Dashboard**: Clear, professional overlay on the webcam feed showing all live data.

## Installation

1. **Clone the project** or navigate to the project folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Prototype

To start the system, run:
```bash
python main.py
```
- Press **'q'** to exit the application.

## Project Structure
- `main.py`: The central integration point and UI renderer.
- `vision.py`: Handles webcam capture and MediaPipe facial landmark extraction.
- `drowsiness.py`: Contains the logic for EAR calculation and drowsiness state detection.
- `health_monitor.py`: Simulates health sensor data and handles emergency detection logic.

## Integration with ESP32 Hardware
Later, this software can be integrated with physical sensors (like MAX30102 for SpO2/HR and AD8232 for ECG) connected to an ESP32.

### How it works:
1. **Hardware Side**: The ESP32 reads raw data from the sensors and sends it over a USB/Serial cable in a JSON or comma-separated format (e.g., `HR:75,SPO2:98,ECG:N`).
2. **Software Side**: In `health_monitor.py`, instead of `random.uniform()`, you would use the `pyserial` library:
   ```python
   import serial
   ser = serial.Serial('COM3', 115200) # Adjust port as needed
   
   def update_sensors(self):
       line = ser.readline().decode('utf-8').strip()
       # Parse 'line' and update self.heart_rate, self.spo2, etc.
   ```
3. **ESP32 Code**: A simple Arduino sketch would read sensors and `Serial.println()` the data to the computer.

## Prototype Mode Note
Since no hardware is connected, `health_monitor.py` generates random data. It occasionally simulates "Emergency" events to demonstrate the alert system (GPS location logging and red screen warnings).
