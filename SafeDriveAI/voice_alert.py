import pyttsx3
import threading
import time

class VoiceAlertSystem:
    def __init__(self):
        # Initialize engine
        self.engine = pyttsx3.init()
        # Set a slightly faster rate for urgency
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', rate + 20)
        
        self.is_speaking = False
        # Cooldown dictionary: {alert_type: last_time_spoken}
        self.cooldowns = {
            "drowsy": 0,
            "distracted": 0,
            "medical": 0,
            "accident": 0
        }
        # Cooldown time in seconds to prevent spamming the same message
        self.cooldown_duration = 10 

    def _speak_thread(self, text):
        self.is_speaking = True
        try:
            # Using independent instance for thread safety
            engine = pyttsx3.init()
            engine.setProperty('rate', engine.getProperty('rate') + 20)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False

    def play_alert(self, alert_type):
        """
        Plays a voice alert in a background thread if it's not currently speaking
        and the cooldown period for this alert type has passed.
        """
        current_time = time.time()
        
        if self.is_speaking:
            return

        if alert_type in self.cooldowns:
            if current_time - self.cooldowns[alert_type] < self.cooldown_duration:
                return # Still in cooldown

        messages = {
            "drowsy": "Driver fatigue detected. Please stay alert.",
            "distracted": "Please keep your eyes on the road.",
            "medical": "Medical emergency detected. Pull over safely.",
            "accident": "Accident detected. Contacting emergency services."
        }
        
        if alert_type in messages:
            self.cooldowns[alert_type] = current_time
            thread = threading.Thread(target=self._speak_thread, args=(messages[alert_type],))
            thread.daemon = True
            thread.start()
