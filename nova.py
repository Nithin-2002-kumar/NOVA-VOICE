import os
import sys
import time
import json
import threading
import subprocess
import webbrowser
import logging
import tkinter as tk
from tkinter import ttk, Canvas, messagebox
from PIL import Image, ImageTk
import speech_recognition as sr
import pyttsx3
import pyautogui
import psutil
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import wikipedia
import requests
import spacy
import pygame

# Setup logging
logging.basicConfig(filename="nova.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
CONFIG_FILE = "nova_config.json"
DEFAULT_CONFIG = {
    "hotword": "nova",
    "speech_rate": 180,
    "voice": 1,  # Female voice for NOVA
    "weather_api_key": "YOUR_OPENWEATHERMAP_API_KEY",
    "news_api_key": "YOUR_NEWSAPI_KEY",
    "email_user": "your_email@gmail.com",
    "email_pass": "your_app_password"
}

class NOVA:
    def __init__(self):
        # Load or initialize configuration
        self.load_config()

        self.root = tk.Tk()
        self.root.title("NOVA Voice Interface")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.9)  # Slightly more opaque
        self.root.configure(bg='#1a1a1a')  # Darker, modern background
        self.running = True

        # Initialize components
        self.init_components()
        self.init_ui()
        self.start_threads()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG
            self.save_config()

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def init_components(self):
        # Speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[self.config["voice"]].id)
        self.engine.setProperty('rate', self.config["speech_rate"])

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        # Computer vision
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # NLP and ML
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC(kernel='linear', probability=True)
        self.train_classifier()

        # Memory
        self.memory = {"history": [], "tasks": [], "reminders": [], "learned_commands": {}}
        self.load_memory()

        # Initialize Pygame for UI effects
        pygame.init()

    def load_memory(self):
        if os.path.exists("nova_memory.json"):
            with open("nova_memory.json", 'r') as f:
                self.memory = json.load(f)

    def save_memory(self):
        with open("nova_memory.json", 'w') as f:
            json.dump(self.memory, f, indent=4)

    def init_ui(self):
        # Main canvas
        self.canvas = Canvas(self.root, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Status display
        self.status_var = tk.StringVar(value="Awaiting your command")
        self.status_label = tk.Label(self.canvas, textvariable=self.status_var, fg='#00ffcc', bg='#1a1a1a',
                                     font=('Montserrat', 16, 'bold'))
        self.canvas.create_window(50, 50, window=self.status_label, anchor='nw')

        # Command display
        self.command_text = tk.Text(self.canvas, height=12, width=80, bg='#1a1a1a', fg='#00ffcc',
                                    font=('Montserrat', 14), bd=0, insertbackground='#00ffcc')
        self.canvas.create_window(self.root.winfo_screenwidth() // 2, 400, window=self.command_text, anchor='center')

        # Dynamic HUD elements
        self.hud_elements = []
        for i in range(6):
            arc = self.canvas.create_oval(20 + i * 40, 20 + i * 40, 80 + i * 40, 80 + i * 40,
                                         outline='#00ffcc', width=2)
            self.hud_elements.append(arc)
        self.animate_hud()

        # Exit button
        self.exit_btn = ttk.Button(self.canvas, text="Power Off", command=self.shutdown,
                                   style='NOVA.TButton')
        self.canvas.create_window(self.root.winfo_screenwidth() - 50, 50, window=self.exit_btn, anchor='ne')

        # Configure ttk style
        style = ttk.Style()
        style.configure('NOVA.TButton', font=('Montserrat', 12), foreground='#00ffcc', background='#1a1a1a')

    def animate_hud(self):
        if self.running:
            for arc in self.hud_elements:
                coords = self.canvas.coords(arc)
                self.canvas.move(arc, np.random.randint(-3, 4), np.random.randint(-3, 4))
                if coords[2] > self.root.winfo_screenwidth() or coords[3] > self.root.winfo_screenheight():
                    self.canvas.coords(arc, 20, 20, 80, 80)
            self.root.after(40, self.animate_hud)

    def train_classifier(self):
        commands = [
            "open notepad", "open browser", "search web", "set reminder", "tell time",
            "take screenshot", "check news", "check weather", "scan room"
        ]
        labels = ["open", "open", "search", "reminder", "time", "screenshot", "news", "weather", "scan"]
        X = self.vectorizer.fit_transform(commands)
        self.classifier.fit(X, labels)

    def speak(self, text):
        self.command_text.insert(tk.END, f"NOVA: {text}\n")
        self.command_text.see(tk.END)
        self.memory["history"].append({"role": "NOVA", "text": text, "time": datetime.now().isoformat()})
        self.save_memory()
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.status_var.set("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio).lower()
                if self.config["hotword"] in text:
                    self.speak("Ready to assist.")
                    command = text.replace(self.config["hotword"], "").strip()
                    self.command_text.insert(tk.END, f"User: {command}\n")
                    self.memory["history"].append({"role": "user", "text": command, "time": datetime.now().isoformat()})
                    return command
            except sr.UnknownValueError:
                self.speak("Could you repeat that, please?")
            except sr.RequestError:
                self.speak("Speech service is down. Please try again.")
            return None

    def process_command(self, command):
        X = self.vectorizer.transform([command])
        intent = self.classifier.predict(X)[0]
        confidence = max(self.classifier.predict_proba(X)[0])

        if confidence < 0.65:
            self.handle_unknown_command(command)
            return

        if intent == "open":
            self.open_app(command)
        elif intent == "search":
            self.search_web(command)
        elif intent == "reminder":
            self.set_reminder(command)
        elif intent == "time":
            self.tell_time()
        elif intent == "screenshot":
            self.take_screenshot()
        elif intent == "news":
            self.check_news(command)
        elif intent == "weather":
            self.check_weather(command)
        elif intent == "scan":
            self.scan_environment()
        elif "gesture" in command:
            self.gesture_control()
        else:
            self.speak("Processing request...")
            response = self.get_wikipedia_response(command)
            self.speak(response)

    def open_app(self, command):
        apps = {
            "notepad": "notepad.exe",
            "browser": "https://www.google.com",
            "calculator": "calc.exe",
            "terminal": "cmd.exe"
        }
        for app, path in apps.items():
            if app in command:
                if app == "browser":
                    webbrowser.open(path)
                else:
                    subprocess.Popen(path)
                self.speak(f"Launching {app}.")
                return
        self.speak("App not recognized.")

    def search_web(self, command):
        query = command.replace("search", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        self.speak(f"Searching for {query}.")

    def set_reminder(self, command):
        doc = self.nlp(command)
        time_str = next((ent.text for ent in doc.ents if ent.label_ == "TIME"), "in 1 hour")
        if "in" in time_str and "hour" in time_str:
            hours = int(time_str.split()[1])
            reminder_time = datetime.now() + timedelta(hours=hours)
        else:
            reminder_time = datetime.now() + timedelta(hours=1)
        message = command.replace(time_str, "").replace("set reminder", "").strip()
        self.memory["reminders"].append({"text": message, "time": reminder_time.isoformat()})
        self.save_memory()
        threading.Thread(target=self.check_reminder, args=(message, reminder_time), daemon=True).start()
        self.speak(f"Reminder set for {time_str}: {message}.")

    def check_reminder(self, message, reminder_time):
        while self.running and datetime.now() < reminder_time:
            time.sleep(10)
        if self.running:
            self.speak(f"Reminder: {message}")

    def tell_time(self):
        time_str = datetime.now().strftime("%I:%M %p")
        self.speak(f"It's {time_str}.")

    def take_screenshot(self):
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        pyautogui.screenshot(filename)
        self.speak(f"Screenshot saved as {filename}.")

    def check_news(self, command):
        if not self.config["news_api_key"]:
            self.speak("News API key not configured.")
            return
        doc = self.nlp(command)
        topic = next((ent.text for ent in doc.ents if ent.label_ in ["GPE", "NORP", "ORG"]), "general")
        url = f"https://newsapi.org/v2/top-headlines?q={topic}&apiKey={self.config['news_api_key']}"
        try:
            response = requests.get(url).json()
            articles = response["articles"][:3]
            self.speak(f"Top news on {topic}:")
            for i, article in enumerate(articles, 1):
                self.speak(f"{i}. {article['title']} from {article['source']['name']}.")
        except:
            self.speak("Unable to fetch news.")

    def check_weather(self, command):
        if not self.config["weather_api_key"]:
            self.speak("Weather API key not configured.")
            return
        doc = self.nlp(command)
        location = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), "current location")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.config['weather_api_key']}&units=metric"
        try:
            response = requests.get(url).json()
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            self.speak(f"In {location}, it's {desc} with a temperature of {temp}°C.")
        except:
            self.speak("Unable to fetch weather data.")

    def gesture_control(self):
        self.speak("Activating gesture control.")
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    thumb_tip = hand_landmarks.landmark[4].y
                    index_tip = hand_landmarks.landmark[8].y
                    if index_tip < hand_landmarks.landmark[6].y:
                        pyautogui.scroll(10)
                    elif index_tip > hand_landmarks.landmark[6].y:
                        pyautogui.scroll(-10)
                    elif abs(thumb_tip - index_tip) < 0.05:
                        pyautogui.click()
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.speak("Gesture control off.")

    def scan_environment(self):
        self.speak("Scanning environment.")
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.speak(f"Face detected at coordinates {x}, {y}.")
            cv2.imshow("Environment Scan", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def get_wikipedia_response(self, command):
        try:
            summary = wikipedia.summary(command, sentences=2)
            return summary
        except:
            return "I couldn't find information on that topic."

    def handle_unknown_command(self, command):
        self.speak("Unrecognized command. Shall I learn it?")
        response = self.listen()
        if response and "yes" in response:
            self.speak("Please describe the action.")
            explanation = self.listen()
            if explanation:
                self.memory["learned_commands"][command] = explanation
                self.save_memory()
                self.speak(f"Learned command '{command}'.")

    def system_monitor(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            if cpu > 85 or mem > 85:
                self.speak(f"Alert: CPU at {cpu}%, Memory at {mem}%.")
            time.sleep(60)

    def start_threads(self):
        threading.Thread(target=self.system_monitor, daemon=True).start()
        threading.Thread(target=self.listen_loop, daemon=True).start()

    def listen_loop(self):
        self.speak("NOVA online. How can I assist you?")
        while self.running:
            command = self.listen()
            if command:
                self.process_command(command)
            time.sleep(1)

    def shutdown(self):
        self.speak("Powering down. Goodbye.")
        self.running = False
        pygame.quit()
        self.save_memory()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    # Ensure 'Montserrat' font is installed
    nova = NOVA()
    nova.root.mainloop()
