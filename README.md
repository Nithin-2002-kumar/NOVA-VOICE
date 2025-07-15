# NOVA Voice Assistant 🧠🎤💻

**NOVA** is an AI-powered offline voice assistant designed with modern UI, intelligent speech recognition, gesture control, and task automation. Built using Python and integrating computer vision, NLP, and speech processing, NOVA is your personal productivity companion.

---

## 🌟 Features

### ✅ Voice Interaction
- Hotword activation (`"nova"`)
- Natural speech recognition with `speech_recognition` + Google Speech API
- Voice responses via `pyttsx3` (offline TTS engine)

### ✅ Visual Interface
- Modern fullscreen **Tkinter GUI**
- Dynamic animated HUD using `Canvas`
- System alerts, logs, and command feedback in a rich-text box

### ✅ AI Capabilities
- **Natural Language Processing (NLP)** using `spaCy`
- **Intent classification** with `TfidfVectorizer` + `SVM`
- Wikipedia integration for knowledge queries
- Memory-based learning for unknown commands

### ✅ Smart Automation
- App launching (Notepad, Calculator, Browser, etc.)
- Web search
- Weather & news updates via APIs
- Time announcement
- Screenshot capturing
- Task reminders (time-based)
- Real-time system monitoring (CPU/Memory)

### ✅ Computer Vision + Gestures
- Hand gesture recognition using `MediaPipe`
  - Scroll up/down
  - Mouse click
- Facial detection with OpenCV (environment scanning)

---

## 🧰 Dependencies

Install the required packages via pip:

```bash
pip install -r requirements.txt
