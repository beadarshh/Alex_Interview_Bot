import cv2
import os
import pyttsx3
import numpy as np
from ultralytics import YOLO
from config.secrets import GEMINI_API_KEY
import PIL.Image
import google.generativeai as genai
import time
import threading
import sounddevice as sd
import wavio
import re
from difflib import SequenceMatcher

# === Setup === #
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
engine = pyttsx3.init()
yolo_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
print("🎥 Starting Alex - Interview Feedback Assistant")

# === Global States === #
is_recording = False
recording_start_time = 0
audio_filename = "speech.wav"
posture_insight = ""
frame_snapshot = None
recording_duration = 10  # seconds
show_start = True
show_stop = False
transcribed_text = "Your spoken content will appear here."
session_count = 0
previous_feedback = ""

# === Audio Recorder === #
def record_audio(duration=10, filename="speech.wav"):
    samplerate = 44100
    print("🎙️ Recording audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    wavio.write(filename, audio, samplerate, sampwidth=2)
    print("🎙️ Audio recording saved.")

# === Button Definitions === #
start_button = {'pos': (30, 50), 'size': (100, 40), 'label': 'Start'}
stop_button = {'pos': (150, 50), 'size': (100, 40), 'label': 'Stop'}

def draw_button(frame, button, color=(0, 255, 0)):
    x, y = button['pos']
    w, h = button['size']
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.putText(frame, button['label'], (x + 10, y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def is_inside_button(x, y, button):
    bx, by = button['pos']
    bw, bh = button['size']
    return bx <= x <= bx + bw and by <= y <= by + bh

def clean_text(text):
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"_+", "", text)
    return text.strip()

def compare_feedback(old, new):
    ratio = SequenceMatcher(None, old, new).ratio()
    if ratio < 0.6:
        return "✅ Noticeable improvement since last session."
    elif ratio < 0.85:
        return "➕ Slight improvement."
    else:
        return "⚠️ Similar feedback as before. Focus on improvements."

def analyze_feedback(image_path, audio_path):
    try:
        img = PIL.Image.open(image_path)
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()

        prompt = """
You are a mock interview coach. This is a still image and audio recording of a person during an interview.

1. Analyze the person's **sitting posture** and suggest improvements.
2. Analyze the **speaking quality** (tone, clarity, pace, and confidence).
3. Suggest better **word choices or vocabulary** if any parts sound too casual or unclear.

Be clear, concise, and supportive.
        """

        contents = [
            prompt,
            img,
            {
                "mime_type": "audio/wav",
                "data": audio_bytes
            }
        ]

        print("🧠 Asking Gemini for full feedback...")
        response = model.generate_content(contents)
        reply = clean_text(response.text.strip())

        print("🧠 Gemini Says:\n", reply)

        # Save to file
        with open("feedback_response.txt", "w", encoding="utf-8") as f:
            f.write(reply)

        global posture_insight, transcribed_text, session_count, previous_feedback
        comparison = compare_feedback(previous_feedback, reply)
        previous_feedback = reply
        posture_insight = f"{reply}\n\n{comparison}"
        transcribed_text = "Simulated transcription of what you said."  # placeholder
        session_count += 1

        def speak_and_clear(text):
            engine.say(text)
            engine.runAndWait()
            global posture_insight
            posture_insight = ""

        threading.Thread(target=speak_and_clear, args=(reply,)).start()

    except Exception as e:
        print("❌ Error:", e)
        posture_insight = "Error analyzing feedback."

def click_event(event, x, y, flags, param):
    global is_recording, frame_snapshot, recording_start_time
    global show_start, show_stop

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_inside_button(x, y, start_button) and show_start and not is_recording:
            print("✅ Started recording session.")
            is_recording = True
            recording_start_time = time.time()
            threading.Thread(target=record_audio, args=(recording_duration, audio_filename)).start()
            show_start = False
            show_stop = True

        elif is_inside_button(x, y, stop_button) and show_stop and is_recording:
            print("🛑 Stopped. Capturing frame for posture analysis.")
            is_recording = False
            show_stop = False
            show_start = True
            if person_box:
                x1, y1, x2, y2 = person_box
                cropped = frame[y1:y2, x1:x2]
                posture_img_path = "captured.jpg"
                cv2.imwrite(posture_img_path, cropped)
                threading.Thread(target=analyze_feedback, args=(posture_img_path, audio_filename)).start()

cv2.namedWindow("Alex - Interview Bot")
cv2.setMouseCallback("Alex - Interview Bot", click_event)

# === Main Loop === #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]
    person_box = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            person_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if label in ["hand", "cell phone"]:  # YOLOv8 may not have "hand", this is illustrative
            hand_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            cv2.putText(frame, "Hand", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

        if label in ["eye", "face", "head"]:
            eye_detected = True
            cv2.circle(frame, (x1 + 5, y1 + 5), 5, (100, 255, 255), -1)
            cv2.putText(frame, "Eye/Head", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2)

    if show_start:
        draw_button(frame, start_button, (0, 255, 0))
    if show_stop:
        draw_button(frame, stop_button, (0, 0, 255))

    # === DASHBOARD === #
    dashboard_width = 400
    dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype=np.uint8)
    cv2.rectangle(dashboard, (0, 0), (dashboard_width, frame.shape[0]), (30, 30, 30), -1)

    cv2.putText(dashboard, "INTERVIEW DASHBOARD", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Sessions: {session_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(dashboard, "You Said:", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, line in enumerate(transcribed_text.splitlines()):
        y = 140 + i * 20
        cv2.putText(dashboard, line[:45], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(dashboard, "Gemini Feedback:", (10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for j, line in enumerate(posture_insight.splitlines()):
        y2 = y + 70 + j * 20
        if y2 < frame.shape[0] - 10:
            cv2.putText(dashboard, line[:45], (10, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Combine camera feed with dashboard
    combined = np.hstack((frame, dashboard))
    cv2.imshow("Alex - Interview Bot", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
