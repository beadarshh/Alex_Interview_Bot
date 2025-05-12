# Alex – AI-Powered Interview Feedback Assistant

Alex is an open-source virtual interview assistant that provides real-time feedback on posture and voice quality. Designed for students, professionals, and anyone looking to improve their soft skills, Alex simulates a mock interview and delivers personalized, AI-generated feedback on your performance.

🧠 Why Alex?

Interview preparation can be stressful — Alex provides a private, interactive, and intelligent platform to practice your soft skills and communication without judgment. It’s like a smart mirror for your body language and speech clarity.

📸 Core Features

* 🧍 Posture Detection: Uses your webcam and YOLOv8 to detect slouching, leaning, and other posture indicators.
* 🎤 Voice Feedback: Analyzes recorded speech for tone, clarity, pace, and filler words.
* 🤖 AI-Powered Insights: Leverages Gemini API to combine video and audio cues and generate constructive feedback.
* 🔁 Performance Comparison: Tracks and compares your sessions to monitor improvement.
* 🗣️ Text-to-Speech: Feedback is read aloud for accessibility using pyttsx3.

🚀 Tech Stack Highlights

This project uses a combination of AI, computer vision, and speech processing tools:

* Python 3.10+
* OpenCV – webcam capture
* YOLOv8 – real-time posture detection
* Gemini API – multimodal AI feedback generation
* pyttsx3 – offline text-to-speech
* Streamlit – interactive front-end interface

🖥️ Installation

1. Clone the repository:
   git clone https://github.com/beadarshh/Alex_Interview_Bot

2. Navigate into the project folder:
   cd Alex_Interview_Bot

3. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   
5. Install dependencies:
   pip install -r requirements.txt
   
7. Run the app:
   puthon main.py

🧪 Sample Output
You’ll receive on-screen feedback like:
* "You're leaning too far forward. Try to maintain an upright posture."
* "You used 8 filler words. Try to pause briefly instead of using 'um'."
* "Tone and pitch are clear and confident. Well done!"

🤝 Contributing
We welcome contributions of all kinds! You can help us by:
* Improving UI/UX with Streamlit components
* Enhancing voice analysis (emotion, filler words)
* Adding support for more posture scenarios
* Writing documentation and tutorials

Check out CONTRIBUTING.md (coming soon) for guidelines.

📄 License
MIT License — free to use, modify, and distribute. See LICENSE for details.

🙌 Acknowledgments
* YOLOv8 by Ultralytics
* Google Gemini API
* Streamlit
* pyttsx3

📬 Contact

Have questions, suggestions, or feedback?
Reach out via Issues or email: aadarshpandey9@gmail.com or adarshpandey4114@gmail.com

