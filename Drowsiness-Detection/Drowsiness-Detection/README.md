😴 Drowsiness Detection System (Streamlit + Deep Learning)

This project is a real-time drowsiness detection system that uses a webcam to monitor eye activity and alerts the user when eyes remain closed for a prolonged time.
It is designed to help prevent road accidents caused by driver fatigue.

📌 Features

🎥 Real-time webcam monitoring

👁️ Face and eye detection using OpenCV

🧠 Deep learning model for eye state classification

🔊 Alarm sound when eyes remain closed

🌐 Interactive web interface using Streamlit

💻 Works on CPU (GPU optional)

🛠️ Technologies Used

Python

Streamlit – Web UI

OpenCV – Face & eye detection

TensorFlow / Keras – Deep learning model

NumPy & Pandas

Winsound – Alarm sound (Windows)

📂 Project Structure
Drowsiness-Detection/
│
├── app.py                     # Main Streamlit application
├── my_model (1).h5            # Trained deep learning model
├── ISHN0619_C3_pic.jpg         # Home page image
├── sleep.jfif                 # Home page image
├── README.md                  # Project documentation
├── requirements.txt           # Required Python packages
└── venv/                      # Virtual environment (optional)

⚙️ Installation Steps
1️⃣ Clone or Download the Project
git clone <repository-link>
cd Drowsiness-Detection

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Required Packages
pip install -r requirements.txt


If requirements.txt is not available:

pip install streamlit opencv-python tensorflow numpy pandas

▶️ How to Run the Application
streamlit run app.py


Open your browser and go to:

http://localhost:8501

📖 How It Works

The webcam captures live video frames

OpenCV detects the face and eyes

Eye images are passed to a trained CNN model

The model predicts whether eyes are open or closed

If eyes remain closed for several frames:

An alarm sound is triggered

Status is displayed on the Streamlit UI in real time

🚦 How to Use

Open the application

Go to Sleep Detection from the sidebar

Sit in front of the camera with proper lighting

Close your eyes for 2–3 seconds to test the alarm

Keep eyes open to stop the alert

🧪 Testing Tips

Use good lighting

Face the camera directly

Avoid covering eyes with hands

Adjust webcam position if detection is inaccurate

⚠️ Notes

CUDA/GPU warnings can be safely ignored if no GPU is installed

Alarm sound works on Windows only (winsound)

Haar Cascade may occasionally misdetect eyes under poor lighting

🎓 Use Case

Final Year Engineering Project

Academic Demonstration

Driver Safety Systems

AI-based Monitoring Applications

🚀 Future Enhancements

Mobile app version

Blink-rate analysis

Head pose detection

Email/SMS alert system

Cloud-based logging