

# ===================== IMPORTS =====================
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import winsound
import streamlit as st

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

# ===================== GPU SAFE CONFIG =====================
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# ===================== LOAD MODELS =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

model = keras.models.load_model("my_model (1).h5")

# ===================== SESSION STATE =====================
if "closed_counter" not in st.session_state:
    st.session_state.closed_counter = 0

# ===================== UI =====================
st.title("😴 Drowsiness Detection System")

nav_choice = st.sidebar.radio(
    "Navigation", ("Home", "Sleep Detection")
)

# ===================== HOME PAGE =====================
if nav_choice == "Home":
    st.header("Preventing Sleep-Deprivation Road Accidents 🚗")
    
    st.image("ISHN0619_C3_pic.jpg", width=800)
    st.image("sleep.jfif", width=350)

    st.markdown(
        """
        ### 📌 How to Use
        1. Go to **Sleep Detection**
        2. Sit in good lighting
        3. Face the camera properly
        4. Close eyes for 2–3 seconds to test alarm
        """
    )

# ===================== SLEEP DETECTION =====================
elif nav_choice == "Sleep Detection":

    st.header("🎥 Live Drowsiness Detection")

    start = st.radio("Camera Control", ("Stop", "Start"), index=0)

    frame_window = st.empty()
    status_text = st.empty()

    if start == "Start":

        cap = cv2.VideoCapture(0)

        while start == "Start":
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera not accessible")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            eyes_open = 0

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                    eye_img = cv2.resize(eye_img, (86, 86))

                    pred = model.predict(
                        np.expand_dims(eye_img, axis=0),
                        verbose=0
                    )

                    confidence = np.max(pred)
                    label = np.argmax(pred)

                    if label == 1 and confidence > 0.7:
                        eyes_open += 1

                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex+ew, ey+eh),
                        (0, 255, 0),
                        2
                    )

            # ===================== DECISION =====================
            if eyes_open == 0:
                st.session_state.closed_counter += 1
                status_text.error("😴 Eyes Closed")

                if st.session_state.closed_counter > 15:
                    winsound.Beep(2500, 1500)
                    st.session_state.closed_counter = 0
            else:
                st.session_state.closed_counter = 0
                status_text.success("👀 Eyes Open")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

            time.sleep(0.03)

        cap.release()

