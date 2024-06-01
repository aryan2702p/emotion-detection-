import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the pre-trained models
try:
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    classifier = load_model('model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
except Exception as e:
    #st.write(f"Error loading models: {e}")
    st.session_state.run = False
    st.experimental_rerun()

# Function to detect and predict emotion
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Function to run the webcam and process frames
def run_webcam():
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image.")
            break
        frame = detect_emotion(frame)
        FRAME_WINDOW.image(frame, channels='BGR')

    cap.release()

# Streamlit app


st.title("Real-time Emotion Detection")
st.write("This application uses a pre-trained model to detect emotions from a live webcam feed.")
st.write("Toggle the checkbox below to start or stop the detection.")



# Initialize the checkbox state
if 'run' not in st.session_state:
    st.session_state.run = False

# Main loop to check for the 'Run' state
while True:
    run_checkbox = st.checkbox('Run', value=st.session_state.run)
    if run_checkbox:
        st.session_state.run = True
        run_webcam()
    else:
        st.session_state.run = False
        break

st.markdown("<hr>", unsafe_allow_html=True)
st.write("Made by Aryan Patel")
