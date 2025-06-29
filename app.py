import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import uuid
from helpers.face_utils import get_face_embedding
from helpers.db_utils import register_student, load_students

# Directories
FACES_DIR = "data/faces"
os.makedirs(FACES_DIR, exist_ok=True)

# --- Webcam image capture ---
def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    st.info("📸 Press 's' to capture, 'q' to cancel")
    captured_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Cannot read from webcam.")
            break

        cv2.imshow("Press 's' to save | 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            captured_img = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_img

# --- Streamlit UI ---
st.set_page_config(page_title="Student Registration", layout="centered")
st.title("🎓 Student Registration - Face Attendance System")

with st.form("register_form"):
    name = st.text_input("Full Name")
    matric = st.text_input("Matric Number")
    dept = st.text_input("Department")
    submit = st.form_submit_button("Register Student")

if submit:
    if name and matric and dept:
        st.info("Opening webcam... Please allow access.")
        img = capture_image_from_webcam()

        if img is not None:
            file_name = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(FACES_DIR, file_name)
            cv2.imwrite(file_path, img)

            embedding = get_face_embedding(file_path)

            if embedding is not None:
                register_student(name, matric, dept, file_path, embedding.tolist())
                st.success(f"✅ {name} registered successfully!")
                st.image(Image.open(file_path), caption="Captured Face", width=300)
            else:
                st.error("Face not detected in image. Please try again.")
        else:
            st.warning("No image captured. Please try again.")
    else:
        st.warning("Please fill all the fields before submitting.")
