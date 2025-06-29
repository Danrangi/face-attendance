import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import uuid
import json
from helpers.face_utils import get_face_embedding

# Constants
DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
CSV_PATH = os.path.join(DATA_DIR, "students.csv")

# Ensure directories exist
os.makedirs(FACES_DIR, exist_ok=True)

# Load or initialize student database
if os.path.exists(CSV_PATH):
    student_df = pd.read_csv(CSV_PATH)
else:
    student_df = pd.DataFrame(columns=["Name", "Matric No", "Department", "Image Path", "Embedding"])

# --- Function to save student record ---
def save_student_data(name, matric_no, department, image_path):
    global student_df

    embedding = get_face_embedding(image_path)
    if embedding is None:
        st.error("Face not detected in image. Please try again.")
        return

    emb_str = json.dumps(embedding.tolist())  # Convert NumPy to JSON string

    new_entry = {
        "Name": name,
        "Matric No": matric_no,
        "Department": department,
        "Image Path": image_path,
        "Embedding": emb_str
    }

    student_df = pd.concat([student_df, pd.DataFrame([new_entry])], ignore_index=True)
    student_df.to_csv(CSV_PATH, index=False)

# --- Webcam capture using OpenCV ---
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


# --- Streamlit App UI ---
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

            save_student_data(name, matric, dept, file_path)
            st.success(f"✅ {name} registered successfully!")
            st.image(Image.open(file_path), caption="Captured Face", width=300)
        else:
            st.warning("No image captured. Please try again.")
    else:
        st.warning("Please fill all the fields before submitting.")
