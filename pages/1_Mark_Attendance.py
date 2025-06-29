import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from scipy.spatial.distance import cosine
import os
import json
from helpers.db_utils import load_students, log_attendance
from helpers.face_utils import get_face_embedding

st.set_page_config(page_title="Mark Attendance")

st.title("🟢 Mark Attendance - Face Recognition")

THRESHOLD = 0.55  # lower is more strict; tune this

# Load registered students
students_df = load_students()

def match_face(input_embedding, student_data):
    min_score = float('inf')
    matched_student = None

    for _, row in student_data.iterrows():
        known_emb = np.array(json.loads(row["Embedding"]))
        score = cosine(input_embedding, known_emb)

        if score < min_score:
            min_score = score
            matched_student = row

    if min_score < THRESHOLD:
        return matched_student, min_score
    else:
        return None, min_score

# Webcam Capture
def capture_face_image():
    cap = cv2.VideoCapture(0)
    st.info("📸 Press 's' to scan face, 'q' to cancel")
    captured_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Failed to access webcam.")
            break

        cv2.imshow("Press 's' to scan | 'q' to cancel", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            captured_img = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_img


# Start Attendance Process
if st.button("📸 Start Face Scan"):
    st.info("Opening webcam...")
    frame = capture_face_image()

    if frame is not None:
        # Save temp file
        tmp_path = "data/temp_scan.jpg"
        cv2.imwrite(tmp_path, frame)

        embedding = get_face_embedding(tmp_path)

        if embedding is None:
            st.error("⚠️ Face not detected. Please try again.")
        else:
            matched_student, score = match_face(embedding, students_df)

            if matched_student is not None:
                name = matched_student["Name"]
                matric = matched_student["Matric No"]
                st.success(f"✅ {name} identified (Matric: {matric})")
                st.write(f"Cosine Similarity Score: `{score:.4f}`")

                # Log attendance
                log_attendance(name, matric)
                st.success("🕒 Attendance marked successfully!")

                st.image(matched_student["Image Path"], caption="Matched Student", width=250)

            else:
                st.warning("❌ No match found. Face not recognized.")
    else:
        st.warning("No image captured.")
