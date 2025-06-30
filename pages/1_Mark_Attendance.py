import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from PIL import Image
import json
from scipy.spatial.distance import cosine
import logging
from datetime import datetime, timedelta
from helpers.face_utils import get_face_embedding
from helpers.db_utils import load_students, log_attendance
from config import FACE_MATCH_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'last_attendance_time' not in st.session_state:
    st.session_state.last_attendance_time = None

class FaceScan(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

def load_student_database():
    """Load and prepare student database"""
    try:
        df = load_students()
        if df.empty:
            st.warning("No students registered in the database.")
            return None
        return df
    except Exception as e:
        logger.error(f"Error loading student database: {str(e)}")
        st.error("Failed to load student database.")
        return None

def check_attendance_cooldown() -> bool:
    """Check if enough time has passed since last attendance"""
    if st.session_state.last_attendance_time:
        time_diff = datetime.now() - st.session_state.last_attendance_time
        if time_diff.total_seconds() < 30:  # 30 seconds cooldown
            st.warning("Please wait 30 seconds between attendance marks.")
            return False
    return True

def find_matching_student(embedding: np.ndarray, students_df: pd.DataFrame):
    """Find matching student from embedding"""
    try:
        min_dist = float('inf')
        matched_student = None

        for _, row in students_df.iterrows():
            known_emb = np.array(json.loads(row["Embedding"]))
            dist = cosine(embedding, known_emb)
            if dist < min_dist:
                min_dist = dist
                matched_student = row

        if min_dist < FACE_MATCH_THRESHOLD:
            return True, min_dist, matched_student
        return False, min_dist, None

    except Exception as e:
        logger.error(f"Error in face matching: {str(e)}")
        return False, float('inf'), None

def main():
    st.title("📸 Mark Attendance")
    st.write("Please ensure good lighting and face the camera directly.")

    # Load student database
    students_df = load_student_database()
    if students_df is None:
        return

    # Video stream
    ctx = webrtc_streamer(
        key="attendance",
        video_transformer_factory=FaceScan,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if ctx.video_transformer:
        if st.button("✅ Scan Face and Mark Attendance"):
            if not check_attendance_cooldown():
                return

            try:
                frame = ctx.video_transformer.frame
                if frame is None:
                    st.error("No frame captured. Please ensure your camera is working.")
                    return

                # Get face embedding
                emb = get_face_embedding(frame)
                if emb is None:
                    st.error("⚠️ No face detected. Please ensure your face is clearly visible.")
                    return

                # Find matching student
                match_found, distance, matched_student = find_matching_student(emb, students_df)

                if match_found:
                    name = matched_student["Name"]
                    matric = matched_student["Matric No"]
                    
                    # Log attendance
                    success, message = log_attendance(name, matric)
                    if success:
                        st.success(f"✅ Welcome {name}! Attendance marked successfully.")
                        st.image(matched_student["Image Path"], caption="Registered Face", width=300)
                        st.session_state.last_attendance_time = datetime.now()
                    else:
                        st.error(message)
                else:
                    st.error(f"❌ No match found. Closest match score: {distance:.4f}")

            except Exception as e:
                logger.error(f"Error during attendance marking: {str(e)}")
                st.error("An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()