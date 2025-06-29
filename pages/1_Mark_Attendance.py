import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
import json
from helpers.face_utils import get_face_embedding
from helpers.db_utils import load_students, log_attendance

THRESHOLD = 0.55

st.title("📸 Mark Attendance (Webcam)")

students_df = load_students()

# Video stream + capture logic
class FaceScan(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img
        return img

ctx = webrtc_streamer(
    key="attendance",
    video_transformer_factory=FaceScan,
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.video_transformer:
    if st.button("✅ Scan Face and Mark Attendance"):
        frame = ctx.video_transformer.latest_frame
        if frame is not None:
            # Save frame temporarily
            temp_path = "data/temp_face.jpg"
            Image.fromarray(frame).save(temp_path)

            emb = get_face_embedding(temp_path)
            if emb is None:
                st.warning("⚠️ Face not detected.")
            else:
                min_dist = float('inf')
                matched_student = None

                for _, row in students_df.iterrows():
                    known_emb = np.array(json.loads(row["Embedding"]))
                    dist = cosine(emb, known_emb)
                    if dist < min_dist:
                        min_dist = dist
                        matched_student = row

                if min_dist < THRESHOLD:
                    name = matched_student["Name"]
                    matric = matched_student["Matric No"]
                    st.success(f"✅ Matched: {name} ({matric}) | Score: {min_dist:.4f}")
                    st.image(matched_student["Image Path"], caption="Matched Face", width=300)
                    log_attendance(name, matric)
                else:
                    st.error(f"❌ No match found. Closest score: {min_dist:.4f}")
