import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from PIL import Image
import os
import uuid
from helpers.face_utils import get_face_embedding
from helpers.db_utils import register_student

# Constants
FACES_DIR = "data/faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Global capture state
st.session_state['captured_face'] = None

# Face capture transformer
class FaceCapture(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

# Streamlit form
st.title("🎓 Student Registration (Webcam - Streamlit Compatible)")

with st.form("registration_form"):
    name = st.text_input("Full Name")
    matric_no = st.text_input("Matric Number")
    department = st.text_input("Department")
    submitted = st.form_submit_button("Register")

if submitted:
    if not all([name, matric_no, department]):
        st.warning("Fill in all fields first.")
    else:
        st.success("Now scan your face below and click ✅ Capture Face")

# Live stream
ctx = webrtc_streamer(
    key="register",
    video_transformer_factory=FaceCapture,
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.video_transformer and submitted:
    if st.button("✅ Capture Face"):
        frame = ctx.video_transformer.frame
        if frame is not None:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(FACES_DIR, filename)
            Image.fromarray(frame).save(filepath)

            emb = get_face_embedding(filepath)
            if emb is not None:
                register_student(name, matric_no, department, filepath, emb.tolist())
                st.image(Image.open(filepath), caption="Captured Face", width=300)
                st.success("🎉 Student registered and face saved!")
            else:
                st.error("⚠️ Face not detected. Try again.")

