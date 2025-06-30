import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from PIL import Image
import uuid
import logging
from helpers.face_utils import get_face_embedding
from helpers.db_utils import register_student
from helpers.validation import validate_student_input
from config import FACES_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'registration_step' not in st.session_state:
    st.session_state.registration_step = 'form'
if 'form_data' not in st.session_state:
    st.session_state.form_data = None

class FaceCapture(VideoTransformerBase):
    def __init__(self):
        self.frame = None
        self.capture_requested = False

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

def registration_form():
    """Display and handle registration form"""
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        matric_no = st.text_input("Matric Number (e.g., U20/FNS/CSC/1111)")
        department = st.text_input("Department")
        submitted = st.form_submit_button("Register")

        if submitted:
            is_valid, message = validate_student_input(name, matric_no, department)
            if not is_valid:
                st.error(message)
                return False
            
            st.session_state.form_data = {
                "name": name,
                "matric_no": matric_no,
                "department": department
            }
            st.session_state.registration_step = 'capture'
            return True
    return False

def face_capture():
    """Handle face capture process"""
    st.write("### Face Capture")
    st.write(f"Registering: {st.session_state.form_data['name']}")
    st.write("Please ensure good lighting and face the camera directly.")

    ctx = webrtc_streamer(
        key="register",
        video_transformer_factory=FaceCapture,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if ctx.video_transformer:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("✅ Capture Face"):
                try:
                    frame = ctx.video_transformer.frame
                    if frame is None:
                        st.error("No frame captured. Please ensure your camera is working.")
                        return

                    # Get face embedding first
                    emb = get_face_embedding(frame)
                    if emb is None:
                        st.error("⚠️ No face detected. Please ensure your face is clearly visible.")
                        return

                    # Save image
                    filename = f"{uuid.uuid4().hex}.jpg"
                    filepath = FACES_DIR / filename
                    Image.fromarray(frame).save(filepath)

                    # Register student
                    success, message = register_student(
                        st.session_state.form_data["name"],
                        st.session_state.form_data["matric_no"],
                        st.session_state.form_data["department"],
                        str(filepath),
                        emb.tolist()
                    )

                    if success:
                        st.success(f"🎉 {message}")
                        st.image(frame, caption="Captured Face", width=300)
                        st.session_state.registration_step = 'form'
                        st.session_state.form_data = None
                        st.experimental_rerun()
                    else:
                        st.error(message)

                except Exception as e:
                    logger.error(f"Error during face capture: {str(e)}")
                    st.error("An unexpected error occurred. Please try again.")

        with col2:
            if st.button("⬅️ Back"):
                st.session_state.registration_step = 'form'
                st.session_state.form_data = None
                st.experimental_rerun()

def main():
    st.title("🎓 Student Registration")
    
    if st.session_state.registration_step == 'form':
        registration_form()
    elif st.session_state.registration_step == 'capture':
        face_capture()

if __name__ == "__main__":
    main()