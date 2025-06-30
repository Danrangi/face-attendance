import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import logging
from typing import Optional
from config import FACE_DETECTION_CONFIDENCE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        logger.info(f"Face processor initialized using device: {self.device}")

    def get_face_embedding(self, image) -> Optional[np.ndarray]:
        """Get face embedding from image"""
        try:
            # Convert different image types to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, str):
                image = Image.open(image)
            
            if not isinstance(image, Image.Image):
                raise ValueError("Unsupported image type")

            # Detect face
            face = self.mtcnn(image)
            if face is None:
                logger.warning("No face detected in image")
                return None

            # Generate embedding
            with torch.no_grad():
                embedding = self.resnet(face.unsqueeze(0).to(self.device))
                embedding = embedding.squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.error(f"Error in face embedding generation: {str(e)}")
            return None

# Initialize global face processor
face_processor = FaceProcessor()

def get_face_embedding(image) -> Optional[np.ndarray]:
    """Global function to get face embedding"""
    return face_processor.get_face_embedding(image)