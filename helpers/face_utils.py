# helpers/face_utils.py

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2

# Load models (only once)
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)

    if face is None:
        return None
    
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0)).squeeze().numpy()
        return embedding
