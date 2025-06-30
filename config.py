import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"

# Create directories if they don't exist
FACES_DIR.mkdir(parents=True, exist_ok=True)

# File paths
STUDENT_CSV = DATA_DIR / "students.csv"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

# Face recognition settings
FACE_MATCH_THRESHOLD = 0.55
FACE_DETECTION_CONFIDENCE = 0.9
MAX_RETRY_ATTEMPTS = 3

# Validation patterns
MATRIC_PATTERN = r'^U\d{2}/[A-Z]{3}/[A-Z]{3}/\d{4}$'