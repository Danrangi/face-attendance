import pandas as pd
import json
from datetime import datetime
import logging
from typing import Tuple, Optional
from config import STUDENT_CSV, ATTENDANCE_CSV

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_students() -> pd.DataFrame:
    """Load student records"""
    try:
        if STUDENT_CSV.exists():
            return pd.read_csv(STUDENT_CSV)
        return pd.DataFrame(columns=["Name", "Matric No", "Department", "Image Path", "Embedding"])
    except Exception as e:
        logger.error(f"Error loading students: {str(e)}")
        return pd.DataFrame(columns=["Name", "Matric No", "Department", "Image Path", "Embedding"])

def register_student(name: str, matric_no: str, department: str, 
                    image_path: str, embedding_list: list) -> Tuple[bool, str]:
    """Register a new student"""
    try:
        df = load_students()
        
        # Check for duplicate matric number
        if matric_no in df["Matric No"].values:
            return False, "Matric number already exists"

        new_entry = {
            "Name": name,
            "Matric No": matric_no,
            "Department": department,
            "Image Path": str(image_path),
            "Embedding": json.dumps(embedding_list)
        }
        
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(STUDENT_CSV, index=False)
        return True, "Student registered successfully"

    except Exception as e:
        logger.error(f"Error in student registration: {str(e)}")
        return False, f"Registration failed: {str(e)}"

def log_attendance(name: str, matric_no: str, status: str = "Check-In") -> Tuple[bool, str]:
    """Log student attendance"""
    try:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        # Load or create attendance DataFrame
        if ATTENDANCE_CSV.exists():
            df = pd.read_csv(ATTENDANCE_CSV)
        else:
            df = pd.DataFrame(columns=["Name", "Matric No", "Date", "Time", "Status"])

        # Check for duplicate attendance
        existing = df[
            (df["Matric No"] == matric_no) & 
            (df["Date"] == today) &
            (df["Status"] == status)
        ]
        
        if not existing.empty:
            return False, "Attendance already marked for today"

        record = {
            "Name": name,
            "Matric No": matric_no,
            "Date": today,
            "Time": now.strftime("%H:%M:%S"),
            "Status": status
        }
        
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)
        return True, "Attendance marked successfully"

    except Exception as e:
        logger.error(f"Error in attendance logging: {str(e)}")
        return False, f"Failed to log attendance: {str(e)}"

def load_attendance() -> pd.DataFrame:
    """Load attendance records"""
    try:
        if ATTENDANCE_CSV.exists():
            return pd.read_csv(ATTENDANCE_CSV)
        return pd.DataFrame(columns=["Name", "Matric No", "Date", "Time", "Status"])
    except Exception as e:
        logger.error(f"Error loading attendance: {str(e)}")
        return pd.DataFrame(columns=["Name", "Matric No", "Date", "Time", "Status"])