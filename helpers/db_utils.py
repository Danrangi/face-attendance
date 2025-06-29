# helpers/db_utils.py

import pandas as pd
import os
import json
from datetime import datetime

# Paths
STUDENT_CSV = "data/students.csv"
ATTENDANCE_CSV = "data/attendance.csv"

# Load student records
def load_students():
    if os.path.exists(STUDENT_CSV):
        return pd.read_csv(STUDENT_CSV)
    else:
        return pd.DataFrame(columns=["Name", "Matric No", "Department", "Image Path", "Embedding"])

# Save student records (used internally)
def save_students(df):
    df.to_csv(STUDENT_CSV, index=False)

# Save a new student (embedding should be JSON-encoded)
def register_student(name, matric_no, department, image_path, embedding_list):
    df = load_students()
    emb_str = json.dumps(embedding_list)
    new_entry = {
        "Name": name,
        "Matric No": matric_no,
        "Department": department,
        "Image Path": image_path,
        "Embedding": emb_str
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    save_students(df)
    return True

# Log attendance
def log_attendance(name, matric_no, status="Check-In"):
    now = datetime.now()
    record = {
        "Name": name,
        "Matric No": matric_no,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": status
    }

    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(ATTENDANCE_CSV, index=False)

# Load attendance records
def load_attendance():
    if os.path.exists(ATTENDANCE_CSV):
        return pd.read_csv(ATTENDANCE_CSV)
    else:
        return pd.DataFrame(columns=["Name", "Matric No", "Date", "Time", "Status"])
