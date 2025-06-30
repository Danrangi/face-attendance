import re
from typing import Tuple
from config import MATRIC_PATTERN

def validate_student_input(name: str, matric_no: str, department: str) -> Tuple[bool, str]:
    """Validate student registration input"""
    if not all([name, matric_no, department]):
        return False, "All fields are required"
    
    if len(name.strip()) < 2:
        return False, "Name is too short"
    
    if not re.match(MATRIC_PATTERN, matric_no):
        return False, "Invalid matric number format (e.g., U20/FNS/CSC/1111)"
    
    if len(department.strip()) < 2:
        return False, "Department name is too short"
    
    return True, ""