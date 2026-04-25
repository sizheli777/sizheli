import face_recognition
import numpy as np

def detect_faces(image):
    """
    检测人脸位置
    """
    face_locations = face_recognition.face_locations(image)
    return face_locations


def encode_faces(image):
    """
    提取人脸特征（128维）
    """
    encodings = face_recognition.face_encodings(image)
    return encodings


def compare_faces(known_encodings, known_names, face_encoding):
    """
    与已知人脸库对比
    """
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    
    if True in matches:
        matched_idx = matches.index(True)
        return known_names[matched_idx]
    else:
        return "Unknown"
