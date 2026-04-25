import streamlit as st
import numpy as np
import cv2
from PIL import Image
import face_recognition

from src.face_utils import detect_faces, encode_faces, compare_faces

st.title("人脸识别系统")

# 是否启用识别
enable_recognition = st.checkbox("启用人脸识别（需人脸库）")

uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

# 示例人脸库（你可以换成自己的）
known_encodings = []
known_names = []

if enable_recognition:
    try:
        img = face_recognition.load_image_file("known.jpg")
        encoding = face_recognition.face_encodings(img)[0]
        known_encodings.append(encoding)
        known_names.append("Known_Person")
    except:
        st.warning("未检测到已知人脸库图片 known.jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # 检测人脸
    face_locations = detect_faces(image_np)
    face_encodings = encode_faces(image_np)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 画框
        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)

        name = "Face"

        if enable_recognition and len(known_encodings) > 0:
            name = compare_faces(known_encodings, known_names, face_encoding)

        # 标注名字
        cv2.putText(
            image_np,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.image(image_np, caption="识别结果", use_column_width=True)

    st.write(f"检测到 {len(face_locations)} 张人脸")
