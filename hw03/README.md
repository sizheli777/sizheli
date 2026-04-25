# 人脸识别系统（hw03）

## 项目简介
本项目基于 face_recognition 实现人脸检测与识别，并使用 Streamlit 构建 Web 界面。

实现功能：
- 上传图片
- 人脸检测（框选）
- 人脸特征提取（128维）
- 可选人脸识别

---

## 项目结构
hw03/
├── src/                # 人脸处理模块
│   └── face_utils.py
├── app.py              # Streamlit界面
├── requirements.txt    # 依赖
└── README.md           # 项目说明
## 运行方法

```bash
pip install -r requirements.txt
streamlit run app.py
