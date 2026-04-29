# HW07 - 胸部X光肺炎检测 (Chest X-Ray Pneumonia Detection)

> **课程作业**：医学影像智能处理 - 肺炎二分类模型
> 
> 基于 Kaggle 公开数据集 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)，完成 Normal vs Pneumonia 二分类实验。

---

## 📁 目录结构
hw07/
├── README.md # 本文件
├── requirements.txt # Python 依赖包列表
├── train.py # 二分类训练脚本（必做）
├── train_three_class.py # 三分类训练脚本（进阶选做）
├── report.md # 实验报告
├── figures/ # 图表输出目录（运行后自动生成）
│ ├── training_history.png # 训练/验证曲线
│ ├── confusion_matrix.png # 二分类混淆矩阵
│ └── confusion_matrix_three_class.png # 三分类混淆矩阵
└── best_model.h5 # 训练保存的最佳模型（运行后生成）

---

## 🚀 快速开始

### 1. 环境要求

- **Python 版本**：3.8 或以上
- **TensorFlow 版本**：2.13.0（推荐使用 GPU 版本以加速训练）
- **运行平台**：本地 / Kaggle Notebook / Google Colab 均可

### 2. 安装依赖

```bash
cd hw07
pip install -r requirements.txt
# Kaggle Notebook（数据集已自动挂载）
import os
data_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/'

# Google Colab
from google.colab import drive
drive.mount('/content/drive')
# 然后上传数据集或从 Google Drive 读取
# 安装 kaggle CLI
pip install kaggle

# 下载数据集
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# 解压
unzip chest-xray-pneumonia.zip -d chest_xray
