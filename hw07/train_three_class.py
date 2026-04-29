#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
胸部X光肺炎检测 - 三分类任务（进阶）
Normal vs Bacterial Pneumonia vs Viral Pneumonia
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'data_path': './chest_xray',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 40,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
    'seed': 42,
    'dropout_rate': 0.5,
    'num_classes': 3  # Normal, Bacterial, Viral
}

np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

# ==================== 数据准备 ====================
def load_and_prepare_data():
    """加载数据并分类为三类别"""
    
    train_dir = os.path.join(CONFIG['data_path'], 'train')
    test_dir = os.path.join(CONFIG['data_path'], 'test')
    
    def get_class_label(filename, class_name):
        """根据文件名确定三分类标签"""
        if class_name == 'NORMAL':
            return 0  # Normal
        else:  # PNEUMONIA
            filename_lower = filename.lower()
            if 'virus' in filename_lower:
                return 1  # Viral Pneumonia
            elif 'bacteria' in filename_lower:
                return 2  # Bacterial Pneumonia
            else:
                # 如果没有明确标识，默认为细菌性（训练集中大部分是细菌性）
                return 2
    
    # 获取训练集
    train_images = []
    train_labels = []
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_images.append(os.path.join(class_dir, img_name))
                train_labels.append(get_class_label(img_name, class_name))
    
    # 获取测试集
    test_images = []
    test_labels = []
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(test_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(class_dir, img_name))
                test_labels.append(get_class_label(img_name, class_name))
    
    # 统计三类别分布
    class_names = ['Normal', 'Viral Pneumonia', 'Bacterial Pneumonia']
    
    print("="*50)
    print("三分类数据集统计:")
    
    # 划分训练验证集前统计
    train_normal = train_labels.count(0)
    train_viral = train_labels.count(1)
    train_bacterial = train_labels.count(2)
    
    print(f"原始训练集: {len(train_images)} 张")
    print(f"  - Normal: {train_normal} 张")
    print(f"  - Viral Pneumonia: {train_viral} 张")
    print(f"  - Bacterial Pneumonia: {train_bacterial} 张")
    
    # 从训练集中按8:2划分训练集和验证集（分层抽样）
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['seed'],
        stratify=train_labels
    )
    
    print(f"\n训练集: {len(train_images)} 张")
    print(f"  - Normal: {train_labels.count(0)} 张")
    print(f"  - Viral: {train_labels.count(1)} 张")
    print(f"  - Bacterial: {train_labels.count(2)} 张")
    
    print(f"\n验证集: {len(val_images)} 张")
    print(f"  - Normal: {val_labels.count(0)} 张")
    print(f"  - Viral: {val_labels.count(1)} 张")
    print(f"  - Bacterial: {val_labels.count(2)} 张")
    
    print(f"\n测试集: {len(test_images)} 张")
    print(f"  - Normal: {test_labels.count(0)} 张")
    print(f"  - Viral: {test_labels.count(1)} 张")
    print(f"  - Bacterial: {test_labels.count(2)} 张")
    print("="*50)
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def load_image(path, label):
    """加载并预处理单张图像"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, CONFIG['img_size'])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def data_augmentation(image, label):
    """数据增强"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def create_datasets(train_data, val_data, test_data, use_augmentation=True):
    """创建TensorFlow数据集"""
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data
    
    # 转换标签为one-hot编码
    train_labels_onehot = tf.keras.utils.to_categorical(train_labels, CONFIG['num_classes'])
    val_labels_onehot = tf.keras.utils.to_categorical(val_labels, CONFIG['num_classes'])
    test_labels_onehot = tf.keras.utils.to_categorical(test_labels, CONFIG['num_classes'])
    
    # 训练集
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels_onehot))
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if use_augmentation:
        train_ds = train_ds.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(CONFIG['batch_size'])
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 验证集
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels_onehot))
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(CONFIG['batch_size'])
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 测试集
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels_onehot))
    test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(CONFIG['batch_size'])
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

# ==================== 模型构建 ====================
def build_three_class_model():
    """三分类CNN模型"""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 全连接层
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(CONFIG['dropout_rate']),
        layers.Dense(128, activation='relu'),
        layers.Dropout(CONFIG['dropout_rate']),
        layers.Dense(CONFIG['num_classes'], activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==================== 评估与可视化 ====================
def plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Viral', 'Bacterial']):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Three-Class Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_three_class.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_ds, test_labels):
    """评估三分类模型"""
    print("\n" + "="*50)
    print("三分类测试集评估")
    print("="*50)
    
    # 获取预测
    y_pred_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.array(test_labels)
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Viral', 'Bacterial'],
                                digits=4, zero_division=0))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred)
    
    return accuracy, y_true, y_pred

# ==================== 主训练流程 ====================
def main():
    # 1. 加载数据
    print("正在加载三分类数据...")
    train_data, val_data, test_data = load_and_prepare_data()
    
    # 2. 创建数据集
    print("\n正在创建数据集...")
    train_ds, val_ds, test_ds = create_datasets(train_data, val_data, test_data, use_augmentation=True)
    
    # 3. 构建模型
    print("\n正在构建三分类模型...")
    model = build_three_class_model()
    model.summary()
    
    # 4. 设置回调
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_model_three_class.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # 5. 训练
    print("\n开始训练三分类模型...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. 评估
    model = keras.models.load_model('best_model_three_class.h5')
    accuracy, y_true, y_pred = evaluate_model(model, test_ds, test_data[1])
    
    print(f"\n三分类最终准确率: {accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    model = main()
