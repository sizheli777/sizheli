#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
胸部X光肺炎检测 - 二分类任务
Normal vs Pneumonia
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, 
                             recall_score, f1_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'data_path': './chest_xray',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-4,
    'validation_split': 0.2,  # 从train中划分20%作为验证集
    'seed': 42,
    'dropout_rate': 0.5,
    'num_classes': 1  # 二分类使用sigmoid
}

# 设置随机种子
np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

# ==================== 1. 数据准备 ====================
def load_and_prepare_data():
    """加载数据并按8:2重新划分训练集和验证集"""
    
    train_dir = os.path.join(CONFIG['data_path'], 'train')
    test_dir = os.path.join(CONFIG['data_path'], 'test')
    
    # 获取所有训练图像路径和标签
    train_images = []
    train_labels = []
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_images.append(os.path.join(class_dir, img_name))
                train_labels.append(0 if class_name == 'NORMAL' else 1)
    
    # 获取测试集图像路径和标签
    test_images = []
    test_labels = []
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(test_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(class_dir, img_name))
                test_labels.append(0 if class_name == 'NORMAL' else 1)
    
    # 从训练集中按8:2划分训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['seed'],
        stratify=train_labels  # 保持类别比例
    )
    
    # 统计信息
    print("="*50)
    print("数据集统计:")
    print(f"训练集: {len(train_images)} 张")
    print(f"  - Normal: {train_labels.count(0)} 张")
    print(f"  - Pneumonia: {train_labels.count(1)} 张")
    print(f"验证集: {len(val_images)} 张")
    print(f"  - Normal: {val_labels.count(0)} 张")
    print(f"  - Pneumonia: {val_labels.count(1)} 张")
    print(f"测试集: {len(test_images)} 张")
    print(f"  - Normal: {test_labels.count(0)} 张")
    print(f"  - Pneumonia: {test_labels.count(1)} 张")
    print("="*50)
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def load_image(path, label):
    """加载并预处理单张图像"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, CONFIG['img_size'])
    img = tf.cast(img, tf.float32) / 255.0  # 归一化到[0,1]
    return img, label

def data_augmentation(image, label):
    """数据增强（仅用于训练集）"""
    # 随机水平翻转
    image = tf.image.random_flip_left_right(image)
    # 随机旋转（-10到10度）
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # 随机亮度调整
    image = tf.image.random_brightness(image, max_delta=0.1)
    # 随机对比度调整
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def create_datasets(train_data, val_data, test_data, use_augmentation=True):
    """创建TensorFlow数据集"""
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data
    
    # 训练集
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if use_augmentation:
        train_ds = train_ds.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(CONFIG['batch_size'])
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 验证集（不增强）
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(CONFIG['batch_size'])
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 测试集（不增强）
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(CONFIG['batch_size'])
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

# ==================== 2. 模型构建 ====================
def build_cnn_model():
    """方案A：自定义CNN模型"""
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
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model

def build_transfer_learning_model():
    """方案B：使用迁移学习（MobileNetV2）"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 冻结底层
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(CONFIG['dropout_rate']),
        layers.Dense(128, activation='relu'),
        layers.Dropout(CONFIG['dropout_rate']),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model

# ==================== 3. 训练与评估 ====================
def plot_training_history(history, save_path='figures/'):
    """绘制训练曲线"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Pneumonia'], save_path='figures/'):
    """绘制混淆矩阵"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_ds):
    """评估模型并计算指标"""
    print("\n" + "="*50)
    print("测试集评估")
    print("="*50)
    
    # 获取预测结果
    y_true = []
    y_pred_prob = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_prob.extend(preds.flatten())
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'],
                                digits=4, zero_division=0))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred
    }

# ==================== 主训练流程 ====================
def main():
    # 1. 加载数据
    print("正在加载数据...")
    train_data, val_data, test_data = load_and_prepare_data()
    
    # 2. 创建数据集
    print("\n正在创建数据集...")
    train_ds, val_ds, test_ds = create_datasets(train_data, val_data, test_data, use_augmentation=True)
    
    # 3. 构建模型
    print("\n正在构建模型...")
    # 选择模型类型：'cnn' 或 'transfer_learning'
    model_type = 'cnn'  # 可改为 'transfer_learning' 使用迁移学习
    
    if model_type == 'cnn':
        model = build_cnn_model()
        print("使用自定义CNN模型")
    else:
        model = build_transfer_learning_model()
        print("使用迁移学习模型 (MobileNetV2)")
    
    model.summary()
    
    # 4. 设置回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # 5. 训练模型
    print("\n" + "="*50)
    print("开始训练模型")
    print("="*50)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(history)
    
    # 7. 评估模型
    print("\n评估模型...")
    # 加载最佳模型
    model = keras.models.load_model('best_model.h5')
    results = evaluate_model(model, test_ds)
    
    # 8. 保存最终指标
    print("\n" + "="*50)
    print("最终测试集指标摘要:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("="*50)
    
    return model, results

if __name__ == "__main__":
    model, results = main()
