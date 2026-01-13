#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Architecture cho HAR
Định nghĩa kiến trúc CNN nhỏ gọn cho ESP32
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def create_har_model(input_shape=(561, 1), num_classes=6, model_type='cnn_simple'):
    """
    Tạo model HAR nhỏ gọn cho ESP32
    
    Args:
        input_shape: Shape của input (features, channels)
        num_classes: Số classes (6 cho UCI HAR)
        model_type: Loại model ('cnn_simple', 'cnn_deep', 'lstm')
        
    Returns:
        Keras model
    """
    print("=" * 60)
    print(f"Tạo model: {model_type}")
    print("=" * 60)
    
    if model_type == 'cnn_simple':
        model = create_cnn_simple(input_shape, num_classes)
    elif model_type == 'cnn_deep':
        model = create_cnn_deep(input_shape, num_classes)
    elif model_type == 'lstm':
        model = create_lstm(input_shape, num_classes)
    else:
        raise ValueError(f"Model type không hợp lệ: {model_type}")
    
    # In thông tin model
    print("\n📊 Thông tin model:")
    model.summary()
    
    # Tính kích thước model
    total_params = model.count_params()
    print(f"\n📏 Tổng số parameters: {total_params:,}")
    print(f"📏 Ước tính kích thước: {total_params * 4 / 1024:.2f} KB (float32)")
    print(f"📏 Sau quantization (int8): ~{total_params / 1024:.2f} KB")
    
    if total_params * 4 / 1024 > 100:
        print("⚠️  Cảnh báo: Model có thể quá lớn cho ESP32!")
        print("   Khuyến nghị: < 100KB trước quantization")
    else:
        print("✅ Kích thước model phù hợp cho ESP32")
    
    return model


def create_cnn_simple(input_shape, num_classes):
    """
    CNN đơn giản, nhỏ gọn (~50KB)
    Phù hợp cho ESP32 với RAM hạn chế
    """
    model = models.Sequential([
        # Conv block 1
        layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Conv block 2
        layers.Conv1D(32, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_Simple')
    
    return model


def create_cnn_deep(input_shape, num_classes):
    """
    CNN sâu hơn, độ chính xác cao hơn (~100KB)
    """
    model = models.Sequential([
        # Conv block 1
        layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Conv block 2
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Conv block 3
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_Deep')
    
    return model


def create_lstm(input_shape, num_classes):
    """
    LSTM model cho temporal patterns
    Kích thước trung bình (~80KB)
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='LSTM')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model với optimizer và loss function
    
    Args:
        model: Keras model
        learning_rate: Learning rate cho Adam optimizer
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\n✅ Model đã compile với learning_rate={learning_rate}")
    return model


if __name__ == '__main__':
    # Test model creation
    print("\n🧪 Testing Model Creation...\n")
    
    # Test các loại model
    for model_type in ['cnn_simple', 'cnn_deep', 'lstm']:
        print(f"\n{'='*60}")
        model = create_har_model(input_shape=(561, 1), num_classes=6, model_type=model_type)
        model = compile_model(model)
        print()
    
    print("\n✅ Tất cả models hoạt động tốt!")

