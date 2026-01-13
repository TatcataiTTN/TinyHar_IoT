#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script cho HAR Model
Train model và lưu kết quả
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime

# Import các module khác
from data_loader import load_uci_har_data
from preprocessing import preprocess_data, reshape_for_cnn
from model import create_har_model, compile_model

def train_model(model_type='cnn_simple', epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train HAR model
    
    Args:
        model_type: Loại model ('cnn_simple', 'cnn_deep', 'lstm')
        epochs: Số epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        model, history
    """
    print("=" * 60)
    print("🚀 BẮT ĐẦU TRAINING HAR MODEL")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    
    # 1. Load dữ liệu
    print("\n📂 BƯỚC 1: Load dữ liệu")
    X_train, X_test, y_train, y_test, _, _, activity_labels = load_uci_har_data()
    
    # 2. Preprocessing
    print("\n🔧 BƯỚC 2: Preprocessing")
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2,
        save_scaler=True
    )
    
    # 3. Reshape cho CNN
    print("\n🔄 BƯỚC 3: Reshape dữ liệu")
    X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # 4. Tạo model
    print("\n🏗️  BƯỚC 4: Tạo model")
    model = create_har_model(
        input_shape=(X_train_r.shape[1], X_train_r.shape[2]),
        num_classes=6,
        model_type=model_type
    )
    model = compile_model(model, learning_rate=learning_rate)
    
    # 5. Callbacks
    print("\n⚙️  BƯỚC 5: Setup callbacks")
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/har_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 6. Training
    print("\n🎯 BƯỚC 6: Bắt đầu training...")
    print("=" * 60)
    
    history = model.fit(
        X_train_r, y_train_p,
        validation_data=(X_val_r, y_val_p),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Đánh giá trên test set
    print("\n📊 BƯỚC 7: Đánh giá trên test set")
    test_loss, test_acc = model.evaluate(X_test_r, y_test_p, verbose=0)
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 8. Lưu model
    print("\n💾 BƯỚC 8: Lưu model")
    model_path = f'models/har_model_{model_type}.h5'
    model.save(model_path)
    print(f"✅ Đã lưu model tại: {model_path}")
    
    # 9. Vẽ training history
    print("\n📈 BƯỚC 9: Vẽ training history")
    plot_training_history(history, model_type)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING HOÀN TẤT!")
    print("=" * 60)
    
    return model, history


def plot_training_history(history, model_type):
    """
    Vẽ biểu đồ training history
    
    Args:
        history: Training history
        model_type: Tên model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'Model Accuracy - {model_type}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'Model Loss - {model_type}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Lưu figure
    os.makedirs('models', exist_ok=True)
    plot_path = f'models/training_history_{model_type}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ tại: {plot_path}")
    
    plt.close()


if __name__ == '__main__':
    # Cấu hình training
    MODEL_TYPE = 'cnn_simple'  # 'cnn_simple', 'cnn_deep', 'lstm'
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Train model
    model, history = train_model(
        model_type=MODEL_TYPE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    print("\n✅ Script hoàn tất!")

