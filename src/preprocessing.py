#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Pipeline cho HAR
Chuẩn hóa và reshape dữ liệu cho model
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocess_data(X_train, X_test, y_train, y_test, 
                   scaler_type='standard', validation_split=0.2,
                   save_scaler=True, scaler_path='models/scaler.pkl'):
    """
    Chuẩn hóa và chia dữ liệu
    
    Args:
        X_train, X_test: Dữ liệu features
        y_train, y_test: Labels
        scaler_type: 'standard' hoặc 'minmax'
        validation_split: Tỷ lệ validation (0.2 = 20%)
        save_scaler: Lưu scaler để dùng sau
        scaler_path: Đường dẫn lưu scaler
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print("=" * 60)
    print("Preprocessing dữ liệu...")
    print("=" * 60)
    
    # Chọn scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
        print("📊 Sử dụng StandardScaler (mean=0, std=1)")
    else:
        scaler = MinMaxScaler()
        print("📊 Sử dụng MinMaxScaler (range 0-1)")
    
    # Fit scaler trên training data
    print("\n🔧 Đang fit scaler trên training data...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ X_train scaled: min={X_train_scaled.min():.4f}, max={X_train_scaled.max():.4f}")
    print(f"✅ X_test scaled: min={X_test_scaled.min():.4f}, max={X_test_scaled.max():.4f}")
    
    # Chia validation set từ training data
    print(f"\n📂 Chia validation set ({validation_split*100:.0f}%)...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, 
        test_size=validation_split, 
        random_state=42,
        stratify=y_train  # Đảm bảo tỷ lệ classes đều
    )
    
    print(f"✅ Training set: {X_train_final.shape[0]} samples")
    print(f"✅ Validation set: {X_val.shape[0]} samples")
    print(f"✅ Test set: {X_test_scaled.shape[0]} samples")
    
    # Kiểm tra phân bố classes
    print("\n📊 Phân bố classes:")
    for i in range(int(y_train.max()) + 1):
        train_count = np.sum(y_train_final == i)
        val_count = np.sum(y_val == i)
        test_count = np.sum(y_test == i)
        print(f"  Class {i}: Train={train_count:4d}, Val={val_count:4d}, Test={test_count:4d}")
    
    # Lưu scaler
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\n💾 Đã lưu scaler tại: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing hoàn tất!")
    print("=" * 60)
    
    return X_train_final, X_val, X_test_scaled, y_train_final, y_val, y_test, scaler


def reshape_for_cnn(X_train, X_val, X_test, window_size=128, n_channels=9):
    """
    Reshape dữ liệu cho CNN input
    UCI HAR có 561 features, ta sẽ reshape thành (samples, timesteps, features)
    
    Args:
        X_train, X_val, X_test: Dữ liệu đã scaled
        window_size: Số timesteps (mặc định 128)
        n_channels: Số channels (mặc định 9 cho IMU)
        
    Returns:
        X_train, X_val, X_test đã reshape
    """
    print("\n🔄 Reshape dữ liệu cho CNN...")
    
    # UCI HAR có 561 features, ta giữ nguyên shape (samples, 561, 1)
    # Hoặc có thể reshape thành (samples, 128, 9) nếu cần
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"✅ X_train shape: {X_train_reshaped.shape}")
    print(f"✅ X_val shape: {X_val_reshaped.shape}")
    print(f"✅ X_test shape: {X_test_reshaped.shape}")
    
    return X_train_reshaped, X_val_reshaped, X_test_reshaped


def load_scaler(scaler_path='models/scaler.pkl'):
    """
    Load scaler đã lưu
    
    Args:
        scaler_path: Đường dẫn file scaler
        
    Returns:
        scaler object
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Đã load scaler từ: {scaler_path}")
    return scaler


if __name__ == '__main__':
    # Test preprocessing
    print("\n🧪 Testing Preprocessing Pipeline...\n")
    
    # Import data loader
    from data_loader import load_uci_har_data
    
    # Load data
    X_train, X_test, y_train, y_test, _, _, _ = load_uci_har_data()
    
    # Preprocess
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2
    )
    
    # Reshape cho CNN
    X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    print("\n✅ Preprocessing pipeline hoạt động tốt!")

