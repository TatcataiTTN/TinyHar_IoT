#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader cho UCI HAR Dataset
Tải và parse dữ liệu từ UCI HAR Dataset
"""

import numpy as np
import os
import sys

def load_uci_har_data(dataset_path='datasets/UCI HAR Dataset'):
    """
    Tải UCI HAR Dataset
    
    Args:
        dataset_path: Đường dẫn đến thư mục dataset
        
    Returns:
        X_train, X_test, y_train, y_test, subject_train, subject_test
    """
    print("=" * 60)
    print("Đang tải UCI HAR Dataset...")
    print("=" * 60)
    
    # Kiểm tra dataset có tồn tại không
    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Không tìm thấy dataset tại {dataset_path}")
        print("Hãy chạy: python scripts/download_dataset.py")
        sys.exit(1)
    
    # Đường dẫn các file
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    # Load training data
    print("\n📂 Đang tải training data...")
    X_train = np.loadtxt(os.path.join(train_path, 'X_train.txt'))
    y_train = np.loadtxt(os.path.join(train_path, 'y_train.txt'))
    subject_train = np.loadtxt(os.path.join(train_path, 'subject_train.txt'))
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ y_train: {y_train.shape}")
    print(f"✅ subject_train: {subject_train.shape}")
    
    # Load test data
    print("\n📂 Đang tải test data...")
    X_test = np.loadtxt(os.path.join(test_path, 'X_test.txt'))
    y_test = np.loadtxt(os.path.join(test_path, 'y_test.txt'))
    subject_test = np.loadtxt(os.path.join(test_path, 'subject_test.txt'))
    
    print(f"✅ X_test: {X_test.shape}")
    print(f"✅ y_test: {y_test.shape}")
    print(f"✅ subject_test: {subject_test.shape}")
    
    # Load activity labels
    activity_labels = {}
    with open(os.path.join(dataset_path, 'activity_labels.txt'), 'r') as f:
        for line in f:
            idx, label = line.strip().split()
            activity_labels[int(idx)] = label
    
    print("\n📋 Các hoạt động:")
    for idx, label in activity_labels.items():
        count_train = np.sum(y_train == idx)
        count_test = np.sum(y_test == idx)
        print(f"  {idx}. {label:20s} - Train: {count_train:4d}, Test: {count_test:4d}")
    
    # Chuyển labels về 0-indexed (từ 1-6 thành 0-5)
    y_train = y_train - 1
    y_test = y_test - 1
    
    print("\n" + "=" * 60)
    print("✅ Tải dữ liệu thành công!")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, subject_train, subject_test, activity_labels


def get_dataset_info(dataset_path='datasets/UCI HAR Dataset'):
    """
    Lấy thông tin về dataset
    
    Args:
        dataset_path: Đường dẫn đến thư mục dataset
        
    Returns:
        dict chứa thông tin dataset
    """
    info = {
        'num_features': 561,
        'num_classes': 6,
        'sampling_rate': 50,  # Hz
        'window_size': 2.56,  # seconds
        'overlap': 0.5,  # 50%
    }
    
    return info


if __name__ == '__main__':
    # Test data loader
    print("\n🧪 Testing Data Loader...\n")
    
    X_train, X_test, y_train, y_test, subject_train, subject_test, labels = load_uci_har_data()
    
    print("\n📊 Thống kê dữ liệu:")
    print(f"  - Số features: {X_train.shape[1]}")
    print(f"  - Số classes: {len(np.unique(y_train))}")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Số người tham gia train: {len(np.unique(subject_train))}")
    print(f"  - Số người tham gia test: {len(np.unique(subject_test))}")
    
    # Kiểm tra giá trị
    print("\n🔍 Kiểm tra dữ liệu:")
    print(f"  - X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"  - X_test min: {X_test.min():.4f}, max: {X_test.max():.4f}")
    print(f"  - y_train unique: {np.unique(y_train)}")
    print(f"  - y_test unique: {np.unique(y_test)}")
    
    print("\n✅ Data loader hoạt động tốt!")

