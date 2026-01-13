#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script cho HAR Model
Đánh giá model và tạo báo cáo chi tiết
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import các module khác
from data_loader import load_uci_har_data
from preprocessing import preprocess_data, reshape_for_cnn

def evaluate_model(model_path='models/har_model_cnn_simple.h5'):
    """
    Đánh giá model trên test set
    
    Args:
        model_path: Đường dẫn đến model đã train
        
    Returns:
        results dict
    """
    print("=" * 60)
    print("📊 ĐÁNH GIÁ HAR MODEL")
    print("=" * 60)
    
    # 1. Load model
    print(f"\n📂 BƯỚC 1: Load model từ {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: Không tìm thấy model tại {model_path}")
        print("Hãy chạy: python src/train.py")
        sys.exit(1)
    
    model = keras.models.load_model(model_path)
    print("✅ Đã load model thành công")
    model.summary()
    
    # 2. Load dữ liệu
    print("\n📂 BƯỚC 2: Load test data")
    X_train, X_test, y_train, y_test, _, _, activity_labels = load_uci_har_data()
    
    # 3. Preprocessing
    print("\n🔧 BƯỚC 3: Preprocessing")
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2,
        save_scaler=False
    )
    
    # 4. Reshape
    print("\n🔄 BƯỚC 4: Reshape dữ liệu")
    _, _, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # 5. Dự đoán
    print("\n🎯 BƯỚC 5: Dự đoán trên test set")
    y_pred_proba = model.predict(X_test_r, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 6. Tính metrics
    print("\n📊 BƯỚC 6: Tính toán metrics")
    accuracy = accuracy_score(y_test_p, y_pred)
    
    print("\n" + "=" * 60)
    print(f"✅ ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)
    
    # 7. Classification report
    print("\n📋 BƯỚC 7: Classification Report")
    class_names = [activity_labels[i+1] for i in range(6)]
    report = classification_report(y_test_p, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # 8. Confusion matrix
    print("\n📊 BƯỚC 8: Confusion Matrix")
    cm = confusion_matrix(y_test_p, y_pred)
    plot_confusion_matrix(cm, class_names, model_path)
    
    # 9. Lưu kết quả
    print("\n💾 BƯỚC 9: Lưu kết quả")
    save_evaluation_results(accuracy, report, cm, model_path)
    
    # 10. Phân tích lỗi
    print("\n🔍 BƯỚC 10: Phân tích lỗi")
    analyze_errors(y_test_p, y_pred, class_names)
    
    print("\n" + "=" * 60)
    print("🎉 ĐÁNH GIÁ HOÀN TẤT!")
    print("=" * 60)
    
    results = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_labels': y_test_p,
        'confusion_matrix': cm
    }
    
    return results


def plot_confusion_matrix(cm, class_names, model_path):
    """
    Vẽ confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Tên các classes
        model_path: Đường dẫn model (để đặt tên file)
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Vẽ heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Lưu figure
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = f'models/confusion_matrix_{model_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu confusion matrix tại: {plot_path}")
    
    plt.close()


def save_evaluation_results(accuracy, report, cm, model_path):
    """
    Lưu kết quả evaluation vào file
    
    Args:
        accuracy: Accuracy score
        report: Classification report
        cm: Confusion matrix
        model_path: Đường dẫn model
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results_path = f'models/evaluation_results_{model_name}.txt'
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("KẾT QUẢ ĐÁNH GIÁ HAR MODEL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 60 + "\n")
        f.write(str(cm))
        f.write("\n")
    
    print(f"✅ Đã lưu kết quả tại: {results_path}")


def analyze_errors(y_true, y_pred, class_names):
    """
    Phân tích các lỗi dự đoán
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Tên các classes
    """
    errors = y_true != y_pred
    num_errors = np.sum(errors)
    
    print(f"\n📊 Tổng số lỗi: {num_errors}/{len(y_true)} ({num_errors/len(y_true)*100:.2f}%)")
    
    if num_errors > 0:
        print("\n🔍 Top 5 cặp lỗi thường gặp:")
        error_pairs = {}
        for true_label, pred_label in zip(y_true[errors], y_pred[errors]):
            pair = (class_names[true_label], class_names[pred_label])
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        sorted_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
        for i, ((true_class, pred_class), count) in enumerate(sorted_pairs[:5], 1):
            print(f"  {i}. {true_class:20s} → {pred_class:20s}: {count:3d} lỗi")


if __name__ == '__main__':
    # Đánh giá model
    MODEL_PATH = 'models/har_model_cnn_simple.h5'
    
    results = evaluate_model(MODEL_PATH)
    
    print("\n✅ Script hoàn tất!")

