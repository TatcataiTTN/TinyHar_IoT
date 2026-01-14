#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo biểu đồ trực quan so sánh các models
Tất cả nhãn và chú thích bằng tiếng Việt
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Cấu hình font tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_training_results():
    """Đọc kết quả training từ file JSON"""
    with open('models/training_results_comparison.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def create_comparison_plots():
    """Tạo các biểu đồ so sánh"""
    
    # Đọc dữ liệu
    results = load_training_results()
    
    # Chuẩn bị dữ liệu
    model_names = []
    accuracies = []
    losses = []
    params = []
    sizes_mb = []
    times = []
    
    # Tên models bằng tiếng Việt
    model_labels = {
        'cnn_simple': 'CNN Đơn Giản',
        'cnn_deep': 'CNN Sâu',
        'lstm': 'LSTM',
        'cnn_lstm': 'CNN-LSTM',
        'depthwise_cnn': 'Depthwise CNN',
        'cnn_attention': 'CNN Attention'
    }
    
    for model_key in ['cnn_simple', 'cnn_deep', 'lstm', 'cnn_lstm', 'depthwise_cnn', 'cnn_attention']:
        if model_key in results:
            model_names.append(model_labels[model_key])
            accuracies.append(results[model_key]['test_accuracy'] * 100)
            losses.append(results[model_key]['test_loss'])
            params.append(results[model_key]['total_params'])
            sizes_mb.append(results[model_key]['model_size_mb'])
            times.append(results[model_key]['training_time'])
    
    # Tạo figure với 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Màu sắc cho các biểu đồ
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    # ========== Biểu đồ 1: So sánh Accuracy ==========
    ax1 = plt.subplot(2, 2, 1)
    bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Độ Chính Xác (%)', fontsize=12, fontweight='bold')
    ax1.set_title('So Sánh Độ Chính Xác Của Các Models', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim([75, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45)
    
    # Thêm giá trị lên đầu mỗi cột
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========== Biểu đồ 2: So sánh Kích thước Model ==========
    ax2 = plt.subplot(2, 2, 2)
    bars2 = ax2.bar(model_names, sizes_mb, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Kích Thước (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('So Sánh Kích Thước Models', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45)
    
    # Thêm giá trị lên đầu mỗi cột
    for bar, size in zip(bars2, sizes_mb):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{size:.2f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========== Biểu đồ 3: Accuracy vs Kích thước (Scatter) ==========
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(sizes_mb, accuracies, c=colors, s=300, alpha=0.7, edgecolors='black', linewidth=2)
    ax3.set_xlabel('Kích Thước Model (MB)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Độ Chính Xác (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Trade-off: Độ Chính Xác vs Kích Thước', fontsize=14, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Thêm nhãn cho từng điểm
    for i, name in enumerate(model_names):
        ax3.annotate(name, (sizes_mb[i], accuracies[i]), 
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    # ========== Biểu đồ 4: So sánh Thời gian Training ==========
    ax4 = plt.subplot(2, 2, 4)
    bars4 = ax4.bar(model_names, times, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Thời Gian Training (giây)', fontsize=12, fontweight='bold')
    ax4.set_title('So Sánh Thời Gian Training', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.tick_params(axis='x', rotation=45)
    
    # Thêm giá trị lên đầu mỗi cột
    for bar, time in zip(bars4, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{time:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    output_path = 'models/model_comparison_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ vào: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    print("=" * 80)
    print("📊 TẠO BIỂU ĐỒ TRỰC QUAN")
    print("=" * 80)
    
    create_comparison_plots()
    
    print("\n✅ HOÀN TẤT!")

