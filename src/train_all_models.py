#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train All Models Script
Train and compare multiple HAR model architectures
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras

# Import các module khác
from data_loader import load_uci_har_data
from preprocessing import preprocess_data, reshape_for_cnn
from model import create_har_model, compile_model

def train_all_models(epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train tất cả các model architectures và so sánh kết quả
    
    Args:
        epochs: Số epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        results_dict: Dictionary chứa kết quả của tất cả models
    """
    print("=" * 80)
    print("🚀 TRAINING ALL HAR MODELS - COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print("=" * 80)
    
    # 1. Load và preprocess dữ liệu (chỉ làm 1 lần)
    print("\n📂 STEP 1: Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _, _, activity_labels = load_uci_har_data()
    
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2,
        save_scaler=True
    )
    
    X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # 2. Định nghĩa các models cần train - TẤT CẢ 6 MODELS
    model_configs = [
        {'name': 'cnn_simple', 'description': 'Baseline CNN Simple'},
        {'name': 'cnn_deep', 'description': 'Deep CNN'},
        {'name': 'lstm', 'description': 'LSTM Model'},
        {'name': 'cnn_lstm', 'description': 'CNN-LSTM Hybrid'},
        {'name': 'depthwise_cnn', 'description': 'Depthwise Separable CNN'},
        {'name': 'cnn_attention', 'description': 'CNN with Attention'},
    ]
    
    results = {}
    
    # 3. Train từng model
    for i, config in enumerate(model_configs, 1):
        model_type = config['name']
        description = config['description']
        
        print("\n" + "=" * 80)
        print(f"🎯 MODEL {i}/{len(model_configs)}: {description} ({model_type})")
        print("=" * 80)
        
        try:
            # Tạo model
            print(f"\n🏗️  Creating {model_type} model...")
            model = create_har_model(
                input_shape=(X_train_r.shape[1], X_train_r.shape[2]),
                num_classes=6,
                model_type=model_type
            )
            model = compile_model(model, learning_rate=learning_rate)
            
            # Setup callbacks
            os.makedirs('models', exist_ok=True)
            model_path = f'models/har_model_{model_type}.h5'
            
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=0
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            # Training
            print(f"\n🎯 Training {model_type}...")
            start_time = datetime.now()
            
            history = model.fit(
                X_train_r, y_train_p,
                validation_data=(X_val_r, y_val_p),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluation
            print(f"\n📊 Evaluating {model_type}...")
            test_loss, test_acc = model.evaluate(X_test_r, y_test_p, verbose=0)
            
            # Lưu kết quả
            results[model_type] = {
                'description': description,
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss),
                'training_time': training_time,
                'total_params': int(model.count_params()),
                'model_size_mb': model.count_params() * 4 / (1024 * 1024),
                'history': {
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            }
            
            print(f"✅ {model_type} completed!")
            print(f"   - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"   - Test Loss: {test_loss:.4f}")
            print(f"   - Training Time: {training_time:.1f}s")
            print(f"   - Parameters: {model.count_params():,}")
            
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # 4. Lưu kết quả
    print("\n" + "=" * 80)
    print("💾 Saving results...")
    results_path = 'models/training_results_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {results_path}")

    # 5. Tạo comparison report
    print("\n📊 Generating comparison report...")
    generate_comparison_report(results)

    # 6. Vẽ comparison plots
    print("\n📈 Generating comparison plots...")
    plot_model_comparison(results)

    return results


def generate_comparison_report(results):
    """
    Tạo báo cáo so sánh các models
    """
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON REPORT")
    print("=" * 80)

    # Tạo comparison table
    comparison_data = []
    for model_name, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                'Model': result['description'],
                'Accuracy (%)': f"{result['test_accuracy']*100:.2f}",
                'Loss': f"{result['test_loss']:.4f}",
                'Parameters': f"{result['total_params']:,}",
                'Size (KB)': f"{result['model_size_mb']*1024:.2f}",
                'Training Time (s)': f"{result['training_time']:.1f}"
            })

    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))

    # Lưu báo cáo
    report_path = 'models/model_comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        # Thêm recommendations
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Best accuracy
        best_acc_model = max(results.items(),
                            key=lambda x: x[1].get('test_accuracy', 0) if 'error' not in x[1] else 0)
        f.write(f"🏆 Best Accuracy: {best_acc_model[1]['description']} ")
        f.write(f"({best_acc_model[1]['test_accuracy']*100:.2f}%)\n\n")

        # Smallest model
        smallest_model = min(results.items(),
                            key=lambda x: x[1].get('total_params', float('inf')) if 'error' not in x[1] else float('inf'))
        f.write(f"📦 Smallest Model: {smallest_model[1]['description']} ")
        f.write(f"({smallest_model[1]['total_params']:,} params, ")
        f.write(f"{smallest_model[1]['model_size_mb']*1024:.2f} KB)\n\n")

        # Fastest training
        fastest_model = min(results.items(),
                           key=lambda x: x[1].get('training_time', float('inf')) if 'error' not in x[1] else float('inf'))
        f.write(f"⚡ Fastest Training: {fastest_model[1]['description']} ")
        f.write(f"({fastest_model[1]['training_time']:.1f}s)\n\n")

    print(f"\n✅ Report saved to: {report_path}")


def plot_model_comparison(results):
    """
    Vẽ biểu đồ so sánh các models
    """
    # Filter out models with errors
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if not valid_results:
        print("⚠️  No valid results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = [v['description'] for v in valid_results.values()]

    # 1. Accuracy comparison
    accuracies = [v['test_accuracy'] * 100 for v in valid_results.values()]
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim([90, 100])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Model size comparison
    sizes = [v['model_size_mb'] * 1024 for v in valid_results.values()]
    axes[0, 1].bar(model_names, sizes, color='lightcoral')
    axes[0, 1].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Size (KB)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(sizes):
        axes[0, 1].text(i, v + 5, f'{v:.1f} KB', ha='center', fontweight='bold')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Training time comparison
    times = [v['training_time'] for v in valid_results.values()]
    axes[1, 0].bar(model_names, times, color='lightgreen')
    axes[1, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        axes[1, 0].text(i, v + max(times)*0.02, f'{v:.1f}s', ha='center', fontweight='bold')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Training history (accuracy)
    for model_name, result in valid_results.items():
        if 'history' in result:
            epochs = range(1, len(result['history']['val_accuracy']) + 1)
            axes[1, 1].plot(epochs, result['history']['val_accuracy'],
                          label=result['description'], linewidth=2)
    axes[1, 1].set_title('Validation Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = 'models/model_comparison_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_path}")
    plt.close()


if __name__ == '__main__':
    # Configuration
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Train all models
    results = train_all_models(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    print("\n" + "=" * 80)
    print("🎉 ALL MODELS TRAINING COMPLETE!")
    print("=" * 80)
    print("\nResults saved in models/ directory:")
    print("  - training_results_comparison.json")
    print("  - model_comparison_report.txt")
    print("  - model_comparison_plots.png")
    print("  - har_model_*.h5 (trained models)")
    print("\n✅ Script complete!")


