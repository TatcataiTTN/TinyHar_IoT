#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Evaluation Script
Evaluate all trained models and generate detailed reports
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras

# Import các module khác
from data_loader import load_uci_har_data
from preprocessing import preprocess_data, reshape_for_cnn


def evaluate_all_models():
    """
    Đánh giá tất cả các models đã train
    
    Returns:
        evaluation_results: Dictionary chứa kết quả evaluation
    """
    print("=" * 80)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    # 1. Load test data
    print("\n📂 Loading test data...")
    X_train, X_test, y_train, y_test, _, _, activity_labels = load_uci_har_data()
    
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2,
        save_scaler=False
    )
    
    X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # 2. Find all trained models
    model_files = [f for f in os.listdir('models') if f.startswith('har_model_') and f.endswith('.h5')]
    
    if not model_files:
        print("❌ No trained models found in models/ directory")
        return {}
    
    print(f"\n✅ Found {len(model_files)} trained models")
    
    # 3. Evaluate each model
    evaluation_results = {}
    class_names = [activity_labels[i+1] for i in range(6)]
    
    for model_file in model_files:
        model_path = os.path.join('models', model_file)
        model_name = model_file.replace('har_model_', '').replace('.h5', '')
        
        print(f"\n{'='*80}")
        print(f"📊 Evaluating: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Load model
            print(f"Loading model from {model_path}...")
            model = keras.models.load_model(model_path)
            
            # Predict
            print("Making predictions...")
            y_pred_proba = model.predict(X_test_r, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_p, y_pred)
            cm = confusion_matrix(y_test_p, y_pred)
            report = classification_report(y_test_p, y_pred, target_names=class_names, 
                                          output_dict=True, digits=4)
            
            # Store results
            evaluation_results[model_name] = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'predictions': y_pred.tolist(),
                'true_labels': y_test_p.tolist()
            }
            
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Plot confusion matrix
            plot_confusion_matrix(cm, class_names, model_name)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            evaluation_results[model_name] = {'error': str(e)}
    
    # 4. Generate comprehensive report
    print(f"\n{'='*80}")
    print("📝 Generating comprehensive evaluation report...")
    generate_evaluation_report(evaluation_results, class_names)
    
    # 5. Save results
    results_path = 'models/evaluation_results_all_models.json'
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"✅ Results saved to: {results_path}")
    
    return evaluation_results


def plot_confusion_matrix(cm, class_names, model_name):
    """
    Vẽ confusion matrix cho một model
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Vẽ heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Lưu figure
    plot_path = f'models/confusion_matrix_{model_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Confusion matrix saved: {plot_path}")
    plt.close()


def generate_evaluation_report(evaluation_results, class_names):
    """
    Tạo báo cáo evaluation chi tiết
    """
    report_path = 'models/comprehensive_evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall comparison
        f.write("1. OVERALL ACCURACY COMPARISON\n")
        f.write("-" * 80 + "\n\n")
        
        comparison_data = []
        for model_name, result in evaluation_results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{result['accuracy']*100:.2f}%"
                })
        
        df = pd.DataFrame(comparison_data)
        f.write(df.to_string(index=False) + "\n\n")
        
        # Per-class performance
        f.write("\n2. PER-CLASS PERFORMANCE\n")
        f.write("-" * 80 + "\n\n")
        
        for model_name, result in evaluation_results.items():
            if 'error' not in result:
                f.write(f"\n{model_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                report = result['classification_report']
                for class_name in class_names:
                    if class_name in report:
                        metrics = report[class_name]
                        f.write(f"  {class_name:20s}: ")
                        f.write(f"Precision={metrics['precision']:.4f}, ")
                        f.write(f"Recall={metrics['recall']:.4f}, ")
                        f.write(f"F1={metrics['f1-score']:.4f}\n")
                f.write("\n")
        
        # Best model per metric
        f.write("\n3. BEST MODELS\n")
        f.write("-" * 80 + "\n\n")
        
        valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}
        
        if valid_results:
            best_acc = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"🏆 Best Overall Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']*100:.2f}%)\n\n")
    
    print(f"✅ Comprehensive report saved to: {report_path}")


if __name__ == '__main__':
    results = evaluate_all_models()
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - evaluation_results_all_models.json")
    print("  - comprehensive_evaluation_report.txt")
    print("  - confusion_matrix_*.png (for each model)")
    print("\n✅ Script complete!")

