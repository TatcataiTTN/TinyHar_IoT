#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Pipeline Script
Complete end-to-end pipeline: Train → Evaluate → Deploy
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_pipeline(train=True, evaluate=True, deploy=True, epochs=50):
    """
    Run complete pipeline
    
    Args:
        train: Run training phase
        evaluate: Run evaluation phase
        deploy: Run deployment phase
        epochs: Number of training epochs
    """
    start_time = time.time()
    
    print_header("🚀 TINYHAR COMPLETE PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Training: {'Enabled' if train else 'Skipped'}")
    print(f"  - Evaluation: {'Enabled' if evaluate else 'Skipped'}")
    print(f"  - Deployment: {'Enabled' if deploy else 'Skipped'}")
    print(f"  - Epochs: {epochs}")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'phases': {}
    }
    
    # Phase 1: Training
    if train:
        print_header("PHASE 1: TRAINING ALL MODELS")
        try:
            from train_all_models import train_all_models
            train_results = train_all_models(epochs=epochs, batch_size=32, learning_rate=0.001)
            results['phases']['training'] = {
                'status': 'success',
                'models_trained': len(train_results)
            }
            print("\n✅ Training phase complete!")
        except Exception as e:
            print(f"\n❌ Training phase failed: {e}")
            results['phases']['training'] = {
                'status': 'failed',
                'error': str(e)
            }
            return results
    
    # Phase 2: Evaluation
    if evaluate:
        print_header("PHASE 2: EVALUATING ALL MODELS")
        try:
            from evaluate_all_models import evaluate_all_models
            eval_results = evaluate_all_models()
            results['phases']['evaluation'] = {
                'status': 'success',
                'models_evaluated': len(eval_results)
            }
            print("\n✅ Evaluation phase complete!")
        except Exception as e:
            print(f"\n❌ Evaluation phase failed: {e}")
            results['phases']['evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Phase 3: Deployment
    if deploy:
        print_header("PHASE 3: DEPLOYING ALL MODELS")
        try:
            from deploy_all_models import deploy_all_models
            deploy_results = deploy_all_models(quantize=True)
            results['phases']['deployment'] = {
                'status': 'success',
                'models_deployed': len(deploy_results)
            }
            print("\n✅ Deployment phase complete!")
        except Exception as e:
            print(f"\n❌ Deployment phase failed: {e}")
            results['phases']['deployment'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Summary
    elapsed_time = time.time() - start_time
    results['end_time'] = datetime.now().isoformat()
    results['elapsed_time_seconds'] = elapsed_time
    
    print_header("🎉 PIPELINE COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_time/60:.1f} minutes")
    
    print("\n📊 Summary:")
    for phase, result in results['phases'].items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status_icon} {phase.capitalize()}: {result['status']}")
    
    print("\n📁 Generated files in models/ directory:")
    print("  Training:")
    print("    - har_model_*.h5 (trained models)")
    print("    - training_results_comparison.json")
    print("    - model_comparison_report.txt")
    print("    - model_comparison_plots.png")
    print("  Evaluation:")
    print("    - evaluation_results_all_models.json")
    print("    - comprehensive_evaluation_report.txt")
    print("    - confusion_matrix_*.png")
    print("  Deployment:")
    print("    - *.tflite (TensorFlow Lite models)")
    print("    - *.h (C header files for ESP32)")
    print("    - deployment_results.json")
    print("    - deployment_report.txt")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TinyHAR Complete Pipeline')
    parser.add_argument('--skip-train', action='store_true', help='Skip training phase')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--skip-deploy', action='store_true', help='Skip deployment phase')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 epochs)')
    
    args = parser.parse_args()
    
    epochs = 10 if args.quick else args.epochs
    
    results = run_pipeline(
        train=not args.skip_train,
        evaluate=not args.skip_eval,
        deploy=not args.skip_deploy,
        epochs=epochs
    )
    
    print("\n✅ All done!")

