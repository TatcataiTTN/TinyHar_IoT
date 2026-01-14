#!/usr/bin/env python3
"""Monitor training progress - Shows only latest results"""

import os
import time
import json
from datetime import datetime

def get_latest_result_for_model(models_dir, model_type):
    """Get the latest result file for a specific model type"""
    # Look for result files matching this model type
    pattern = f"training_results_{model_type}"
    matching_files = []

    for f in os.listdir(models_dir):
        if f.startswith(pattern) and f.endswith('.json'):
            file_path = os.path.join(models_dir, f)
            mtime = os.path.getmtime(file_path)
            matching_files.append((f, mtime))

    if matching_files:
        # Return the most recently modified file
        latest = max(matching_files, key=lambda x: x[1])
        return latest[0]
    return None

def check_training_progress():
    """Check training progress from log files and model files"""

    print("=" * 80)
    print(f"📊 TRAINING PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Define the 6 models we care about
    model_types = [
        'cnn_simple',
        'cnn_deep',
        'lstm',
        'cnn_lstm',
        'depthwise_cnn',
        'cnn_attention'
    ]

    models_dir = 'models'

    # Check for trained models
    if os.path.exists(models_dir):
        print(f"\n✅ LATEST TRAINED MODELS (6 models):")
        print("-" * 80)

        for model_type in model_types:
            model_file = f'har_model_{model_type}.h5'
            model_path = os.path.join(models_dir, model_file)

            if os.path.exists(model_path):
                size_kb = os.path.getsize(model_path) / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                status = "✅"
            else:
                size_kb = 0
                mtime = None
                status = "⏳"

            if mtime:
                print(f"{status} {model_type:20s} {size_kb:8.2f} KB  (Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print(f"{status} {model_type:20s} {'Not trained yet':>30s}")
    else:
        print("\n❌ No models directory found yet")

    # Check training logs - only latest
    logs_dir = 'training_logs'
    if os.path.exists(logs_dir):
        print(f"\n📝 TRAINING STATUS:")
        print("-" * 80)

        for model_type in model_types:
            log_file = f'{model_type}_training.log'
            log_path = os.path.join(logs_dir, log_file)

            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()

                    # Find epoch progress
                    epoch_lines = [l for l in lines if 'Epoch' in l and '/' in l]
                    if epoch_lines:
                        last_epoch = epoch_lines[-1].strip()
                        # Extract epoch number
                        if 'Epoch' in last_epoch:
                            epoch_part = last_epoch.split('Epoch')[1].split(':')[0].strip()
                            print(f"✅ {model_type:20s} {epoch_part:>15s}")
                        else:
                            print(f"🔄 {model_type:20s} {'Training...':>15s}")
                    else:
                        print(f"⏳ {model_type:20s} {'Starting...':>15s}")
                except:
                    print(f"⏳ {model_type:20s} {'Reading...':>15s}")
            else:
                print(f"⏳ {model_type:20s} {'Not started':>15s}")

    # Check results - only latest for each model
    print(f"\n📊 LATEST RESULTS (Test Accuracy):")
    print("-" * 80)

    if os.path.exists(models_dir):
        all_results = {}

        # Collect latest results for each model
        for model_type in model_types:
            result_file = get_latest_result_for_model(models_dir, model_type)

            if result_file:
                result_path = os.path.join(models_dir, result_file)
                try:
                    with open(result_path, 'r') as f:
                        results = json.load(f)

                    # Extract accuracy for this model
                    if model_type in results and isinstance(results[model_type], dict):
                        metrics = results[model_type]
                        if 'test_accuracy' in metrics:
                            all_results[model_type] = {
                                'accuracy': metrics['test_accuracy'] * 100,
                                'loss': metrics.get('test_loss', 0),
                                'params': metrics.get('total_params', 0),
                                'time': metrics.get('training_time', 0)
                            }
                except:
                    pass

        # Display results sorted by accuracy
        if all_results:
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

            for i, (model_type, metrics) in enumerate(sorted_results, 1):
                acc = metrics['accuracy']
                loss = metrics['loss']
                params = metrics['params']
                time_s = metrics['time']

                # Color coding
                if acc >= 95:
                    icon = "🥇"
                elif acc >= 90:
                    icon = "🥈"
                elif acc >= 85:
                    icon = "🥉"
                else:
                    icon = "⚠️"

                print(f"{icon} {i}. {model_type:20s} Acc: {acc:6.2f}%  Loss: {loss:.4f}  Params: {params:>8,}  Time: {time_s:>5.0f}s")
        else:
            print("   (No results available yet)")

    print("\n" + "=" * 80)
    print("💡 Tips:")
    print("   - Run this script again to see updated progress")
    print("   - Check individual logs: tail -f training_logs/<model>_training.log")
    print("   - View full output: cat final_training_output.txt")
    print("=" * 80)

if __name__ == '__main__':
    check_training_progress()

