#!/usr/bin/env python3
"""
View Final Training Results - Clean Summary
"""

import os
import json
from datetime import datetime

def get_latest_results():
    """Get latest training results for all 6 models"""
    
    models_dir = 'models'
    
    # Define the 6 models
    model_types = [
        'cnn_simple',
        'cnn_deep', 
        'lstm',
        'cnn_lstm',
        'depthwise_cnn',
        'cnn_attention'
    ]
    
    print("=" * 80)
    print("📊 FINAL TRAINING RESULTS - ALL 6 MODELS")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check for comparison file first
    comparison_file = os.path.join(models_dir, 'training_results_comparison.json')
    
    if os.path.exists(comparison_file):
        print("\n✅ Found training_results_comparison.json")
        
        with open(comparison_file, 'r') as f:
            results = json.load(f)
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
        
        print("\n" + "=" * 80)
        print("🏆 RANKING BY ACCURACY")
        print("=" * 80)
        print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'Loss':<10} {'Params':<12} {'Time':<10}")
        print("-" * 80)
        
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            acc = metrics.get('test_accuracy', 0) * 100
            loss = metrics.get('test_loss', 0)
            params = metrics.get('total_params', 0)
            time_s = metrics.get('training_time', 0)
            
            # Medal icons
            if i == 1:
                icon = "🥇"
            elif i == 2:
                icon = "🥈"
            elif i == 3:
                icon = "🥉"
            else:
                icon = f"{i}."
            
            print(f"{icon:<6} {model_name:<25} {acc:>6.2f}%     {loss:>6.4f}    {params:>10,}  {time_s:>7.1f}s")
        
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE CATEGORIES")
        print("=" * 80)
        
        excellent = [m for m, r in sorted_results if r.get('test_accuracy', 0) >= 0.95]
        good = [m for m, r in sorted_results if 0.90 <= r.get('test_accuracy', 0) < 0.95]
        acceptable = [m for m, r in sorted_results if 0.85 <= r.get('test_accuracy', 0) < 0.90]
        poor = [m for m, r in sorted_results if r.get('test_accuracy', 0) < 0.85]
        
        print(f"\n🥇 EXCELLENT (≥95%): {len(excellent)} models")
        for m in excellent:
            acc = results[m]['test_accuracy'] * 100
            print(f"   - {m}: {acc:.2f}%")
        
        print(f"\n🥈 GOOD (90-95%): {len(good)} models")
        for m in good:
            acc = results[m]['test_accuracy'] * 100
            print(f"   - {m}: {acc:.2f}%")
        
        print(f"\n🥉 ACCEPTABLE (85-90%): {len(acceptable)} models")
        for m in acceptable:
            acc = results[m]['test_accuracy'] * 100
            print(f"   - {m}: {acc:.2f}%")
        
        if poor:
            print(f"\n⚠️  NEEDS IMPROVEMENT (<85%): {len(poor)} models")
            for m in poor:
                acc = results[m]['test_accuracy'] * 100
                print(f"   - {m}: {acc:.2f}%")
        
        print("\n" + "=" * 80)
        print("💾 MODEL SIZES")
        print("=" * 80)
        
        for model_name, metrics in sorted_results:
            size_kb = metrics.get('model_size_kb', 0)
            params = metrics.get('total_params', 0)
            print(f"{model_name:<25} {size_kb:>8.2f} KB  ({params:>10,} params)")
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        
    else:
        print("\n⏳ Training still in progress...")
        print("   Run 'python monitor_training.py' to check status")

if __name__ == '__main__':
    get_latest_results()

