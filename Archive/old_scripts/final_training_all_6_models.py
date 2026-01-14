#!/usr/bin/env python3
"""
FINAL TRAINING - ALL 6 MODELS
Train all 6 models with improved architectures
"""

import sys
sys.path.insert(0, 'src')

print("=" * 80)
print("🚀 FINAL TRAINING - ALL 6 MODELS WITH IMPROVED ARCHITECTURES")
print("=" * 80)

from train_all_models import train_all_models

try:
    print("\n📋 Models to train:")
    print("   1. CNN Simple - Baseline model")
    print("   2. CNN Deep - Deep CNN with BatchNorm")
    print("   3. LSTM - Temporal modeling")
    print("   4. CNN-LSTM Hybrid - Best of both worlds")
    print("   5. Depthwise Separable CNN - IMPROVED with BatchNorm")
    print("   6. CNN with Attention - Attention mechanism")
    
    print("\n⚙️  Configuration:")
    print("   - Epochs: 50")
    print("   - Batch Size: 32")
    print("   - Learning Rate: 0.001")
    print("   - Early Stopping: patience=10")
    print("   - ReduceLROnPlateau: patience=5")
    
    print("\n" + "=" * 80)
    print("🏃 STARTING TRAINING...")
    print("=" * 80)
    
    results = train_all_models(epochs=50, batch_size=32, learning_rate=0.001)
    
    print("\n" + "=" * 80)
    print("✅ ALL TRAINING COMPLETE!")
    print("=" * 80)
    
    print("\n📊 FINAL RESULTS:")
    print("-" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_results, 1):
        acc = result.get('test_accuracy', 0) * 100
        loss = result.get('test_loss', 0)
        params = result.get('total_params', 0)
        time = result.get('training_time', 0)
        
        print(f"\n{i}. {model_name.upper()}")
        print(f"   Accuracy: {acc:.2f}%")
        print(f"   Loss: {loss:.4f}")
        print(f"   Parameters: {params:,}")
        print(f"   Training Time: {time:.1f}s")
    
    print("\n" + "=" * 80)
    print("📁 FILES GENERATED:")
    print("=" * 80)
    print("   ✅ models/har_model_*.h5 (6 trained models)")
    print("   ✅ models/training_results_comparison.json")
    print("   ✅ models/model_comparison_report.txt")
    print("   ✅ models/model_comparison_plots.png")
    
    print("\n" + "=" * 80)
    print("🎉 SUCCESS! ALL 6 MODELS TRAINED SUCCESSFULLY!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

