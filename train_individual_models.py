#!/usr/bin/env python3
"""Train individual models separately - can run in parallel"""

import sys
import argparse
sys.path.insert(0, 'src')

from train_all_models import train_all_models

def train_specific_models(model_names, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train specific models only
    
    Args:
        model_names: List of model names to train
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    print("=" * 80)
    print(f"TRAINING SPECIFIC MODELS: {', '.join(model_names)}")
    print("=" * 80)
    
    # Import and modify train_all_models to only train specific models
    import os
    import json
    import numpy as np
    from datetime import datetime
    from tensorflow import keras
    
    from data_loader import load_uci_har_data
    from preprocessing import preprocess_data, reshape_for_cnn
    from model import create_har_model, compile_model
    
    # Load data once
    print("\n📂 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _, _, activity_labels = load_uci_har_data()
    
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2,
        save_scaler=True
    )
    
    X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # Model configs
    all_configs = {
        'cnn_simple': 'Baseline CNN Simple',
        'cnn_deep': 'Deep CNN',
        'lstm': 'LSTM Model',
        'cnn_lstm': 'CNN-LSTM Hybrid',
        'depthwise_cnn': 'Depthwise Separable CNN',
        'cnn_attention': 'CNN with Attention',
    }
    
    results = {}
    
    # Train each specified model
    for i, model_type in enumerate(model_names, 1):
        if model_type not in all_configs:
            print(f"❌ Unknown model type: {model_type}")
            continue
            
        description = all_configs[model_type]
        
        print("\n" + "=" * 80)
        print(f"🎯 MODEL {i}/{len(model_names)}: {description} ({model_type})")
        print("=" * 80)
        
        try:
            # Create model
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
            
            # Train
            print(f"\n🚀 Training {model_type} for {epochs} epochs...")
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
            
            # Save results
            results[model_type] = {
                'description': description,
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss),
                'training_time': training_time,
                'total_params': int(model.count_params()),
                'model_size_kb': model.count_params() * 4 / 1024,
            }
            
            print(f"\n✅ {model_type} complete!")
            print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"   Test Loss: {test_loss:.4f}")
            print(f"   Training Time: {training_time:.1f}s")
            print(f"   Parameters: {model.count_params():,}")
            
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_path = f'models/training_results_{"_".join(model_names)}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ TRAINING COMPLETE FOR: {', '.join(model_names)}")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train specific HAR models')
    parser.add_argument('--models', nargs='+', required=True,
                       choices=['cnn_simple', 'cnn_deep', 'lstm', 'cnn_lstm', 'depthwise_cnn', 'cnn_attention'],
                       help='Models to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    results = train_specific_models(
        model_names=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

