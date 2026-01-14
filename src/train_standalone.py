#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Training Script - Minimal Dependencies
Train one model at a time to avoid memory issues
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 80)
print("STANDALONE MODEL TRAINING")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Import TensorFlow
print("\n1. Loading TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__} loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading TensorFlow: {e}")
    sys.exit(1)

# Import project modules
print("\n2. Loading project modules...")
try:
    from src.data_loader import load_uci_har_data
    from src.preprocessing import preprocess_data, reshape_for_cnn
    from src.model import (
        create_cnn_simple,
        create_cnn_lstm_hybrid,
        create_depthwise_separable_cnn,
        create_cnn_attention
    )
    print("   ✅ All modules loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading modules: {e}")
    sys.exit(1)

# Load dataset
print("\n3. Loading UCI HAR dataset...")
try:
    X_train, X_test, y_train, y_test, _, _, activity_labels = load_uci_har_data()
    print(f"   ✅ Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
except Exception as e:
    print(f"   ❌ Error loading dataset: {e}")
    sys.exit(1)

# Preprocess data
print("\n4. Preprocessing data...")
try:
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, scaler = preprocess_data(
        X_train, X_test, y_train, y_test,
        scaler_type='standard',
        validation_split=0.2,
        save_scaler=True
    )
    X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    print(f"   ✅ Data preprocessed and reshaped")
    print(f"      Train: {X_train_r.shape}, Val: {X_val_r.shape}, Test: {X_test_r.shape}")
except Exception as e:
    print(f"   ❌ Error preprocessing data: {e}")
    sys.exit(1)

# Configuration
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model configurations
MODELS = {
    'cnn_simple': {
        'name': 'CNN Simple (Baseline)',
        'create_fn': create_cnn_simple,
        'description': 'Baseline CNN model'
    },
    'cnn_lstm': {
        'name': 'CNN-LSTM Hybrid',
        'create_fn': create_cnn_lstm_hybrid,
        'description': 'CNN with LSTM for temporal modeling'
    },
    'depthwise_cnn': {
        'name': 'Depthwise Separable CNN',
        'create_fn': create_depthwise_separable_cnn,
        'description': 'Ultra-lightweight CNN for ESP32'
    },
    'cnn_attention': {
        'name': 'CNN with Attention',
        'create_fn': create_cnn_attention,
        'description': 'CNN with self-attention mechanism'
    }
}

# Ask user which model to train
print("\n" + "=" * 80)
print("SELECT MODEL TO TRAIN")
print("=" * 80)
for i, (key, config) in enumerate(MODELS.items(), 1):
    print(f"{i}. {config['name']} - {config['description']}")
print(f"{len(MODELS)+1}. Train ALL models (one by one)")

choice = input(f"\nEnter your choice (1-{len(MODELS)+1}): ").strip()

try:
    choice_num = int(choice)
    if choice_num == len(MODELS) + 1:
        models_to_train = list(MODELS.keys())
    elif 1 <= choice_num <= len(MODELS):
        models_to_train = [list(MODELS.keys())[choice_num - 1]]
    else:
        print("Invalid choice!")
        sys.exit(1)
except ValueError:
    print("Invalid input!")
    sys.exit(1)

# Train selected models
results = {}

for model_key in models_to_train:
    config = MODELS[model_key]
    
    print("\n" + "=" * 80)
    print(f"TRAINING: {config['name']}")
    print("=" * 80)
    
    try:
        # Create model
        print(f"\n📦 Creating model...")
        model = config['create_fn']((561, 1), 6)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print summary
        print(f"\n📊 Model Summary:")
        model.summary()
        
        # Train model
        print(f"\n🏋️  Training for {EPOCHS} epochs...")
        history = model.fit(
            X_train_r, y_train_p,
            validation_data=(X_val_r, y_val_p),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\n📊 Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test_r, y_test_p, verbose=0)
        
        # Save model
        model_path = f'models/har_model_{model_key}.h5'
        os.makedirs('models', exist_ok=True)
        model.save(model_path)
        
        # Store results
        results[model_key] = {
            'name': config['name'],
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'model_path': model_path,
            'epochs': EPOCHS
        }
        
        print(f"\n✅ {config['name']} training complete!")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Model saved: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error training {config['name']}: {e}")
        results[model_key] = {'error': str(e)}

# Save results
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

for model_key, result in results.items():
    if 'error' not in result:
        print(f"\n✅ {result['name']}")
        print(f"   Accuracy: {result['test_accuracy']*100:.2f}%")
        print(f"   Model: {result['model_path']}")
    else:
        print(f"\n❌ {MODELS[model_key]['name']}")
        print(f"   Error: {result['error']}")

# Save JSON results
results_path = 'models/standalone_training_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {results_path}")
print(f"\n🎉 Training complete!")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

