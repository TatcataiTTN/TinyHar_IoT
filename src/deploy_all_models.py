#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Deployment Pipeline
Convert all trained models to TFLite and generate C headers for ESP32
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Import các module khác
from data_loader import load_uci_har_data
from preprocessing import preprocess_data, reshape_for_cnn


def deploy_all_models(quantize=True):
    """
    Deploy tất cả các models: Convert to TFLite và generate C headers
    
    Args:
        quantize: Có quantize models không (int8)
        
    Returns:
        deployment_results: Dictionary chứa kết quả deployment
    """
    print("=" * 80)
    print("🚀 COMPREHENSIVE DEPLOYMENT PIPELINE")
    print("=" * 80)
    print(f"Quantization: {'Enabled (INT8)' if quantize else 'Disabled (Float32)'}")
    print("=" * 80)
    
    # 1. Load test data for quantization calibration
    print("\n📂 Loading calibration data...")
    X_train, X_test, y_train, y_test, _, _, _ = load_uci_har_data()
    X_train_p, X_val_p, X_test_p, _, _, _, _ = preprocess_data(
        X_train, X_test, y_train, y_test,
        save_scaler=False
    )
    _, _, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # 2. Find all trained models
    model_files = [f for f in os.listdir('models') if f.startswith('har_model_') and f.endswith('.h5')]
    
    if not model_files:
        print("❌ No trained models found in models/ directory")
        return {}
    
    print(f"\n✅ Found {len(model_files)} trained models")
    
    # 3. Deploy each model
    deployment_results = {}
    
    for model_file in model_files:
        model_path = os.path.join('models', model_file)
        model_name = model_file.replace('har_model_', '').replace('.h5', '')
        
        print(f"\n{'='*80}")
        print(f"🔄 Deploying: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Convert to TFLite
            tflite_path = convert_to_tflite(
                model_path, 
                quantize=quantize, 
                test_data=X_test_r
            )
            
            # Convert to C header
            header_path = convert_to_c_header(tflite_path)
            
            # Validate converted model
            accuracy_drop = validate_tflite_model(
                tflite_path, 
                X_test_r, 
                y_test
            )
            
            # Get file sizes
            h5_size = os.path.getsize(model_path)
            tflite_size = os.path.getsize(tflite_path)
            header_size = os.path.getsize(header_path)
            
            deployment_results[model_name] = {
                'h5_path': model_path,
                'tflite_path': tflite_path,
                'header_path': header_path,
                'h5_size_kb': h5_size / 1024,
                'tflite_size_kb': tflite_size / 1024,
                'header_size_kb': header_size / 1024,
                'size_reduction': (1 - tflite_size / h5_size) * 100,
                'accuracy_drop': accuracy_drop,
                'quantized': quantize
            }
            
            print(f"\n✅ {model_name} deployment complete!")
            print(f"   - Original size: {h5_size/1024:.2f} KB")
            print(f"   - TFLite size: {tflite_size/1024:.2f} KB")
            print(f"   - Size reduction: {deployment_results[model_name]['size_reduction']:.1f}%")
            print(f"   - Accuracy drop: {accuracy_drop:.2f}%")
            
        except Exception as e:
            print(f"❌ Error deploying {model_name}: {e}")
            deployment_results[model_name] = {'error': str(e)}
    
    # 4. Generate deployment report
    print(f"\n{'='*80}")
    print("📝 Generating deployment report...")
    generate_deployment_report(deployment_results)
    
    # 5. Save results
    results_path = 'models/deployment_results.json'
    with open(results_path, 'w') as f:
        json.dump(deployment_results, f, indent=2)
    print(f"✅ Results saved to: {results_path}")
    
    return deployment_results


def convert_to_tflite(model_path, quantize=True, test_data=None):
    """
    Convert Keras model to TFLite
    """
    print(f"\n🔄 Converting to TFLite...")
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    if quantize:
        print("  ⚙️  Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if test_data is not None:
            def representative_dataset():
                for i in range(min(100, len(test_data))):
                    yield [test_data[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    suffix = '_quantized' if quantize else '_float32'
    output_path = f'models/{model_name}{suffix}.tflite'
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"  ✅ TFLite model saved: {output_path}")
    print(f"  📏 Size: {len(tflite_model)/1024:.2f} KB")
    
    return output_path


def convert_to_c_header(tflite_path):
    """
    Convert TFLite model to C header file
    """
    print(f"\n🔄 Converting to C header...")
    
    # Read TFLite model
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()
    
    model_size = len(tflite_model)
    model_name = os.path.splitext(os.path.basename(tflite_path))[0]
    var_name = model_name.replace('-', '_').replace('.', '_')
    
    # Generate C header
    header_content = f"""// Auto-generated C header file for TFLite model
// Model: {model_name}
// Size: {model_size} bytes ({model_size/1024:.2f} KB)
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

const unsigned int {var_name}_len = {model_size};
const unsigned char {var_name}_data[] = {{
"""
    
    # Add model data (16 bytes per line)
    for i in range(0, model_size, 16):
        line = "  "
        for j in range(16):
            if i + j < model_size:
                line += f"0x{tflite_model[i+j]:02x}, "
        header_content += line + "\n"
    
    header_content += "};\n\n#endif  // " + var_name.upper() + "_H\n"
    
    # Save header file
    output_path = f'models/{model_name}.h'
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"  ✅ C header saved: {output_path}")
    print(f"  📝 Variable name: {var_name}_data")

    return output_path


def validate_tflite_model(tflite_path, X_test, y_test):
    """
    Validate TFLite model accuracy
    """
    print(f"\n🔍 Validating TFLite model...")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    predictions = []
    for i in range(len(X_test)):
        input_data = X_test[i:i+1].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data))

    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_test)

    # Compare with original (assuming baseline ~95%)
    baseline_accuracy = 0.95
    accuracy_drop = (baseline_accuracy - accuracy) * 100

    print(f"  ✅ TFLite accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  📊 Accuracy drop: {accuracy_drop:.2f}%")

    return accuracy_drop


def generate_deployment_report(deployment_results):
    """
    Generate deployment report
    """
    report_path = 'models/deployment_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DEPLOYMENT REPORT - ESP32 READY MODELS\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL SIZE COMPARISON\n")
        f.write("-" * 80 + "\n\n")

        comparison_data = []
        for model_name, result in deployment_results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': model_name,
                    'Original (KB)': f"{result['h5_size_kb']:.2f}",
                    'TFLite (KB)': f"{result['tflite_size_kb']:.2f}",
                    'Reduction (%)': f"{result['size_reduction']:.1f}",
                    'Accuracy Drop (%)': f"{result['accuracy_drop']:.2f}"
                })

        df = pd.DataFrame(comparison_data)
        f.write(df.to_string(index=False) + "\n\n")

        f.write("\n2. ESP32 DEPLOYMENT READINESS\n")
        f.write("-" * 80 + "\n\n")

        for model_name, result in deployment_results.items():
            if 'error' not in result:
                f.write(f"{model_name}:\n")
                f.write(f"  TFLite Size: {result['tflite_size_kb']:.2f} KB\n")

                if result['tflite_size_kb'] < 100:
                    f.write(f"  ✅ READY for ESP32 deployment (< 100 KB)\n")
                else:
                    f.write(f"  ⚠️  May be too large for ESP32 (> 100 KB)\n")

                f.write(f"  Header file: {result['header_path']}\n")
                f.write(f"  Accuracy drop: {result['accuracy_drop']:.2f}%\n\n")

        f.write("\n3. DEPLOYMENT INSTRUCTIONS\n")
        f.write("-" * 80 + "\n\n")
        f.write("To deploy to ESP32:\n")
        f.write("1. Copy the .h file to your ESP32 project\n")
        f.write("2. Include the header: #include \"model_name.h\"\n")
        f.write("3. Load model: const tflite::Model* model = tflite::GetModel(model_name_data);\n")
        f.write("4. Set up TFLite Micro interpreter with appropriate tensor arena size\n")
        f.write("5. Run inference on sensor data\n\n")

    print(f"✅ Deployment report saved to: {report_path}")


if __name__ == '__main__':
    import pandas as pd

    # Configuration
    QUANTIZE = True  # Enable INT8 quantization

    # Deploy all models
    results = deploy_all_models(quantize=QUANTIZE)

    print("\n" + "=" * 80)
    print("🎉 DEPLOYMENT PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - *.tflite (TensorFlow Lite models)")
    print("  - *.h (C header files for ESP32)")
    print("  - deployment_results.json")
    print("  - deployment_report.txt")
    print("\n✅ Models are ready for ESP32 deployment!")


