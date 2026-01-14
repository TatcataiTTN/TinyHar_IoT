#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chuyển đổi models sang TensorFlow Lite và C arrays cho ESP32
Tất cả comments bằng tiếng Việt
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

def convert_to_tflite(model_path, output_path, quantize=True, model_type='cnn'):
    """
    Chuyển đổi model Keras sang TensorFlow Lite

    Args:
        model_path: Đường dẫn đến file .h5
        output_path: Đường dẫn output file .tflite
        quantize: Có áp dụng quantization không (int8)
        model_type: Loại model ('cnn', 'lstm', 'cnn_lstm')

    Returns:
        Kích thước file .tflite (bytes)
    """
    # Load model
    model = keras.models.load_model(model_path)

    # Tạo converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        # Áp dụng quantization int8 (chỉ weights, không quantize input/output)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset để calibrate quantization
        def representative_dataset():
            # Load dữ liệu mẫu để calibrate
            # Shape phải khớp với input của model: (1, 561, 1)
            for _ in range(100):
                data = np.random.randn(1, 561, 1).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset

        # Đối với LSTM models, cần thêm SELECT_TF_OPS
        if 'lstm' in model_type.lower():
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Convert
    tflite_model = converter.convert()

    # Lưu file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    return len(tflite_model)

def convert_to_c_array(tflite_path, output_path, model_name):
    """
    Chuyển đổi file .tflite sang C header file
    
    Args:
        tflite_path: Đường dẫn đến file .tflite
        output_path: Đường dẫn output file .h
        model_name: Tên model (dùng cho tên biến trong C)
    """
    # Đọc file .tflite
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    # Tạo C array
    hex_array = ', '.join([f'0x{b:02x}' for b in tflite_data])
    
    # Tạo nội dung file .h
    c_code = f"""// Auto-generated C header file cho model: {model_name}
// Kích thước: {len(tflite_data)} bytes
// Ngày tạo: 2026-01-14

#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

// Kích thước model (bytes)
const unsigned int {model_name}_model_len = {len(tflite_data)};

// Model data (TensorFlow Lite format)
const unsigned char {model_name}_model[] = {{
  {hex_array}
}};

#endif  // {model_name.upper()}_MODEL_H
"""
    
    # Lưu file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(c_code)
    
    return len(tflite_data)

def process_all_models():
    """Xử lý tất cả 6 models"""
    
    # Tạo thư mục output
    os.makedirs('models/tflite', exist_ok=True)
    os.makedirs('models/c_arrays', exist_ok=True)
    
    # Danh sách models
    models = {
        'cnn_simple': 'CNN Đơn Giản',
        'cnn_deep': 'CNN Sâu',
        'lstm': 'LSTM',
        'cnn_lstm': 'CNN-LSTM',
        'depthwise_cnn': 'Depthwise CNN',
        'cnn_attention': 'CNN Attention'
    }
    
    results = {}
    
    print("=" * 80)
    print("🔄 CHUYỂN ĐỔI MODELS SANG TENSORFLOW LITE VÀ C ARRAYS")
    print("=" * 80)
    
    for i, (model_key, model_name) in enumerate(models.items(), 1):
        print(f"\n📦 [{i}/6] Xử lý: {model_name} ({model_key})")
        print("-" * 80)
        
        model_path = f'models/har_model_{model_key}.h5'
        
        if not os.path.exists(model_path):
            print(f"   ⚠️  File không tồn tại: {model_path}")
            continue
        
        try:
            # Lấy kích thước gốc
            original_size = os.path.getsize(model_path)
            
            # Chuyển sang TFLite (float32)
            tflite_float_path = f'models/tflite/{model_key}_float32.tflite'
            print(f"   🔄 Chuyển sang TFLite (float32)...")
            tflite_float_size = convert_to_tflite(model_path, tflite_float_path, quantize=False, model_type=model_key)

            # Chuyển sang TFLite (int8 quantized)
            tflite_int8_path = f'models/tflite/{model_key}_int8.tflite'
            print(f"   🔄 Chuyển sang TFLite (int8 quantized)...")
            tflite_int8_size = convert_to_tflite(model_path, tflite_int8_path, quantize=True, model_type=model_key)
            
            # Chuyển sang C array (int8)
            c_array_path = f'models/c_arrays/{model_key}_model.h'
            print(f"   🔄 Chuyển sang C header file...")
            c_array_size = convert_to_c_array(tflite_int8_path, c_array_path, model_key)
            
            # Tính tỷ lệ giảm
            reduction_float = (1 - tflite_float_size / original_size) * 100
            reduction_int8 = (1 - tflite_int8_size / original_size) * 100
            
            # Lưu kết quả
            results[model_key] = {
                'name': model_name,
                'original_size_bytes': original_size,
                'original_size_kb': original_size / 1024,
                'tflite_float32_size_bytes': tflite_float_size,
                'tflite_float32_size_kb': tflite_float_size / 1024,
                'tflite_int8_size_bytes': tflite_int8_size,
                'tflite_int8_size_kb': tflite_int8_size / 1024,
                'reduction_float32_percent': reduction_float,
                'reduction_int8_percent': reduction_int8
            }
            
            print(f"   ✅ Hoàn tất!")
            print(f"      • Gốc (.h5):          {original_size:,} bytes ({original_size/1024:.2f} KB)")
            print(f"      • TFLite (float32):   {tflite_float_size:,} bytes ({tflite_float_size/1024:.2f} KB) - Giảm {reduction_float:.1f}%")
            print(f"      • TFLite (int8):      {tflite_int8_size:,} bytes ({tflite_int8_size/1024:.2f} KB) - Giảm {reduction_int8:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
    
    # Lưu kết quả vào JSON
    results_path = 'models/conversion_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT CHUYỂN ĐỔI TẤT CẢ MODELS")
    print("=" * 80)
    print(f"\n📄 Kết quả đã lưu vào: {results_path}")
    
    return results

if __name__ == '__main__':
    results = process_all_models()

