#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFLite Conversion Script
Chuyển đổi Keras model sang TFLite và C header file cho ESP32
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

def convert_to_tflite(model_path, output_path=None, quantize=True, test_data=None):
    """
    Chuyển đổi Keras model sang TFLite
    
    Args:
        model_path: Đường dẫn Keras model (.h5)
        output_path: Đường dẫn output (.tflite)
        quantize: Có quantize không (int8)
        test_data: Dữ liệu test để calibrate quantization
        
    Returns:
        tflite_model (bytes)
    """
    print("=" * 60)
    print("🔄 CHUYỂN ĐỔI MODEL SANG TFLITE")
    print("=" * 60)
    
    # 1. Load Keras model
    print(f"\n📂 BƯỚC 1: Load Keras model từ {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: Không tìm thấy model tại {model_path}")
        sys.exit(1)
    
    model = keras.models.load_model(model_path)
    print("✅ Đã load model thành công")
    
    # Kích thước model gốc
    model_size = os.path.getsize(model_path)
    print(f"📏 Kích thước model gốc: {model_size / 1024:.2f} KB")
    
    # 2. Tạo converter
    print("\n🔧 BƯỚC 2: Tạo TFLite converter")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. Quantization
    if quantize:
        print("\n⚙️  BƯỚC 3: Áp dụng quantization (int8)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Nếu có test data, dùng để calibrate
        if test_data is not None:
            print("📊 Sử dụng representative dataset để calibrate")
            
            def representative_dataset():
                for i in range(min(100, len(test_data))):
                    yield [test_data[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        print("✅ Đã cấu hình quantization")
    else:
        print("\n⚙️  BƯỚC 3: Không quantize (giữ float32)")
    
    # 4. Convert
    print("\n🔄 BƯỚC 4: Đang convert...")
    try:
        tflite_model = converter.convert()
        print("✅ Convert thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi convert: {e}")
        sys.exit(1)
    
    # 5. Lưu TFLite model
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f'models/{model_name}.tflite'
    
    print(f"\n💾 BƯỚC 5: Lưu TFLite model tại {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = len(tflite_model)
    print(f"✅ Đã lưu TFLite model")
    print(f"📏 Kích thước TFLite: {tflite_size / 1024:.2f} KB")
    print(f"📉 Giảm: {(1 - tflite_size/model_size) * 100:.1f}%")
    
    # 6. Kiểm tra model
    print("\n🧪 BƯỚC 6: Kiểm tra TFLite model")
    verify_tflite_model(output_path, test_data)
    
    print("\n" + "=" * 60)
    print("✅ CHUYỂN ĐỔI HOÀN TẤT!")
    print("=" * 60)
    
    return tflite_model


def verify_tflite_model(tflite_path, test_data=None):
    """
    Kiểm tra TFLite model hoạt động đúng
    
    Args:
        tflite_path: Đường dẫn TFLite model
        test_data: Dữ liệu test
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Lấy input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Input shape: {input_details[0]['shape']}")
    print(f"✅ Input dtype: {input_details[0]['dtype']}")
    print(f"✅ Output shape: {output_details[0]['shape']}")
    print(f"✅ Output dtype: {output_details[0]['dtype']}")
    
    # Test inference nếu có test data
    if test_data is not None and len(test_data) > 0:
        print("\n🧪 Test inference với 1 sample...")
        
        # Chuẩn bị input
        test_sample = test_data[0:1].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        
        # Run inference
        interpreter.invoke()
        
        # Lấy output
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output[0])
        
        print(f"✅ Inference thành công!")
        print(f"✅ Predicted class: {predicted_class}")
        print(f"✅ Confidence: {output[0][predicted_class]:.4f}")


def convert_to_c_header(tflite_path, output_path=None):
    """
    Chuyển đổi TFLite model sang C header file cho ESP32
    
    Args:
        tflite_path: Đường dẫn TFLite model
        output_path: Đường dẫn output (.h)
        
    Returns:
        header_content (string)
    """
    print("\n" + "=" * 60)
    print("🔄 CHUYỂN ĐỔI SANG C HEADER FILE")
    print("=" * 60)
    
    # Đọc TFLite model
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()
    
    model_size = len(tflite_model)
    print(f"📏 Kích thước model: {model_size} bytes ({model_size/1024:.2f} KB)")
    
    # Tạo tên biến
    model_name = os.path.splitext(os.path.basename(tflite_path))[0]
    var_name = model_name.replace('-', '_').replace('.', '_')
    
    # Tạo C header content
    header_content = f"""// Auto-generated C header file for TFLite model
// Model: {model_name}
// Size: {model_size} bytes ({model_size/1024:.2f} KB)
// Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

const unsigned int {var_name}_len = {model_size};
const unsigned char {var_name}_data[] = {{
"""
    
    # Thêm dữ liệu model (16 bytes per line)
    for i in range(0, model_size, 16):
        line = "  "
        for j in range(16):
            if i + j < model_size:
                line += f"0x{tflite_model[i+j]:02x}, "
        header_content += line + "\n"
    
    header_content += "};\n\n#endif  // " + var_name.upper() + "_H\n"
    
    # Lưu header file
    if output_path is None:
        output_path = f'models/{model_name}.h'
    
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"✅ Đã lưu C header tại: {output_path}")
    print(f"📝 Tên biến: {var_name}_data")
    print(f"📝 Kích thước: {var_name}_len")
    
    print("\n💡 Cách sử dụng trong ESP32:")
    print(f'   #include "{os.path.basename(output_path)}"')
    print(f'   const tflite::Model* model = tflite::GetModel({var_name}_data);')
    
    return header_content


if __name__ == '__main__':
    # Cấu hình
    MODEL_PATH = 'models/har_model_cnn_simple.h5'
    QUANTIZE = True
    
    # Load test data để calibrate quantization
    print("📂 Load test data để calibrate quantization...")
    from data_loader import load_uci_har_data
    from preprocessing import preprocess_data, reshape_for_cnn
    
    X_train, X_test, y_train, y_test, _, _, _ = load_uci_har_data()
    X_train_p, X_val_p, X_test_p, _, _, _, _ = preprocess_data(
        X_train, X_test, y_train, y_test,
        save_scaler=False
    )
    _, _, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)
    
    # Convert sang TFLite
    tflite_model = convert_to_tflite(
        MODEL_PATH,
        quantize=QUANTIZE,
        test_data=X_test_r
    )
    
    # Convert sang C header
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    tflite_path = f'models/{model_name}.tflite'
    convert_to_c_header(tflite_path)
    
    print("\n✅ Script hoàn tất!")

