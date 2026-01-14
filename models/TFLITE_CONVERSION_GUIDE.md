# 📊 BÁO CÁO CHUYỂN ĐỔI MODELS SANG TENSORFLOW LITE VÀ C ARRAYS

## ✅ TỔNG QUAN

**Ngày:** 2026-01-14
**Số models đã chuyển đổi thành công:** 4/6
**Định dạng output:** TensorFlow Lite (.tflite) và C Header Files (.h)

---

## 📈 KẾT QUẢ CHUYỂN ĐỔI

### ✅ **Models đã chuyển đổi thành công:**

| Model | Gốc (.h5) | TFLite Float32 | TFLite Int8 | Giảm Int8 | C Array |
|-------|-----------|----------------|-------------|-----------|---------|
| **CNN Simple** | 3,365 KB | 1,114 KB | **287 KB** | 91.5% | ✅ |
| **CNN Deep** | 696 KB | 217 KB | **73 KB** | 89.6% | ✅ |
| **Depthwise CNN** | 462 KB | 132 KB | **61 KB** | 86.7% | ✅ |
| **CNN Attention** | 444 KB | 137 KB | **55 KB** | 87.6% | ✅ |

### ❌ **Models KHÔNG thể chuyển đổi:**

| Model | Lý do | Giải pháp |
|-------|-------|-----------|
| **LSTM** | TensorFlow Lite không hỗ trợ LSTM layers tốt | Sử dụng SELECT_TF_OPS (tăng kích thước đáng kể) |
| **CNN-LSTM** | Chứa LSTM layer không tương thích | Sử dụng SELECT_TF_OPS (tăng kích thước đáng kể) |

---

## 🎯 KHUYẾN NGHỊ CHO ESP32

### **Top 3 Models phù hợp nhất:**

#### 🥇 **1. CNN Deep (KHUYẾN NGHỊ NHẤT)**
- **Kích thước:** 73 KB (int8 quantized)
- **Accuracy:** 92.06%
- **Trade-off:** Tuyệt vời - Accuracy cao, kích thước nhỏ
- **Phù hợp:** ESP32 với 4 MB Flash
- **File C header:** `models/c_arrays/cnn_deep_model.h`

#### 🥈 **2. Depthwise CNN**
- **Kích thước:** 61 KB (int8 quantized)
- **Accuracy:** 81.71%
- **Trade-off:** Nhỏ nhất nhưng accuracy thấp
- **Phù hợp:** ESP32 với 2 MB Flash hoặc bộ nhớ hạn chế
- **File C header:** `models/c_arrays/depthwise_cnn_model.h`

#### 🥉 **3. CNN Attention**
- **Kích thước:** 55 KB (int8 quantized)
- **Accuracy:** 86.83%
- **Trade-off:** Nhỏ nhất, accuracy khá tốt
- **Phù hợp:** ESP32 với bộ nhớ hạn chế
- **File C header:** `models/c_arrays/cnn_attention_model.h`

---

## 📁 CẤU TRÚC THƯ MỤC

```
models/
├── tflite/                          # TensorFlow Lite models
│   ├── cnn_simple_float32.tflite   # 1,114 KB
│   ├── cnn_simple_int8.tflite      # 287 KB
│   ├── cnn_deep_float32.tflite     # 217 KB
│   ├── cnn_deep_int8.tflite        # 73 KB ⭐
│   ├── depthwise_cnn_float32.tflite # 132 KB
│   ├── depthwise_cnn_int8.tflite   # 61 KB
│   ├── cnn_attention_float32.tflite # 137 KB
│   └── cnn_attention_int8.tflite   # 55 KB
│
└── c_arrays/                        # C header files cho ESP32
    ├── cnn_simple_model.h          # 287 KB
    ├── cnn_deep_model.h            # 73 KB ⭐
    ├── depthwise_cnn_model.h       # 61 KB
    └── cnn_attention_model.h       # 55 KB
```

---

## 💻 HƯỚNG DẪN SỬ DỤNG TRÊN ESP32

### **Bước 1: Include header file**

```cpp
// Trong file Arduino sketch (.ino)
#include "cnn_deep_model.h"  // Hoặc model khác

// Model data đã được định nghĩa sẵn:
// - const unsigned char cnn_deep_model[]
// - const unsigned int cnn_deep_model_len
```

### **Bước 2: Load model vào TensorFlow Lite Micro**

```cpp
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Khai báo biến global
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena - bộ nhớ làm việc cho model
constexpr int kTensorArenaSize = 60 * 1024;  // 60 KB
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  
  // 1. Load model từ C array
  model = tflite::GetModel(cnn_deep_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version không khớp!");
    return;
  }
  
  // 2. Tạo ops resolver
  static tflite::AllOpsResolver resolver;
  
  // 3. Tạo interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // 4. Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }
  
  // 5. Lấy input và output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Model loaded successfully!");
  Serial.printf("Input shape: [%d, %d, %d]\n", 
                input->dims->data[0], 
                input->dims->data[1], 
                input->dims->data[2]);
  Serial.printf("Output shape: [%d, %d]\n", 
                output->dims->data[0], 
                output->dims->data[1]);
}
```

### **Bước 3: Inference**

```cpp
void loop() {
  // 1. Đọc dữ liệu từ IMU (ví dụ: MPU6050)
  float sensor_data[561];  // 561 features
  read_sensor_data(sensor_data);
  
  // 2. Copy dữ liệu vào input tensor
  for (int i = 0; i < 561; i++) {
    input->data.f[i] = sensor_data[i];
  }
  
  // 3. Chạy inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }
  
  // 4. Đọc kết quả
  // Output có 6 classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, 
  //                      SITTING, STANDING, LAYING
  int predicted_class = 0;
  float max_prob = output->data.f[0];
  
  for (int i = 1; i < 6; i++) {
    if (output->data.f[i] > max_prob) {
      max_prob = output->data.f[i];
      predicted_class = i;
    }
  }
  
  // 5. In kết quả
  const char* activities[] = {
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
  };
  
  Serial.printf("Predicted: %s (%.2f%%)\n", 
                activities[predicted_class], 
                max_prob * 100);
  
  delay(1000);
}
```

---

## 🔧 YÊU CẦU HỆ THỐNG

### **Phần cứng:**
- **ESP32** với ít nhất:
  - 4 MB Flash (cho CNN Deep)
  - 520 KB SRAM
  - IMU sensor (MPU6050, MPU9250, hoặc tương tự)

### **Thư viện:**
- **TensorFlow Lite for Microcontrollers**
  ```bash
  # Cài đặt qua Arduino Library Manager
  # Tìm: "TensorFlowLite_ESP32"
  ```

### **Cấu hình Arduino IDE:**
- Board: ESP32 Dev Module
- Flash Size: 4MB (32Mb)
- Partition Scheme: Default 4MB with spiffs
- Upload Speed: 921600

---

## 📊 SO SÁNH HIỆU SUẤT

### **Kích thước vs Accuracy:**

```
CNN Simple:     287 KB  →  95.89% ✅ (Accuracy cao nhất)
CNN Deep:        73 KB  →  92.06% ⭐ (Trade-off tốt nhất)
CNN Attention:   55 KB  →  86.83% 
Depthwise CNN:   61 KB  →  81.71%
```

### **Inference Time (ước tính trên ESP32 @ 240 MHz):**

| Model | Parameters | Inference Time |
|-------|------------|----------------|
| CNN Simple | 283,718 | ~150-200 ms |
| CNN Deep | 53,382 | ~50-80 ms ⭐ |
| Depthwise CNN | 29,520 | ~30-50 ms |
| CNN Attention | 31,814 | ~60-90 ms |

---

## ⚠️ LƯU Ý QUAN TRỌNG

### **1. Quantization:**
- Tất cả models đã được quantize sang **int8**
- Accuracy có thể giảm **1-3%** so với float32
- Cần test lại accuracy trên ESP32 với dữ liệu thực

### **2. Input preprocessing:**
- Models yêu cầu input đã được **normalized** (StandardScaler)
- Cần lưu scaler parameters và áp dụng trên ESP32
- File scaler: `models/scaler.pkl`

### **3. Memory management:**
- Tensor arena size cần điều chỉnh tùy model
- CNN Deep: ~60 KB
- CNN Simple: ~100 KB
- Nếu thiếu memory, giảm tensor_arena_size hoặc dùng model nhỏ hơn

### **4. LSTM models:**
- LSTM và CNN-LSTM **KHÔNG thể** chuyển đổi sang TFLite chuẩn
- Nếu cần dùng, phải enable SELECT_TF_OPS (tăng kích thước lên ~500 KB)
- **Không khuyến nghị** cho ESP32

---

## 🎉 KẾT LUẬN

### **Model được khuyến nghị:**
→ **CNN Deep** (73 KB, 92.06% accuracy)

### **Lý do:**
1. ✅ Kích thước nhỏ (73 KB) - phù hợp ESP32
2. ✅ Accuracy cao (92.06%) - chỉ kém CNN Simple 3.83%
3. ✅ Inference nhanh (~50-80 ms)
4. ✅ Trade-off tốt nhất giữa size và performance

### **Các bước tiếp theo:**
1. Upload `cnn_deep_model.h` lên ESP32
2. Implement code inference theo hướng dẫn trên
3. Test với dữ liệu thực từ IMU
4. Fine-tune preprocessing và threshold nếu cần

---

**Tác giả:** AI Training System  
**Ngày:** 2026-01-14  
**Version:** 1.0

