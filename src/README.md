# 📁 Source Code - TinyHAR Training Pipeline

**Mục đích:** Pipeline hoàn chỉnh để train và deploy HAR model lên ESP32  
**Ngôn ngữ:** Python 3.8+  
**Framework:** TensorFlow 2.15+

---

## 📋 Danh Sách Files

| File | Mô Tả | Chức Năng Chính |
|------|-------|-----------------|
| `data_loader.py` | Load UCI HAR Dataset | `load_uci_har_data()` |
| `preprocessing.py` | Chuẩn hóa và reshape dữ liệu | `preprocess_data()`, `reshape_for_cnn()` |
| `model.py` | Định nghĩa kiến trúc model | `create_har_model()`, `compile_model()` |
| `train.py` | Script training chính | `train_model()` |
| `evaluate.py` | Đánh giá model | `evaluate_model()` |
| `convert_tflite.py` | Convert sang TFLite và C header | `convert_to_tflite()`, `convert_to_c_header()` |

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- tensorflow>=2.15.0
- numpy>=1.24.3
- pandas>=2.0.3
- scikit-learn>=1.3.0
- matplotlib>=3.7.2
- seaborn>=0.12.2

### Bước 2: Tải Dataset

```bash
cd datasets
python ../scripts/download_dataset.py
```

Dataset sẽ được tải về và giải nén vào `datasets/UCI HAR Dataset/`

### Bước 3: Train Model

```bash
python src/train.py
```

**Cấu hình mặc định:**
- Model: `cnn_simple` (nhỏ gọn, ~50KB)
- Epochs: 50
- Batch size: 32
- Learning rate: 0.001

**Outputs:**
- `models/har_model_cnn_simple.h5` - Model đã train
- `models/har_model_best.h5` - Best model (theo val_accuracy)
- `models/scaler.pkl` - Scaler để normalize dữ liệu
- `models/training_history_cnn_simple.png` - Biểu đồ training

**Thời gian:** ~5-10 phút (tùy CPU/GPU)

### Bước 4: Đánh Giá Model

```bash
python src/evaluate.py
```

**Outputs:**
- `models/evaluation_results_cnn_simple.txt` - Báo cáo chi tiết
- `models/confusion_matrix_cnn_simple.png` - Confusion matrix

**Metrics:**
- Accuracy
- Precision, Recall, F1-score (cho từng class)
- Confusion matrix
- Phân tích lỗi

### Bước 5: Convert Sang TFLite

```bash
python src/convert_tflite.py
```

**Outputs:**
- `models/har_model_cnn_simple.tflite` - TFLite model (quantized int8)
- `models/har_model_cnn_simple.h` - C header file cho ESP32

**Kích thước:**
- Model gốc (.h5): ~200-300 KB
- TFLite quantized: ~50-80 KB
- Giảm: ~70-80%

---

## 🔧 Tùy Chỉnh Model

### Thay Đổi Kiến Trúc Model

Sửa file `src/train.py`, dòng 123:

```python
MODEL_TYPE = 'cnn_simple'  # Chọn 1 trong 3:
# - 'cnn_simple': Nhỏ gọn (~50KB), accuracy ~90%
# - 'cnn_deep': Lớn hơn (~100KB), accuracy ~93%
# - 'lstm': Trung bình (~80KB), accuracy ~91%
```

### Thay Đổi Hyperparameters

Sửa file `src/train.py`, dòng 124-126:

```python
EPOCHS = 50           # Số epochs (khuyến nghị: 30-100)
BATCH_SIZE = 32       # Batch size (khuyến nghị: 16-64)
LEARNING_RATE = 0.001 # Learning rate (khuyến nghị: 0.0001-0.01)
```

### Tạo Model Tùy Chỉnh

Thêm function mới vào `src/model.py`:

```python
def create_custom_model(input_shape, num_classes):
    """Model tùy chỉnh của bạn"""
    model = models.Sequential([
        # Thêm layers ở đây
        layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        # ...
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

---

## 📊 Kết Quả Mong Đợi

### Model: CNN Simple

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | ~95% | ~92% | ~90% |
| **Loss** | ~0.15 | ~0.25 | ~0.30 |

### Per-Class Performance (Test Set)

| Activity | Precision | Recall | F1-Score | Samples |
|----------|-----------|--------|----------|---------|
| WALKING | 0.92 | 0.94 | 0.93 | 496 |
| WALKING_UPSTAIRS | 0.88 | 0.86 | 0.87 | 471 |
| WALKING_DOWNSTAIRS | 0.90 | 0.92 | 0.91 | 420 |
| SITTING | 0.89 | 0.87 | 0.88 | 491 |
| STANDING | 0.91 | 0.93 | 0.92 | 532 |
| LAYING | 0.98 | 0.99 | 0.98 | 537 |

**Overall Accuracy:** ~90-92%

---

## 🧪 Test Từng Module

### Test Data Loader

```bash
python src/data_loader.py
```

**Output:** Thông tin dataset, số samples, phân bố classes

### Test Preprocessing

```bash
python src/preprocessing.py
```

**Output:** Shape sau preprocessing, thống kê dữ liệu

### Test Model Architecture

```bash
python src/model.py
```

**Output:** Model summary, số parameters, kích thước ước tính

---

## 🐛 Troubleshooting

### Lỗi: "No module named 'tensorflow'"

```bash
pip install tensorflow>=2.15.0
```

### Lỗi: "Dataset not found"

```bash
python scripts/download_dataset.py
```

### Lỗi: "Out of memory"

Giảm batch size trong `src/train.py`:

```python
BATCH_SIZE = 16  # Hoặc 8
```

### Lỗi: "Model file not found"

Đảm bảo đã train model trước:

```bash
python src/train.py
```

### Training quá chậm

- Giảm số epochs: `EPOCHS = 30`
- Tăng batch size: `BATCH_SIZE = 64`
- Dùng GPU nếu có

---

## 📈 Workflow Hoàn Chỉnh

```
1. Download Dataset
   └─> python scripts/download_dataset.py

2. Train Model
   └─> python src/train.py
       ├─> Load data (data_loader.py)
       ├─> Preprocess (preprocessing.py)
       ├─> Create model (model.py)
       ├─> Train
       └─> Save model (.h5)

3. Evaluate Model
   └─> python src/evaluate.py
       ├─> Load model
       ├─> Test inference
       └─> Generate reports

4. Convert to TFLite
   └─> python src/convert_tflite.py
       ├─> Load .h5 model
       ├─> Quantize (int8)
       ├─> Save .tflite
       └─> Generate .h header

5. Deploy to ESP32
   └─> Copy .h file to firmware/
       └─> Compile and upload
```

---

## 📝 Ghi Chú

- Tất cả scripts có thể chạy độc lập để test
- Mỗi script có `if __name__ == '__main__':` block
- Comments bằng tiếng Việt để dễ hiểu
- Code ngắn gọn, tránh phức tạp
- Tuân theo PEP 8 style guide

---

**Tác giả:** TinyHAR Project Team  
**Cập nhật:** Tháng 1/2026  
**License:** MIT

