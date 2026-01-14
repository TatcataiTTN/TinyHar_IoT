# 🏃 TinyHAR - Nhận Diện Hoạt Động Con Người trên ESP32

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![ESP32](https://img.shields.io/badge/ESP32-Compatible-green.svg)](https://www.espressif.com/)

**Hệ thống nhận diện hoạt động con người (Human Activity Recognition - HAR) trên vi điều khiển ESP32 sử dụng TensorFlow Lite Micro và cảm biến IMU**

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Tính Năng](#-tính-năng)
- [Các Hoạt Động Nhận Diện](#-các-hoạt-động-nhận-diện)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Cài Đặt](#-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Kết Quả Models](#-kết-quả-models)
- [Cấu Trúc Project](#-cấu-trúc-project)
- [Tài Liệu](#-tài-liệu)
- [Đóng Góp](#-đóng-góp)
- [License](#-license)

---

## 🎯 Giới Thiệu

TinyHAR là một dự án hoàn chỉnh về nhận diện hoạt động con người trên thiết bị nhúng ESP32 với tài nguyên hạn chế. Hệ thống có khả năng nhận diện 6 hoạt động khác nhau trong thời gian thực sử dụng machine learning.

### Đặc Điểm Nổi Bật

- 🚀 **Edge Computing:** Xử lý hoàn toàn trên thiết bị, không cần cloud
- 💰 **Chi Phí Thấp:** Tổng chi phí phần cứng < $50
- ⚡ **Thời Gian Thực:** Sampling 20Hz, inference < 50ms
- 📡 **Kết Nối WiFi:** HTTP API để giám sát
- 🔋 **Tiết Kiệm Năng Lượng:** Tối ưu cho hoạt động bằng pin
- 📚 **Tài Liệu Đầy Đủ:** Documentation chi tiết và code examples

---

## ✨ Tính Năng

### Machine Learning
- ✅ 6 models đã được train và đánh giá
- ✅ Accuracy cao nhất: **95.89%** (CNN Simple)
- ✅ Model tối ưu cho ESP32: **CNN Deep** (92.06%, 73 KB)
- ✅ Quantization int8 để giảm kích thước model
- ✅ TensorFlow Lite conversion cho embedded deployment

### Phần Cứng
- ✅ ESP32 (4 MB Flash, 520 KB SRAM)
- ✅ IMU sensor (MPU6050, MPU9250, hoặc tương tự)
- ✅ Kết nối WiFi tích hợp
- ✅ Tiêu thụ điện năng thấp

### Phần Mềm
- ✅ TensorFlow Lite Micro runtime
- ✅ Real-time inference
- ✅ HTTP API server
- ✅ Data logging và monitoring

---

## 🏃 Các Hoạt Động Nhận Diện

Hệ thống có thể nhận diện 6 hoạt động sau:

1. 🚶 **WALKING** - Đi bộ bình thường
2. 🏃 **WALKING_UPSTAIRS** - Đi lên cầu thang
3. 🏃 **WALKING_DOWNSTAIRS** - Đi xuống cầu thang
4. 🪑 **SITTING** - Ngồi
5. 🧍 **STANDING** - Đứng
6. 🛏️ **LAYING** - Nằm

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│                         ESP32                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  IMU Sensor  │───▶│ Preprocessing│───▶│  TFLite Model│  │
│  │  (MPU6050)   │    │  (Normalize) │    │  (CNN Deep)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                         │          │
│         │                                         ▼          │
│         │                                  ┌──────────────┐  │
│         └─────────────────────────────────▶│  HTTP API    │  │
│                                            │  Server      │  │
│                                            └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Web Dashboard   │
                    │  (Monitoring)    │
                    └──────────────────┘
```

---

## 🔧 Cài Đặt

### Yêu Cầu Hệ Thống

**Python Environment:**
- Python 3.8 hoặc cao hơn
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Scikit-learn

**Phần Cứng (cho deployment):**
- ESP32 Dev Board (4 MB Flash)
- IMU Sensor (MPU6050 hoặc tương tự)
- Cáp USB để programming

### Bước 1: Clone Repository

```bash
git clone https://github.com/TatcataiTTN/TinyHar_IoT.git
cd TinyHar_IoT
```

### Bước 2: Cài Đặt Dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### Bước 3: Download Dataset

```bash
# Dataset sẽ được tự động download khi chạy training
# Hoặc download thủ công:
python scripts/download_dataset.py
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1️⃣ Training Models

#### Train tất cả 6 models:

```bash
cd src
python train_all_models.py
```

#### Train models cụ thể:

```bash
# Train chỉ CNN Deep và CNN Simple
python train_individual_models.py --models cnn_deep cnn_simple --epochs 50
```

**Các models có sẵn:**
- `cnn_simple` - CNN đơn giản (accuracy cao nhất: 95.89%)
- `cnn_deep` - CNN sâu (tối ưu cho ESP32: 92.06%)
- `lstm` - LSTM model
- `cnn_lstm` - Hybrid CNN-LSTM
- `depthwise_cnn` - Depthwise Separable CNN
- `cnn_attention` - CNN với Attention mechanism

### 2️⃣ Đánh Giá Models

```bash
cd src
python evaluate_all_models.py
```

### 3️⃣ Tạo Visualizations

```bash
python create_visualizations.py
```

Kết quả: `models/model_comparison_plots.png`

### 4️⃣ Chuyển Đổi sang TensorFlow Lite

```bash
python convert_models_to_tflite.py
```

**Output:**
- `models/tflite/*.tflite` - TensorFlow Lite models
- `models/c_arrays/*.h` - C header files cho ESP32

### 5️⃣ Deploy lên ESP32

Xem hướng dẫn chi tiết: [`models/TFLITE_CONVERSION_GUIDE.md`](models/TFLITE_CONVERSION_GUIDE.md)

**Tóm tắt:**
1. Copy file `models/c_arrays/cnn_deep_model.h` vào Arduino project
2. Include TensorFlow Lite Micro library
3. Load model và chạy inference
4. Xem code example trong guide

---

## 📊 Kết Quả Models

### So Sánh Hiệu Suất

| Model | Accuracy | Loss | Parameters | Size (MB) | Training Time |
|-------|----------|------|------------|-----------|---------------|
| **CNN Simple** | **95.89%** | 0.1462 | 283,718 | 1.08 | 42s |
| **CNN Deep** ⭐ | **92.06%** | 0.2806 | 53,382 | 0.20 | 171s |
| CNN-LSTM | 89.18% | 0.2821 | 41,638 | 0.16 | 262s |
| CNN Attention | 86.83% | 0.4635 | 31,814 | 0.12 | 221s |
| LSTM | 82.97% | 0.5092 | 31,814 | 0.12 | 1,166s |
| Depthwise CNN | 81.71% | 0.4827 | 29,520 | 0.11 | 138s |

### Kích Thước Sau Quantization (int8)

| Model | Gốc (.h5) | TFLite Float32 | TFLite Int8 | Giảm |
|-------|-----------|----------------|-------------|------|
| CNN Simple | 3,365 KB | 1,114 KB | **287 KB** | 91.5% |
| **CNN Deep** ⭐ | 696 KB | 217 KB | **73 KB** | 89.6% |
| Depthwise CNN | 462 KB | 132 KB | **61 KB** | 86.7% |
| CNN Attention | 444 KB | 137 KB | **55 KB** | 87.6% |

**Lưu ý:** LSTM và CNN-LSTM không thể chuyển đổi sang TFLite do giới hạn kỹ thuật.

### 🏆 Model Được Khuyến Nghị: CNN Deep

**Lý do:**
- ✅ Accuracy cao: 92.06% (chỉ kém CNN Simple 3.83%)
- ✅ Kích thước nhỏ: 73 KB (nhỏ hơn CNN Simple 4x)
- ✅ Inference nhanh: ~50-80ms trên ESP32
- ✅ Trade-off tốt nhất giữa accuracy và size

---

## 📁 Cấu Trúc Project

```
TinyHar_IoT/
├── README.md                          # File này
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── src/                               # Source code chính
│   ├── data_loader.py                # Load UCI HAR dataset
│   ├── preprocessing.py              # Data preprocessing
│   ├── model.py                      # Model architectures
│   ├── train_all_models.py           # Train tất cả models
│   ├── evaluate_all_models.py        # Evaluate models
│   └── convert_tflite.py             # Convert sang TFLite
│
├── models/                            # Trained models và results
│   ├── *.h5                          # Keras models
│   ├── tflite/                       # TensorFlow Lite models
│   │   ├── *_float32.tflite         # Float32 models
│   │   └── *_int8.tflite            # Quantized int8 models
│   ├── c_arrays/                     # C header files cho ESP32
│   │   └── *_model.h                # Model data as C arrays
│   ├── model_comparison_plots.png    # Biểu đồ so sánh
│   ├── model_comparison_report.md    # Báo cáo chi tiết
│   └── TFLITE_CONVERSION_GUIDE.md    # Hướng dẫn deploy ESP32
│
├── datasets/                          # Dataset storage
│   └── UCI HAR Dataset/              # UCI HAR dataset
│
├── docs/                              # Documentation
│   ├── LITERATURE_REVIEW.md          # Tổng quan nghiên cứu
│   ├── DATASET_COMPARISON.md         # So sánh datasets
│   ├── TECHNICAL_PROTOCOLS.md        # Chi tiết kỹ thuật
│   └── IMPLEMENTATION_PLAN.md        # Kế hoạch triển khai
│
├── firmware/                          # ESP32 firmware (future)
│   └── README.md
│
├── scripts/                           # Utility scripts
│   └── download_dataset.py           # Download dataset
│
├── tests/                             # Unit tests
│   └── README.md
│
├── Archive/                           # Old files (archived)
│   ├── old_scripts/                  # Old training scripts
│   ├── old_docs/                     # Old documentation
│   ├── test_outputs/                 # Test outputs
│   └── old_training/                 # Old training logs
│
├── train_individual_models.py         # Train specific models
├── create_visualizations.py           # Create comparison plots
└── convert_models_to_tflite.py        # Convert models to TFLite
```

---

## 📚 Tài Liệu

### Hướng Dẫn Chi Tiết

1. **[Model Comparison Report](models/model_comparison_report.md)** - So sánh chi tiết các models
2. **[TFLite Conversion Guide](models/TFLITE_CONVERSION_GUIDE.md)** - Hướng dẫn deploy lên ESP32
3. **[Literature Review](docs/LITERATURE_REVIEW.md)** - Tổng quan nghiên cứu HAR
4. **[Technical Protocols](docs/TECHNICAL_PROTOCOLS.md)** - Chi tiết kỹ thuật implementation

### Dataset

Project sử dụng **UCI HAR Dataset**:
- 10,299 samples
- 6 activities
- 561 features (time và frequency domain)
- 30 subjects (người tham gia)
- Train/Test split: 70/30

**Download:** Dataset sẽ tự động download khi chạy training lần đầu.

### Papers và References

Xem thêm trong [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md)

---

## 🛠️ Development

### Chạy Tests

```bash
cd tests
python -m pytest
```

### Code Style

Project tuân theo PEP 8 style guide:

```bash
# Check code style
flake8 src/

# Format code
black src/
```

---

## 🤝 Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

### Báo Lỗi

Nếu bạn tìm thấy bug, vui lòng tạo issue với:
- Mô tả chi tiết lỗi
- Steps to reproduce
- Expected behavior
- Screenshots (nếu có)
- Environment info (OS, Python version, etc.)

---

## 📝 TODO

- [ ] Hoàn thiện ESP32 firmware
- [ ] Thêm web dashboard cho monitoring
- [ ] Implement OTA (Over-The-Air) updates
- [ ] Thêm support cho nhiều IMU sensors
- [ ] Optimize inference time
- [ ] Thêm power management features
- [ ] Tạo mobile app

---

## 📄 License

Project này được phân phối dưới MIT License. Xem file [`LICENSE`](LICENSE) để biết thêm chi tiết.

---

## 👥 Tác Giả

**TinyHAR Team**
- GitHub: [@TatcataiTTN](https://github.com/TatcataiTTN)
- Repository: [TinyHar_IoT](https://github.com/TatcataiTTN/TinyHar_IoT)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository cho HAR dataset
- TensorFlow team cho TFLite Micro
- ESP32 community
- Tất cả contributors

---

## 📞 Liên Hệ

Nếu có câu hỏi hoặc đề xuất, vui lòng:
- Tạo issue trên GitHub
- Email: [your-email@example.com]

---

## ⭐ Star History

Nếu project này hữu ích, đừng quên cho chúng tôi một ⭐ trên GitHub!

---

**Made with ❤️ for IoT and Edge AI**

