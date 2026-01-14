# 📝 CHANGELOG

Tất cả các thay đổi quan trọng của project TinyHAR sẽ được ghi lại trong file này.

Format dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và project tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-14

### 🎉 Release Chính Thức

Phiên bản đầu tiên hoàn chỉnh của TinyHAR với đầy đủ tính năng training, evaluation, và deployment.

### ✨ Added

#### Machine Learning
- **6 model architectures** đã được implement và train:
  - CNN Simple (95.89% accuracy)
  - CNN Deep (92.06% accuracy) - Khuyến nghị cho ESP32
  - LSTM (82.97% accuracy)
  - CNN-LSTM Hybrid (89.18% accuracy)
  - Depthwise Separable CNN (81.71% accuracy)
  - CNN with Attention (86.83% accuracy)

#### Training & Evaluation
- Script `train_all_models.py` để train tất cả models
- Script `train_individual_models.py` để train models cụ thể
- Script `evaluate_all_models.py` để đánh giá models
- Automatic model checkpointing và early stopping
- Training history visualization
- Comprehensive evaluation metrics (accuracy, loss, confusion matrix)

#### TensorFlow Lite Conversion
- Script `convert_models_to_tflite.py` để chuyển đổi models
- Float32 và int8 quantized versions
- C header files generation cho ESP32
- 4/6 models đã chuyển đổi thành công:
  - CNN Simple: 287 KB (giảm 91.5%)
  - CNN Deep: 73 KB (giảm 89.6%)
  - Depthwise CNN: 61 KB (giảm 86.7%)
  - CNN Attention: 55 KB (giảm 87.6%)

#### Visualization & Analysis
- Script `create_visualizations.py` để tạo biểu đồ so sánh
- Model comparison plots (accuracy, size, trade-off, training time)
- Comprehensive comparison report (`model_comparison_report.md`)
- Training history plots

#### Documentation
- README.md toàn diện bằng tiếng Việt
- TFLite Conversion Guide với code examples
- Model Comparison Report chi tiết
- Literature Review về HAR
- Technical Protocols documentation
- Dataset Comparison guide

#### Project Structure
- Cấu trúc thư mục rõ ràng và có tổ chức
- Archive/ folder cho old files
- Proper separation of concerns (src/, models/, docs/, etc.)

### 🔧 Changed

#### Reorganization
- Di chuyển old scripts vào `Archive/old_scripts/`
- Di chuyển old documentation vào `Archive/old_docs/`
- Di chuyển test outputs vào `Archive/test_outputs/`
- Di chuyển training logs vào `Archive/old_training/`
- Cập nhật README.md với cấu trúc mới

#### Improvements
- Tối ưu training pipeline
- Cải thiện data preprocessing
- Tăng cường error handling
- Thêm progress indicators
- Cải thiện logging

### 📊 Results

#### Best Models
1. **CNN Simple**: 95.89% accuracy, 287 KB (quantized)
2. **CNN Deep**: 92.06% accuracy, 73 KB (quantized) ⭐ KHUYẾN NGHỊ
3. **CNN-LSTM**: 89.18% accuracy (không thể convert sang TFLite)

#### Performance Metrics
- Training time: 42s - 1,166s tùy model
- Model size reduction: 86.7% - 91.5% sau quantization
- Inference time (ước tính): 30-200ms trên ESP32

### 🐛 Fixed
- Sửa lỗi data loading với UCI HAR dataset
- Sửa lỗi memory leak trong training loop
- Sửa lỗi quantization cho một số models
- Sửa lỗi visualization với matplotlib backend

### ⚠️ Known Issues
- LSTM và CNN-LSTM không thể convert sang TFLite standard
  - Workaround: Sử dụng SELECT_TF_OPS (tăng kích thước đáng kể)
- Quantization có thể làm giảm accuracy 1-3%
- ESP32 firmware chưa hoàn thiện (đang phát triển)

### 📦 Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy 1.19+
- Pandas 1.2+
- Matplotlib 3.3+
- Scikit-learn 0.24+

---

## [0.9.0] - 2026-01-13

### 🚧 Pre-Release

#### Added
- Initial project structure
- Basic model implementations
- UCI HAR dataset integration
- Training scripts (experimental)

#### Changed
- Multiple iterations on model architectures
- Experimented with different preprocessing techniques

---

## [0.5.0] - 2026-01-10

### 🔬 Experimental Phase

#### Added
- Proof of concept implementations
- Literature review
- Dataset research
- Initial documentation

---

## Roadmap

### [1.1.0] - Planned

#### ESP32 Firmware
- [ ] Complete ESP32 firmware implementation
- [ ] IMU sensor integration (MPU6050)
- [ ] Real-time inference on device
- [ ] WiFi HTTP API server
- [ ] Web dashboard for monitoring

#### Improvements
- [ ] Add more model architectures
- [ ] Implement model pruning
- [ ] Add quantization-aware training
- [ ] Optimize inference speed
- [ ] Add power management

#### Documentation
- [ ] ESP32 setup guide
- [ ] Hardware assembly guide
- [ ] API documentation
- [ ] Troubleshooting guide

### [1.2.0] - Future

#### Features
- [ ] OTA (Over-The-Air) updates
- [ ] Mobile app for monitoring
- [ ] Support for multiple IMU sensors
- [ ] Cloud integration (optional)
- [ ] Data logging to SD card
- [ ] Battery monitoring

#### Advanced ML
- [ ] Online learning capabilities
- [ ] Transfer learning support
- [ ] Multi-task learning
- [ ] Federated learning

---

## Contributing

Để đóng góp vào project:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

Mọi đóng góp đều được ghi nhận trong CHANGELOG.

---

## Links

- **Repository**: https://github.com/TatcataiTTN/TinyHar_IoT
- **Issues**: https://github.com/TatcataiTTN/TinyHar_IoT/issues
- **Documentation**: [docs/](docs/)

---

**Maintained by TinyHAR Team**

