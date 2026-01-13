# 📚 Tài Liệu Tham Khảo - TinyHAR Project

**Mục đích:** Tổng hợp các bài báo khoa học quan trọng cho dự án TinyHAR  
**Cập nhật:** Tháng 1/2026

---

## 📖 Danh Sách Papers Chính (15 Papers Quan Trọng Nhất)

### 1️⃣ HAR Trên Vi Điều Khiển

#### Paper 1: Human Activity Recognition on Microcontrollers (ACM 2022)
- **Link:** https://dl.acm.org/doi/full/10.1145/3542819
- **DOI:** 10.1145/3542819
- **Tóm tắt:** Nghiên cứu về khả năng chạy deep learning HAR trên vi điều khiển sử dụng quantization. Chứng minh tính khả thi của việc triển khai mô hình học sâu trên ESP32.
- **Ứng dụng:** Nền tảng cho kiến trúc TinyHAR, chiến lược quantization
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 2: TinierHAR - Ultra-Lightweight Models (arXiv 2025)
- **Link:** https://arxiv.org/html/2507.07949v1
- **arXiv:** 2507.07949v1
- **Tóm tắt:** Mô hình HAR cực nhẹ đạt độ chính xác cao với tài nguyên tối thiểu. Tối ưu hóa kiến trúc mạng cho thiết bị nhúng.
- **Ứng dụng:** Thiết kế kiến trúc model nhỏ gọn cho ESP32
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 3: Towards Generalizable HAR (arXiv 2025)
- **Link:** https://arxiv.org/html/2508.12213v1
- **arXiv:** 2508.12213v1
- **Tóm tắt:** Khảo sát về thách thức generalization trong HAR. Xử lý vấn đề cross-domain adaptation và độ bền vững của model.
- **Ứng dụng:** Cải thiện khả năng tổng quát hóa của model
- **Độ ưu tiên:** ⭐⭐⭐⭐

---

### 2️⃣ Quantization và Tối Ưu Hóa Model

#### Paper 4: Quantized Neural Networks Survey (arXiv 2025)
- **Link:** https://arxiv.org/html/2508.15008v1
- **arXiv:** 2508.15008v1
- **Tóm tắt:** Khảo sát toàn diện về quantization cho vi điều khiển. Bao gồm PTQ (Post-Training Quantization) và QAT (Quantization-Aware Training).
- **Ứng dụng:** Chuyển đổi model 32-bit → 8-bit cho ESP32
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 5: Quantization and Deployment on Microcontrollers (MDPI 2021)
- **Link:** https://www.mdpi.com/1424-8220/21/9/2984
- **DOI:** 10.3390/s21092984
- **Tóm tắt:** Pipeline thực tế từ training đến deployment trên embedded. Phân tích tiêu thụ năng lượng chi tiết.
- **Ứng dụng:** Quy trình hoàn chỉnh training → quantization → deployment
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 6: Efficient HAR Using Quantization (Nature 2025)
- **Link:** https://www.nature.com/articles/s41598-025-98571-2
- **Tóm tắt:** Full integer quantization cho vi điều khiển. Giảm 78% kích thước model với độ chính xác chỉ giảm <2%.
- **Ứng dụng:** Mục tiêu tối ưu hóa cho TinyHAR
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 7: Emerging Trends in TinyML (ScienceDirect 2025)
- **Link:** https://www.sciencedirect.com/science/article/pii/S0925231225014183
- **Tóm tắt:** Tổng quan về các phương pháp tối ưu TinyML: quantization, pruning, clustering, knowledge distillation.
- **Ứng dụng:** Chiến lược tối ưu đa phương pháp
- **Độ ưu tiên:** ⭐⭐⭐⭐

---

### 3️⃣ Sensor Fusion và Kiến Trúc Deep Learning

#### Paper 8: Multi-Channel Hybrid Deep Learning (ScienceDirect 2024)
- **Link:** https://www.sciencedirect.com/science/article/pii/S1110016824000425
- **Tóm tắt:** Fusion dữ liệu từ accelerometer, gyroscope, magnetometer. Kiến trúc CNN đa kênh cho sensor fusion.
- **Ứng dụng:** Tích hợp dữ liệu từ GY85 (9-DOF IMU)
- **Độ ưu tiên:** ⭐⭐⭐⭐

#### Paper 9: Sensor Data Acquisition and Fusion (MDPI 2019)
- **Link:** https://www.mdpi.com/1424-8220/19/7/1716
- **DOI:** 10.3390/s19071716
- **Tóm tắt:** So sánh hệ thống các sensor modalities. Chiến lược lựa chọn và fusion sensor tối ưu.
- **Ứng dụng:** Lựa chọn sensor và thuật toán fusion
- **Độ ưu tiên:** ⭐⭐⭐⭐

#### Paper 10: WISNet Deep Neural Network (ScienceDirect 2024)
- **Link:** https://www.sciencedirect.com/science/article/pii/S0957417424018669
- **Tóm tắt:** Đánh giá trên WISDM, UCI-HAR, OPPORTUNITY, PAMAP2. DNN tối ưu cho wearable sensors.
- **Ứng dụng:** Benchmark so sánh trên nhiều datasets
- **Độ ưu tiên:** ⭐⭐⭐⭐

---

### 4️⃣ TensorFlow Lite Micro và ESP32

#### Paper 11: TinyML with ESP32 Tutorial (TeachMeMicro 2024)
- **Link:** https://www.teachmemicro.com/tinyml-with-esp32-tutorial/
- **Tóm tắt:** Hướng dẫn chi tiết TFLite Micro trên ESP32. Bao gồm model conversion, deployment, sensor integration.
- **Ứng dụng:** Tài liệu tham khảo implementation từng bước
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 12: Iris Classification on ESP32 (Medium 2024)
- **Link:** https://medium.com/@eduardo.bl/iris-dataset-classification-model-in-esp32-tensorflow-lite-micro-http-server-5aa5a66f7543
- **Tóm tắt:** Ví dụ thực tế với HTTP server integration. Code hoàn chỉnh cho ESP32 + TFLite + Web Server.
- **Ứng dụng:** Template cho ESP32 + TFLite + WiFi AP
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

#### Paper 13: Porting TensorFlow to Embedded (Medium 2024)
- **Link:** https://medium.com/@johnos3747/embedded-ai-systems-part-11-ebd18aceb4cf
- **Tóm tắt:** Quy trình chuyển đổi TFLite model → C header file. Scripts tự động hóa conversion pipeline.
- **Ứng dụng:** Tự động hóa quá trình convert model
- **Độ ưu tiên:** ⭐⭐⭐⭐

#### Paper 14: Deploying CNNs on Microcontrollers (Medium 2024)
- **Link:** https://nathanbaileyw.medium.com/deploying-convolutional-neural-networks-on-microcontrollers-a-tinyml-blog-5f9b4fa37864
- **Tóm tắt:** Workflow hoàn chỉnh: Keras → TFLite → C header. Tối ưu hóa quantization và compression.
- **Ứng dụng:** Hướng dẫn deployment end-to-end
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

---

### 5️⃣ Datasets và Benchmarks

#### Paper 15: UCI HAR Dataset Paper (ESANN 2013)
- **Citation:** Anguita et al., "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
- **Link:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Tóm tắt:** Dataset chuẩn công nghiệp cho HAR. 30 người, 6 hoạt động, 561 features, 50Hz sampling.
- **Ứng dụng:** Dataset chính cho training và evaluation
- **Độ ưu tiên:** ⭐⭐⭐⭐⭐

---

## 📥 Hướng Dẫn Tải Papers

### Cách 1: Tải Thủ Công
1. Click vào link của từng paper
2. Download PDF và lưu vào `references/papers/`
3. Đặt tên file theo format: `[Số]_[Tên_Ngắn].pdf`
   - Ví dụ: `01_HAR_Microcontrollers_ACM2022.pdf`

### Cách 2: Sử Dụng Script (TODO)
```bash
# Script tự động tải papers (sẽ được tạo sau)
python scripts/download_papers.py
```

---

## 📊 Phân Loại Theo Chủ Đề

| Chủ Đề | Papers | Độ Ưu Tiên |
|---------|--------|------------|
| **HAR trên MCU** | #1, #2, #3 | Cao nhất |
| **Quantization** | #4, #5, #6, #7 | Cao nhất |
| **Sensor Fusion** | #8, #9, #10 | Cao |
| **TFLite + ESP32** | #11, #12, #13, #14 | Cao nhất |
| **Datasets** | #15 | Cao nhất |

---

## 🎯 Papers Bắt Buộc Phải Đọc (Top 5)

1. **Paper #4** - Quantization Survey (Nền tảng lý thuyết)
2. **Paper #11** - TinyML ESP32 Tutorial (Implementation guide)
3. **Paper #14** - CNN Deployment (End-to-end workflow)
4. **Paper #1** - HAR on MCU (Proof of concept)
5. **Paper #15** - UCI HAR Dataset (Data understanding)

---

## 📚 Tài Liệu Bổ Sung

### Official Documentation
- **TensorFlow Lite Micro:** https://github.com/tensorflow/tflite-micro
- **ESP32 Technical Reference:** https://www.espressif.com/en/products/socs/esp32
- **Arduino ESP32:** https://github.com/espressif/arduino-esp32

### Datasheets
- **ADXL345 (Accelerometer):** https://www.analog.com/en/products/adxl345.html
- **ITG3200 (Gyroscope):** https://www.invensense.com/products/motion-tracking/3-axis/itg-3200/
- **HMC5883L (Magnetometer):** https://www.honeywell.com/

### Tutorials
- **GY-85 IMU Tutorial:** https://www.instructables.com/Tutorial-to-Interface-GY-85-IMU-9DOF-Sensor-With-A/
- **ESP32 WiFi AP:** https://www.dfrobot.com/blog-851.html

---

## 📝 Ghi Chú

- Tất cả papers đều được chọn lọc từ `docs/LITERATURE_REVIEW.md`
- Ưu tiên papers từ 2020-2026 (nghiên cứu mới nhất)
- Focus vào practical implementation hơn là lý thuyết thuần túy
- Papers có code/tutorial được ưu tiên cao hơn

---

**Trạng thái:** ✅ Danh sách hoàn chỉnh  
**Tổng số papers:** 15 papers chính + nhiều tài liệu bổ sung  
**Cập nhật:** Tháng 1/2026

