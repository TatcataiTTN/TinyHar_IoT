# BÁO CÁO SO SÁNH CÁC MODELS NHẬN DIỆN HOẠT ĐỘNG (HAR)

## 📊 TỔNG QUAN KẾT QUẢ TRAINING

Ngày tạo: 2026-01-14
Tổng số models: 6
Dataset: UCI HAR Dataset (7,352 training samples, 2,947 test samples)

---

## 📈 BẢNG KẾT QUẢ CHI TIẾT

| STT | Model | Độ Chính Xác | Loss | Số Parameters | Kích Thước | Thời Gian Training |
|-----|-------|--------------|------|---------------|------------|-------------------|
| 1 | CNN Đơn Giản | **95.89%** | 0.1462 | 283,718 | 1.08 MB | 42.4s |
| 2 | CNN Sâu | **92.06%** | 0.2806 | 53,382 | 0.20 MB | 170.9s |
| 3 | LSTM | 82.97% | 0.5092 | 31,814 | 0.12 MB | 1,166.1s |
| 4 | CNN-LSTM | **89.18%** | 0.2821 | 41,638 | 0.16 MB | 261.5s |
| 5 | Depthwise CNN | 81.71% | 0.4827 | 29,520 | 0.11 MB | 137.8s |
| 6 | CNN Attention | **86.83%** | 0.4635 | 31,814 | 0.12 MB | 221.2s |

---

## 🥇 PHÂN TÍCH CHI TIẾT

### 1. MODEL CÓ HIỆU SUẤT TỐT NHẤT: CNN ĐƠN GIẢN

**Kết quả:**
- Độ chính xác: **95.89%** (cao nhất)
- Test loss: 0.1462 (thấp nhất - tốt nhất)
- Thời gian training: 42.4 giây (nhanh thứ 2)

**Tại sao CNN Đơn Giản hoạt động tốt nhất?**

1. **Kiến trúc phù hợp với dữ liệu:**
   - UCI HAR dataset có 561 features đã được xử lý sẵn (time-domain và frequency-domain)
   - CNN đơn giản với các conv layers có thể học được các patterns từ features này rất hiệu quả
   - Không cần kiến trúc quá phức tạp vì features đã được engineering tốt

2. **Số lượng parameters hợp lý:**
   - 283,718 parameters - đủ lớn để học được các patterns phức tạp
   - Không quá lớn nên tránh được overfitting
   - Validation accuracy cao (99.11%) cho thấy model generalize tốt

3. **Training ổn định:**
   - Loss giảm đều đặn qua các epochs
   - Không có dấu hiệu overfitting nghiêm trọng
   - Early stopping và ReduceLROnPlateau giúp tối ưu hóa tốt

**Nhược điểm:**
- Kích thước lớn nhất (1.08 MB) - có thể khó deploy lên ESP32 với bộ nhớ hạn chế
- Cần quantization để giảm kích thước xuống còn ~270 KB

---

### 2. MODEL HIỆU QUẢ NHẤT: CNN SÂU

**Kết quả:**
- Độ chính xác: **92.06%** (cao thứ 2)
- Kích thước: **0.20 MB** (nhỏ thứ 3)
- Tỷ lệ accuracy/size: **460.3 %/MB** (tốt nhất)

**Tại sao CNN Sâu là model hiệu quả nhất?**

1. **Trade-off tuyệt vời:**
   - Accuracy chỉ kém CNN Đơn Giản 3.83%
   - Nhưng kích thước nhỏ hơn **5.4 lần** (1.08 MB vs 0.20 MB)
   - Chỉ có 53,382 parameters - rất phù hợp cho embedded systems

2. **Kiến trúc tối ưu:**
   - Sử dụng nhiều conv layers với BatchNormalization
   - Giảm số filters ở mỗi layer để giảm parameters
   - Vẫn đủ sâu để học được features phức tạp

3. **Phù hợp cho ESP32:**
   - Kích thước 0.20 MB có thể giảm xuống ~50 KB sau quantization
   - Inference time nhanh do ít parameters
   - Độ chính xác 92% vẫn rất tốt cho ứng dụng thực tế

**Khuyến nghị:** ⭐ **ĐÂY LÀ MODEL TỐT NHẤT ĐỂ DEPLOY LÊN ESP32**

---

### 3. PHÂN TÍCH CÁC MODELS KHÁC

#### 3.1. LSTM (82.97%)

**Tại sao LSTM hoạt động kém?**

1. **Không phù hợp với dữ liệu:**
   - UCI HAR features đã được aggregate (mean, std, max, min) từ time windows
   - Không còn temporal dependencies rõ ràng
   - LSTM cần raw time-series data để phát huy tối đa

2. **Training chậm:**
   - 1,166 giây (19.4 phút) - chậm nhất
   - LSTM có nhiều operations tuần tự, khó parallelize
   - Không hiệu quả cho training

3. **Overfitting:**
   - Validation accuracy (89.67%) cao hơn test accuracy (82.97%)
   - Chênh lệch 6.7% cho thấy model không generalize tốt

**Kết luận:** LSTM không phải lựa chọn tốt cho UCI HAR dataset với features đã được xử lý

---

#### 3.2. CNN-LSTM (89.18%)

**Phân tích:**

1. **Kết hợp hai kiến trúc:**
   - CNN extract spatial features
   - LSTM học temporal dependencies
   - Accuracy 89.18% - khá tốt nhưng không xuất sắc

2. **Vấn đề:**
   - Kích thước 0.16 MB - trung bình
   - Training time 261.5s - khá lâu
   - Phức tạp hơn nhưng không cải thiện nhiều so với CNN thuần

3. **Trade-off không tốt:**
   - Accuracy chỉ cao hơn CNN Sâu 2.88%
   - Nhưng training chậm hơn 1.5 lần
   - Kích thước lớn hơn 1.25 lần

**Kết luận:** CNN-LSTM không mang lại lợi ích đáng kể so với CNN thuần

---

#### 3.3. Depthwise CNN (81.71%)

**Tại sao Depthwise CNN có accuracy thấp nhất?**

1. **Kiến trúc quá đơn giản:**
   - Chỉ có 29,520 parameters - ít nhất trong tất cả models
   - Depthwise separable convolutions giảm parameters quá nhiều
   - Không đủ capacity để học các patterns phức tạp

2. **Cải thiện đã thực hiện:**
   - Đã thêm BatchNormalization
   - Đã thêm block thứ 3 với 128 filters
   - Đã tăng Dense layer từ 64 → 128 units
   - Nhưng vẫn chưa đủ để đạt accuracy cao

3. **Ưu điểm:**
   - Kích thước nhỏ nhất: 0.11 MB (~28 KB sau quantization)
   - Training nhanh: 137.8s
   - Rất phù hợp cho devices có bộ nhớ cực kỳ hạn chế

**Kết luận:** Depthwise CNN phù hợp khi cần model cực kỳ nhỏ gọn, chấp nhận accuracy thấp hơn

---

#### 3.4. CNN Attention (86.83%)

**Phân tích:**

1. **Attention mechanism:**
   - Giúp model focus vào các features quan trọng
   - Accuracy 86.83% - khá tốt
   - Kích thước 0.12 MB - nhỏ

2. **Vấn đề:**
   - Không cải thiện nhiều so với CNN thuần
   - Training time 221.2s - khá lâu
   - Attention layer tăng complexity nhưng không tăng accuracy đáng kể

3. **Nguyên nhân:**
   - UCI HAR features đã được xử lý tốt
   - Không cần attention để select features
   - Simple CNN đã đủ hiệu quả

**Kết luận:** Attention không mang lại lợi ích rõ rệt cho bài toán này

---

## 🎯 KHUYẾN NGHỊ DEPLOY LÊN ESP32

### Lựa chọn 1: CNN SÂU (KHUYẾN NGHỊ) ⭐⭐⭐⭐⭐

**Lý do:**
- ✅ Accuracy cao: 92.06%
- ✅ Kích thước nhỏ: 0.20 MB → ~50 KB sau quantization
- ✅ Trade-off tốt nhất giữa accuracy và size
- ✅ Inference nhanh do ít parameters
- ✅ Phù hợp với ESP32 (520 KB SRAM, 4 MB Flash)

**Ứng dụng:** Phù hợp cho hầu hết các ứng dụng HAR trên ESP32

---

### Lựa chọn 2: CNN ĐƠN GIẢN (Nếu cần accuracy cao nhất) ⭐⭐⭐⭐

**Lý do:**
- ✅ Accuracy cao nhất: 95.89%
- ⚠️ Kích thước lớn: 1.08 MB → ~270 KB sau quantization
- ⚠️ Cần ESP32 với Flash lớn (4 MB trở lên)

**Ứng dụng:** Khi accuracy là ưu tiên số 1 và ESP32 có đủ bộ nhớ

---

### Lựa chọn 3: DEPTHWISE CNN (Nếu bộ nhớ cực kỳ hạn chế) ⭐⭐⭐

**Lý do:**
- ✅ Kích thước nhỏ nhất: 0.11 MB → ~28 KB sau quantization
- ⚠️ Accuracy thấp: 81.71%
- ✅ Phù hợp cho ESP32 với Flash nhỏ (2 MB)

**Ứng dụng:** Khi bộ nhớ là giới hạn chính, chấp nhận accuracy thấp hơn

---

## 📊 SO SÁNH KÍCH THƯỚC SAU QUANTIZATION (DỰ KIẾN)

| Model | Kích Thước Gốc | Sau Quantization (int8) | Giảm |
|-------|----------------|-------------------------|------|
| CNN Đơn Giản | 1.08 MB | ~270 KB | 75% |
| CNN Sâu | 0.20 MB | ~50 KB | 75% |
| LSTM | 0.12 MB | ~30 KB | 75% |
| CNN-LSTM | 0.16 MB | ~40 KB | 75% |
| Depthwise CNN | 0.11 MB | ~28 KB | 75% |
| CNN Attention | 0.12 MB | ~30 KB | 75% |

---

## 🎓 KẾT LUẬN TỔNG QUAN

### Models Xuất Sắc (≥90%):
1. **CNN Đơn Giản: 95.89%** - Tốt nhất về accuracy
2. **CNN Sâu: 92.06%** - Tốt nhất về trade-off

### Models Tốt (85-90%):
3. **CNN-LSTM: 89.18%** - Tốt nhưng phức tạp không cần thiết
4. **CNN Attention: 86.83%** - Attention không mang lại lợi ích rõ rệt

### Models Cần Cải Thiện (<85%):
5. **LSTM: 82.97%** - Không phù hợp với dữ liệu đã xử lý
6. **Depthwise CNN: 81.71%** - Quá đơn giản, cần thêm capacity

---

## 💡 KHUYẾN NGHỊ CUỐI CÙNG

**Cho ESP32 với 4 MB Flash:**
→ Sử dụng **CNN SÂU** (92.06% accuracy, ~50 KB sau quantization)

**Cho ESP32 với 2 MB Flash:**
→ Sử dụng **Depthwise CNN** (81.71% accuracy, ~28 KB sau quantization)

**Cho ứng dụng cần accuracy cao nhất:**
→ Sử dụng **CNN Đơn Giản** (95.89% accuracy, ~270 KB sau quantization)

---

**Tác giả:** AI Training System
**Ngày:** 2026-01-14
**Version:** 1.0

