# 📦 Archive - Lưu Trữ Files Cũ

Thư mục này chứa các file cũ, script thử nghiệm, và documentation đã được thay thế bởi phiên bản mới.

**Lưu ý:** Các file trong Archive không còn được sử dụng trong workflow chính của project, nhưng được giữ lại để tham khảo lịch sử phát triển.

---

## 📁 Cấu Trúc

### `old_scripts/` - Scripts Cũ

Chứa các training scripts và test scripts đã bị thay thế:

**Training Scripts:**
- `full_training.py` - Script training đầu tiên (đã thay thế bởi `train_all_models.py`)
- `final_training_all_6_models.py` - Phiên bản training cũ
- `run_training.py` - Wrapper script cũ
- `launch_parallel_training.py` - Thử nghiệm parallel training

**Test Scripts:**
- `quick_test.py` - Quick test script
- `monitor_training.py` - Training monitoring tool
- `view_final_results.py` - Results viewer

**Notebooks:**
- `TinyHAR_Training_Colab.ipynb` - Google Colab notebook thử nghiệm

**Lý do archive:** Đã được thay thế bởi `src/train_all_models.py` và `train_individual_models.py` với code tốt hơn, có tổ chức hơn.

---

### `old_docs/` - Documentation Cũ

Chứa các file documentation đã lỗi thời:

**Files:**
- `README_old.md` - README phiên bản cũ
- `INSTALLATION_COMPLETE.md` - Hướng dẫn cài đặt cũ
- `TRAINING_ANALYSIS.md` - Phân tích training cũ
- `SUMMARY.md` - Tóm tắt project cũ

**Đã xóa khỏi git:**
- `FINAL_RESULTS_SUMMARY.md`
- `PROJECT_COMPLETE.md`
- `PROJECT_REBUILD_SUMMARY.md`
- `README_REBUILD.md`
- `TRAINING_TROUBLESHOOTING.md`
- `USAGE_GUIDE.md`

**Lý do archive:** Đã được thay thế bởi:
- `README.md` mới (toàn diện, bằng tiếng Việt)
- `CHANGELOG.md` (lịch sử project)
- `models/model_comparison_report.md` (phân tích chi tiết)
- `models/TFLITE_CONVERSION_GUIDE.md` (hướng dẫn deployment)

---

### `test_outputs/` - Test Outputs

Chứa các file output từ quá trình test và debug:

**Files:**
- `*.txt` - Các file output text
- `quick_test_output.txt`
- `quick_test_output_new.txt`
- `quick_test_run.txt`
- `final_training_output.txt`
- `full_training_output.txt`
- `Miniconda3-latest-MacOSX-arm64.sh` - Installer file

**Lý do archive:** Chỉ là output tạm thời, không cần thiết cho production.

---

### `old_training/` - Training Logs Cũ

Chứa training logs từ các lần training trước:

**Files:**
- `training_logs/` - Thư mục chứa logs
  - `cnn_simple_training.log`
  - `cnn_deep_training.log`
  - `cnn_lstm_training.log`
  - `lstm_training.log`
  - `depthwise_cnn_training.log`
  - `cnn_attention_training.log`

**Lý do archive:** Logs cũ, kết quả cuối cùng đã được lưu trong `models/training_results_*.json`.

---

## 🔄 Migration Guide

Nếu bạn cần tham khảo code cũ:

### Training Scripts

**Cũ:**
```bash
python full_training.py
python final_training_all_6_models.py
```

**Mới:**
```bash
cd src
python train_all_models.py
# hoặc
python train_individual_models.py --models cnn_deep cnn_simple
```

### Documentation

**Cũ:**
- `INSTALLATION_COMPLETE.md`
- `TRAINING_ANALYSIS.md`
- `SUMMARY.md`

**Mới:**
- `README.md` - Hướng dẫn đầy đủ
- `CHANGELOG.md` - Lịch sử thay đổi
- `models/model_comparison_report.md` - Phân tích models
- `models/TFLITE_CONVERSION_GUIDE.md` - Hướng dẫn deployment

---

## ⚠️ Lưu Ý

1. **Không sử dụng code trong Archive/** cho production
2. Files trong Archive chỉ để tham khảo lịch sử
3. Nếu cần chức năng từ script cũ, hãy implement lại trong code mới
4. Archive không được maintain và có thể chứa bugs

---

## 🗑️ Cleanup Policy

Files trong Archive sẽ được giữ lại cho đến khi:
- Không còn giá trị tham khảo
- Project đã stable và không cần rollback
- Sau 6 tháng kể từ khi archive

Sau đó có thể xóa hoàn toàn để giảm kích thước repository.

---

## 📊 Statistics

**Tổng số files archived:** ~20+
**Tổng dung lượng:** ~5 MB
**Ngày archive:** 2026-01-14
**Lý do:** Reorganize project structure for production

---

**Nếu có câu hỏi về files trong Archive, vui lòng tạo issue trên GitHub.**

