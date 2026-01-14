# Baseline Model Analysis - TinyHAR Project

**Date:** January 2026  
**Status:** ✅ Complete  
**Model:** CNN Simple (Baseline)

---

## Executive Summary

The baseline CNN model has been successfully trained and evaluated on the UCI HAR dataset, achieving **95.83% accuracy** on the test set. This provides an excellent foundation for exploring more advanced architectures.

---

## 1. Current State Analysis

### 1.1 Baseline Model Architecture

**Model Type:** CNN Simple  
**Purpose:** Lightweight model optimized for ESP32 deployment

**Architecture Details:**
```
Layer 1: Conv1D(16 filters, kernel=5, activation='relu')
Layer 2: MaxPooling1D(pool_size=2)
Layer 3: Dropout(0.2)
Layer 4: Conv1D(32 filters, kernel=5, activation='relu')
Layer 5: MaxPooling1D(pool_size=2)
Layer 6: Dropout(0.2)
Layer 7: Flatten()
Layer 8: Dense(64, activation='relu')
Layer 9: Dropout(0.3)
Layer 10: Dense(6, activation='softmax')
```

**Model Statistics:**
- **Total Parameters:** 283,718 (trainable)
- **Model Size:** 1.08 MB (float32)
- **Estimated Size (quantized):** ~270 KB (int8)
- **Target for ESP32:** < 100 KB

### 1.2 Performance Metrics

**Overall Performance:**
- **Test Accuracy:** 95.83%
- **Error Rate:** 4.17% (123/2947 samples)

**Per-Class Performance:**

| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| WALKING | 97.58% | 97.58% | 97.58% | 496 |
| WALKING_UPSTAIRS | 94.87% | 98.09% | 96.45% | 471 |
| WALKING_DOWNSTAIRS | 97.28% | 93.57% | 95.39% | 420 |
| SITTING | 93.70% | 90.84% | 92.24% | 491 |
| STANDING | 92.11% | 94.36% | 93.22% | 532 |
| LAYING | 99.63% | 100.00% | 99.81% | 537 |

**Key Observations:**
1. ✅ Excellent performance on LAYING (99.81% F1)
2. ✅ Strong performance on dynamic activities (WALKING variants: 95-97% F1)
3. ⚠️ Lower performance on static activities (SITTING/STANDING: 92-93% F1)
4. 🎯 Main confusion: SITTING vs STANDING (expected due to similarity)

### 1.3 Dataset Information

**UCI HAR Dataset:**
- **Training Samples:** 7,352
- **Test Samples:** 2,947
- **Features:** 561 (pre-computed from raw sensor data)
- **Classes:** 6 activities
- **Subjects:** 30 people (21 train, 9 test)
- **Sampling Rate:** 50 Hz
- **Window Size:** 2.56 seconds (128 samples)

**Data Distribution:**
- Balanced across classes (each ~15-18% of total)
- Proper train/test split by subjects (no data leakage)

---

## 2. Strengths and Limitations

### 2.1 Strengths
✅ **High Accuracy:** 95.83% is excellent for HAR tasks  
✅ **Lightweight:** Only 283K parameters  
✅ **Fast Training:** Converges quickly  
✅ **Good Generalization:** Strong performance across all activities  
✅ **Production Ready:** Model files exist and are functional

### 2.2 Limitations
⚠️ **Model Size:** 1.08 MB is too large for ESP32 (target: <100KB)  
⚠️ **Static Activity Confusion:** SITTING/STANDING confusion (92-93% F1)  
⚠️ **No Temporal Modeling:** Simple CNN doesn't capture long-term dependencies  
⚠️ **Fixed Architecture:** Not optimized for specific constraints  
⚠️ **Single Model:** No ensemble or alternative architectures tested

---

## 3. Areas for Improvement

### 3.1 Model Architecture
1. **Add Temporal Modeling:** LSTM/GRU layers for sequential patterns
2. **Residual Connections:** Skip connections for better gradient flow
3. **Attention Mechanisms:** Focus on important time steps
4. **Depthwise Separable Convolutions:** Reduce parameters while maintaining accuracy

### 3.2 Model Optimization
1. **Quantization:** INT8 quantization to reduce size by 4x
2. **Pruning:** Remove redundant connections
3. **Knowledge Distillation:** Train smaller model from larger teacher
4. **Neural Architecture Search:** Automated architecture optimization

### 3.3 Training Improvements
1. **Data Augmentation:** Time warping, jittering, rotation
2. **Advanced Regularization:** Mixup, CutMix for time series
3. **Learning Rate Scheduling:** Cosine annealing, warm restarts
4. **Class Balancing:** Focus on confused classes (SITTING/STANDING)

---

## 4. Next Steps

### Phase 2: Research and Model Selection
- [ ] Review recent HAR papers (2023-2026)
- [ ] Identify 2-3 promising architectures
- [ ] Analyze trade-offs (accuracy vs size vs speed)
- [ ] Select models for implementation

### Phase 3: Implementation
- [ ] Implement selected architectures
- [ ] Train and validate each model
- [ ] Compare performance metrics
- [ ] Document results

### Phase 4: Deployment Pipeline
- [ ] Implement comprehensive evaluation framework
- [ ] Convert best model to TFLite
- [ ] Generate C header for ESP32
- [ ] Validate quantized model accuracy
- [ ] Document deployment process

---

**Document Status:** ✅ Complete  
**Next Action:** Begin Phase 2 - Research and Model Selection

