# Model Research and Selection - TinyHAR Project

**Date:** January 2026  
**Status:** ✅ Complete  
**Phase:** 2 - Research and Model Selection

---

## Executive Summary

Based on comprehensive research of recent HAR literature (2024-2025), this document identifies **3 promising architectures** that can improve upon the baseline CNN model while maintaining deployment feasibility on ESP32.

**Recommended Models:**
1. **CNN-LSTM Hybrid** - Temporal pattern recognition
2. **Depthwise Separable CNN (MobileNet-inspired)** - Parameter efficiency
3. **CNN with Attention Mechanism** - Feature importance weighting

---

## 1. Literature Review Summary

### 1.1 Recent Trends in HAR (2024-2025)

**Key Findings from Research:**

1. **Hybrid CNN-LSTM Architectures** (Nature, 2025)
   - CNN for spatial feature extraction + LSTM for temporal dependencies
   - Achieves 96-98% accuracy on UCI HAR dataset
   - Reference: "Efficient human activity recognition on edge devices"
   - Link: https://www.nature.com/articles/s41598-025-98571-2

2. **TinierHAR Architecture** (arXiv, 2025)
   - Ultra-lightweight using depthwise separable convolutions
   - Residual connections + gated recurrent units
   - Optimized for microcontroller deployment
   - Reference: arXiv:2507.07949v1
   - Link: https://arxiv.org/html/2507.07949v1

3. **Attention-Based Models** (Nature, 2025)
   - Self-attention mechanisms for temporal pattern focus
   - Dual Inception-Attention-BiGRU architecture
   - State-of-the-art accuracy on multiple datasets
   - Reference: "Enhanced Dual Inception-Attention-BiGRU"
   - Link: https://www.nature.com/articles/s41598-025-32859-1

4. **Microcontroller LSTM Implementation** (MDPI, 2024)
   - Practical LSTM deployment on resource-constrained devices
   - Memory optimization techniques
   - Reference: "Microcontroller Implementation of LSTM Neural Networks"
   - Link: https://www.mdpi.com/1424-8220/25/12/3831

### 1.2 Key Architectural Components

**Proven Techniques for TinyML HAR:**

1. **Depthwise Separable Convolutions**
   - Reduces parameters by 8-9x compared to standard convolutions
   - Maintains accuracy with proper training
   - Used in MobileNet, EfficientNet architectures

2. **Temporal Modeling**
   - LSTM/GRU layers capture sequential dependencies
   - Bidirectional variants improve context understanding
   - Critical for activity recognition

3. **Attention Mechanisms**
   - Focus on important time steps and features
   - Improves interpretability
   - Moderate parameter overhead

4. **Residual Connections**
   - Enable deeper networks without gradient vanishing
   - Improve training stability
   - Minimal parameter addition

---

## 2. Recommended Model Architectures

### 2.1 Model 1: CNN-LSTM Hybrid

**Architecture Overview:**
```
Input (561, 1)
├─> Conv1D(32, kernel=3) + ReLU
├─> MaxPooling1D(2)
├─> Conv1D(64, kernel=3) + ReLU
├─> MaxPooling1D(2)
├─> LSTM(64 units)
├─> Dropout(0.3)
├─> Dense(32) + ReLU
├─> Dropout(0.3)
└─> Dense(6) + Softmax
```

**Rationale:**
- CNN layers extract local patterns from sensor data
- LSTM captures temporal dependencies across time steps
- Proven architecture in recent literature (96-98% accuracy)

**Expected Performance:**
- Accuracy: 96-97% (improvement over 95.83% baseline)
- Parameters: ~150K (smaller than baseline 283K)
- Model Size: ~600 KB (before quantization)

**Advantages:**
✅ Better temporal modeling than baseline CNN
✅ Captures long-term dependencies
✅ Proven effectiveness in HAR tasks

**Challenges:**
⚠️ LSTM adds computational complexity
⚠️ May require careful memory management on ESP32

---

### 2.2 Model 2: Depthwise Separable CNN (MobileNet-inspired)

**Architecture Overview:**
```
Input (561, 1)
├─> DepthwiseConv1D(kernel=5) + ReLU
├─> PointwiseConv1D(32) + ReLU
├─> MaxPooling1D(2)
├─> Dropout(0.2)
├─> DepthwiseConv1D(kernel=5) + ReLU
├─> PointwiseConv1D(64) + ReLU
├─> MaxPooling1D(2)
├─> Dropout(0.2)
├─> GlobalAveragePooling1D()
├─> Dense(64) + ReLU
├─> Dropout(0.3)
└─> Dense(6) + Softmax
```

**Rationale:**
- Depthwise separable convolutions drastically reduce parameters
- Inspired by MobileNet architecture for mobile/embedded devices
- TinierHAR paper demonstrates effectiveness for HAR

**Expected Performance:**
- Accuracy: 94-96% (comparable to baseline)
- Parameters: ~30-50K (5-9x reduction!)
- Model Size: ~120-200 KB (before quantization), ~30-50 KB (after)

**Advantages:**
✅ Extremely parameter-efficient
✅ Much smaller model size (closer to ESP32 target)
✅ Faster inference time
✅ Lower memory footprint

**Challenges:**
⚠️ May have slightly lower accuracy than standard CNN
⚠️ Requires careful hyperparameter tuning

---

### 2.3 Model 3: CNN with Self-Attention

**Architecture Overview:**
```
Input (561, 1)
├─> Conv1D(32, kernel=5) + ReLU
├─> MaxPooling1D(2)
├─> Conv1D(64, kernel=5) + ReLU
├─> MaxPooling1D(2)
├─> Self-Attention Layer (multi-head, 4 heads)
├─> GlobalAveragePooling1D()
├─> Dense(64) + ReLU
├─> Dropout(0.3)
└─> Dense(6) + Softmax
```

**Rationale:**
- Attention mechanism focuses on important time steps
- Improves model interpretability
- Recent papers show 2-3% accuracy improvement

**Expected Performance:**
- Accuracy: 96-97% (improvement over baseline)
- Parameters: ~200K (moderate size)
- Model Size: ~800 KB (before quantization)

**Advantages:**
✅ Better feature importance weighting
✅ Improved accuracy on confused classes (SITTING/STANDING)
✅ More interpretable predictions

**Challenges:**
⚠️ Attention adds parameters and complexity
⚠️ May be challenging to quantize effectively

---

## 3. Comparative Analysis

### 3.1 Model Comparison Matrix

| Metric | Baseline CNN | CNN-LSTM | Depthwise CNN | CNN-Attention |
|--------|--------------|----------|---------------|---------------|
| **Expected Accuracy** | 95.83% | 96-97% | 94-96% | 96-97% |
| **Parameters** | 283K | ~150K | ~30-50K | ~200K |
| **Model Size (float32)** | 1.08 MB | ~600 KB | ~120-200 KB | ~800 KB |
| **Model Size (int8)** | ~270 KB | ~150 KB | ~30-50 KB | ~200 KB |
| **Inference Speed** | Fast | Medium | Very Fast | Medium |
| **Memory Usage** | Medium | High | Low | Medium |
| **Temporal Modeling** | ❌ | ✅✅ | ❌ | ✅ |
| **Parameter Efficiency** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **ESP32 Suitability** | ⚠️ | ⚠️ | ✅✅ | ⚠️ |
| **Implementation Complexity** | Easy | Medium | Medium | Hard |

### 3.2 Trade-off Analysis

**Accuracy vs Size:**
- **Best Accuracy:** CNN-LSTM, CNN-Attention (96-97%)
- **Best Size:** Depthwise CNN (~30-50 KB quantized) ✅ **Meets ESP32 target!**
- **Balanced:** CNN-LSTM (good accuracy, moderate size)

**Speed vs Accuracy:**
- **Fastest:** Depthwise CNN (fewer operations)
- **Slowest:** CNN-LSTM (sequential LSTM processing)
- **Balanced:** CNN-Attention (parallel attention computation)

**Deployment Feasibility:**
- **Most Feasible:** Depthwise CNN (small size, fast inference)
- **Challenging:** CNN-LSTM (memory for LSTM states)
- **Moderate:** CNN-Attention (attention computation overhead)

---

## 4. Final Recommendations

### 4.1 Primary Recommendation: Depthwise Separable CNN

**Why this model?**
1. ✅ **Meets ESP32 constraints:** ~30-50 KB after quantization (< 100 KB target)
2. ✅ **Parameter efficient:** 5-9x reduction vs baseline
3. ✅ **Fast inference:** Fewer operations = lower latency
4. ✅ **Proven architecture:** MobileNet/TinierHAR success
5. ✅ **Acceptable accuracy:** 94-96% is still excellent for HAR

**Implementation Priority:** HIGH

### 4.2 Secondary Recommendation: CNN-LSTM Hybrid

**Why this model?**
1. ✅ **Better temporal modeling:** Captures sequential patterns
2. ✅ **Improved accuracy:** 96-97% expected
3. ✅ **Smaller than baseline:** ~150K parameters
4. ⚠️ **May need optimization:** Memory management for LSTM states

**Implementation Priority:** MEDIUM

### 4.3 Tertiary Recommendation: CNN-Attention

**Why this model?**
1. ✅ **Improved accuracy:** 96-97% expected
2. ✅ **Better interpretability:** Attention weights show important features
3. ⚠️ **Larger size:** May still exceed ESP32 constraints
4. ⚠️ **Complex implementation:** Attention mechanism requires careful coding

**Implementation Priority:** LOW (Optional)

---

## 5. Implementation Strategy

### 5.1 Phased Approach

**Phase 3A: Implement Depthwise Separable CNN**
- Priority: HIGH
- Timeline: 1-2 days
- Goal: Achieve <100 KB model size while maintaining >94% accuracy

**Phase 3B: Implement CNN-LSTM Hybrid**
- Priority: MEDIUM
- Timeline: 1-2 days
- Goal: Achieve 96-97% accuracy with reasonable size

**Phase 3C: Implement CNN-Attention (Optional)**
- Priority: LOW
- Timeline: 1-2 days
- Goal: Explore attention mechanisms for future improvements

### 5.2 Success Criteria

**Minimum Requirements:**
- ✅ Accuracy ≥ 94% on UCI HAR test set
- ✅ Model size < 100 KB after INT8 quantization
- ✅ Inference time < 50ms on ESP32

**Stretch Goals:**
- 🎯 Accuracy ≥ 96%
- 🎯 Model size < 50 KB
- 🎯 Inference time < 30ms

---

## 6. References

### Key Papers

1. **TinierHAR** (arXiv, 2025)
   - https://arxiv.org/html/2507.07949v1
   - Ultra-lightweight depthwise separable architecture

2. **Efficient HAR on Edge Devices** (Nature, 2025)
   - https://www.nature.com/articles/s41598-025-98571-2
   - CNN-LSTM hybrid with 96-98% accuracy

3. **Microcontroller LSTM Implementation** (MDPI, 2024)
   - https://www.mdpi.com/1424-8220/25/12/3831
   - Practical LSTM deployment on microcontrollers

4. **Enhanced Dual Inception-Attention-BiGRU** (Nature, 2025)
   - https://www.nature.com/articles/s41598-025-32859-1
   - State-of-the-art attention-based HAR

5. **Comprehensive Survey on TinyML for HAR** (IEEE, 2025)
   - https://ieeexplore.ieee.org/iel8/6488907/11121332/10979983.pdf
   - Overview of TinyML optimization techniques

---

**Document Status:** ✅ Complete
**Next Action:** Begin Phase 3 - Implementation of selected models

