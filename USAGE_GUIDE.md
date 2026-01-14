# Complete Usage Guide - TinyHAR Project

**Date:** January 2026  
**Purpose:** Step-by-step guide to train, evaluate, and deploy HAR models

---

## Prerequisites

### 1. Environment Setup
```bash
# Ensure you're in the project directory
cd "/Users/tuannghiat/Downloads/Project IoT và Ứng Dụng HUST"

# Activate virtual environment (if using one)
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Verify dependencies
pip install -r requirements.txt
```

### 2. Dataset Verification
```bash
# Check if UCI HAR dataset exists
ls -la datasets/UCI\ HAR\ Dataset/

# Should see:
# - train/ directory
# - test/ directory
# - activity_labels.txt
```

---

## Quick Start (Recommended for First Run)

### Option 1: Quick Test (10 epochs, ~30 minutes)
```bash
python src/run_pipeline.py --quick
```

This will:
- ✅ Train all 4 models (10 epochs each)
- ✅ Evaluate all models on test set
- ✅ Convert to TFLite and generate C headers
- ✅ Generate comprehensive reports

**Expected Output:**
```
models/
├── har_model_cnn_simple.h5
├── har_model_cnn_lstm.h5
├── har_model_depthwise_cnn.h5
├── har_model_cnn_attention.h5
├── *.tflite (TFLite models)
├── *.h (C headers for ESP32)
├── training_results_comparison.json
├── model_comparison_report.txt
├── model_comparison_plots.png
├── evaluation_results_all_models.json
├── comprehensive_evaluation_report.txt
├── confusion_matrix_*.png
├── deployment_results.json
└── deployment_report.txt
```

---

## Full Training (50 epochs, ~2-3 hours)

### Option 2: Complete Pipeline
```bash
python src/run_pipeline.py
```

### Option 3: Custom Configuration
```bash
# Custom number of epochs
python src/run_pipeline.py --epochs 30

# Skip specific phases
python src/run_pipeline.py --skip-train    # Use existing models
python src/run_pipeline.py --skip-eval     # Skip evaluation
python src/run_pipeline.py --skip-deploy   # Skip deployment
```

---

## Step-by-Step Execution

### Step 1: Train All Models
```bash
python src/train_all_models.py
```

**What it does:**
- Loads UCI HAR dataset
- Trains 4 model architectures:
  1. CNN Simple (baseline)
  2. CNN-LSTM Hybrid
  3. Depthwise Separable CNN
  4. CNN with Attention
- Saves trained models (.h5 files)
- Generates comparison report and plots

**Expected time:** 2-3 hours (50 epochs)

**Output files:**
- `models/har_model_*.h5` - Trained models
- `models/training_results_comparison.json`
- `models/model_comparison_report.txt`
- `models/model_comparison_plots.png`

---

### Step 2: Evaluate All Models
```bash
python src/evaluate_all_models.py
```

**What it does:**
- Loads all trained models
- Evaluates on test set
- Calculates metrics (accuracy, precision, recall, F1)
- Generates confusion matrices
- Creates comprehensive evaluation report

**Expected time:** 5-10 minutes

**Output files:**
- `models/evaluation_results_all_models.json`
- `models/comprehensive_evaluation_report.txt`
- `models/confusion_matrix_*.png`

---

### Step 3: Deploy All Models
```bash
python src/deploy_all_models.py
```

**What it does:**
- Converts models to TFLite format
- Applies INT8 quantization
- Generates C header files for ESP32
- Validates quantized models
- Creates deployment report

**Expected time:** 5-10 minutes

**Output files:**
- `models/*.tflite` - TensorFlow Lite models
- `models/*.h` - C header files
- `models/deployment_results.json`
- `models/deployment_report.txt`

---

## Understanding the Results

### 1. Training Results

**File:** `models/model_comparison_report.txt`

**Key metrics to look for:**
- **Accuracy:** Higher is better (target: >94%)
- **Parameters:** Lower is better (target: <50K for ESP32)
- **Size:** Lower is better (target: <100KB before quantization)
- **Training Time:** Informational

**Example:**
```
Model                        Accuracy  Parameters  Size (KB)
Baseline CNN Simple          95.83%    283,718     1108.27
CNN-LSTM Hybrid              96.50%    41,638      162.65
Depthwise Separable CNN      95.20%    6,924       27.05
CNN with Attention           96.80%    31,814      124.27
```

---

### 2. Evaluation Results

**File:** `models/comprehensive_evaluation_report.txt`

**What to check:**
- Overall accuracy on test set
- Per-class performance (precision, recall, F1)
- Confusion matrices (which activities are confused)

**Good signs:**
- ✅ Accuracy ≥ 94%
- ✅ Balanced performance across all classes
- ✅ Low confusion between different activities

---

### 3. Deployment Results

**File:** `models/deployment_report.txt`

**Key information:**
- Model sizes after quantization
- ESP32 deployment readiness
- Accuracy drop after quantization

**ESP32 Ready criteria:**
- ✅ TFLite size < 100 KB
- ✅ Accuracy drop < 2%

**Example:**
```
depthwise_cnn:
  TFLite Size: 6.76 KB
  ✅ READY for ESP32 deployment (< 100 KB)
  Accuracy drop: 0.50%
```

---

## Selecting the Best Model

### For Maximum Accuracy
**Choose:** CNN-Attention or CNN-LSTM
- Expected: 96-97% accuracy
- Size: 31-41 KB (quantized)
- Use case: When accuracy is critical

### For Minimum Size (ESP32 Recommended)
**Choose:** Depthwise Separable CNN
- Expected: 94-96% accuracy
- Size: ~7 KB (quantized)
- Use case: Resource-constrained deployment

### For Balanced Performance
**Choose:** CNN-Attention
- Expected: 96-97% accuracy
- Size: ~31 KB (quantized)
- Use case: Good balance of accuracy and size

---

## Deploying to ESP32

### 1. Locate the C Header File
```bash
# Example for Depthwise CNN
ls models/har_model_depthwise_cnn_quantized.h
```

### 2. Copy to ESP32 Project
```bash
cp models/har_model_depthwise_cnn_quantized.h /path/to/esp32/project/
```

### 3. Include in ESP32 Code
```cpp
#include "har_model_depthwise_cnn_quantized.h"

// Load model
const tflite::Model* model = tflite::GetModel(har_model_depthwise_cnn_quantized_data);

// Set up interpreter
// ... (see TECHNICAL_PROTOCOLS.md for details)
```

### 4. Reference Documentation
- `docs/TECHNICAL_PROTOCOLS.md` - TFLite Micro integration
- `docs/IMPLEMENTATION_PLAN.md` - ESP32 firmware guide

---

## Troubleshooting

### Issue: "No module named 'tensorflow'"
```bash
pip install tensorflow==2.15.0
```

### Issue: "Dataset not found"
```bash
# Download UCI HAR dataset
# See datasets/README.md for instructions
```

### Issue: "Out of memory during training"
```bash
# Reduce batch size in train_all_models.py
# Change: batch_size=32 to batch_size=16
```

### Issue: "Model file not found"
```bash
# Make sure you ran training first
python src/train_all_models.py
```

---

## Advanced Usage

### Train Single Model
```python
# Edit src/train.py
MODEL_TYPE = 'depthwise_cnn'  # or 'cnn_lstm', 'cnn_attention'
python src/train.py
```

### Evaluate Single Model
```python
# Edit src/evaluate.py
MODEL_PATH = 'models/har_model_depthwise_cnn.h5'
python src/evaluate.py
```

### Convert Single Model
```python
# Edit src/convert_tflite.py
MODEL_PATH = 'models/har_model_depthwise_cnn.h5'
python src/convert_tflite.py
```

---

## Performance Benchmarks

### Training Time (50 epochs, CPU)
- CNN Simple: ~45 minutes
- CNN-LSTM: ~60 minutes
- Depthwise CNN: ~30 minutes
- CNN-Attention: ~50 minutes

### Model Sizes (after INT8 quantization)
- CNN Simple: ~277 KB
- CNN-LSTM: ~41 KB ✅
- Depthwise CNN: ~7 KB ✅✅
- CNN-Attention: ~31 KB ✅

### Expected Accuracy
- CNN Simple: 95.83%
- CNN-LSTM: 96-97%
- Depthwise CNN: 94-96%
- CNN-Attention: 96-97%

---

## Next Steps After Training

1. ✅ Review comparison reports
2. ✅ Select best model for your use case
3. ✅ Copy .h file to ESP32 project
4. 🔄 Integrate with ESP32 firmware
5. 🔄 Test on actual hardware
6. 🔄 Optimize if needed

---

**Document Status:** ✅ Complete  
**Last Updated:** January 2026  
**Ready to use!**

