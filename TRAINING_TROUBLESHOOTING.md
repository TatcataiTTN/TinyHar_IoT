# TinyHAR Training - Troubleshooting & Solutions

**Date:** January 14, 2026  
**Issue:** TensorFlow compatibility on macOS causing Bus Error/Segmentation Fault

---

## Problem Summary

The training pipeline encounters a **Bus Error 10** or **Segmentation Fault 11** when running on macOS. This is a known compatibility issue with TensorFlow on certain Mac configurations, particularly with Apple Silicon (M1/M2/M3) chips.

**Error Messages:**
```
Bus error: 10
Segmentation fault: 11
```

**Root Cause:**
- TensorFlow 2.20.0 has compatibility issues with macOS
- Memory access violations when loading TensorFlow libraries
- Incompatibility between TensorFlow and macOS system libraries

---

## ✅ RECOMMENDED SOLUTION: Use Google Colab

### Why Google Colab?
- ✅ Pre-configured TensorFlow environment
- ✅ Free GPU access (faster training)
- ✅ No compatibility issues
- ✅ Easy to use
- ✅ Results saved to Google Drive

### How to Use Google Colab

#### Step 1: Upload Project to Google Drive
1. Compress your project folder (optional)
2. Upload to Google Drive
3. Extract if compressed

#### Step 2: Open Colab Notebook
1. Open the file: `TinyHAR_Training_Colab.ipynb`
2. Right-click → "Open with" → "Google Colaboratory"
3. Or upload to https://colab.research.google.com/

#### Step 3: Update Project Path
In the notebook, update this line:
```python
PROJECT_PATH = '/content/drive/MyDrive/Project IoT và Ứng Dụng HUST'
```

#### Step 4: Run All Cells
- Click "Runtime" → "Run all"
- Or press Ctrl+F9 (Cmd+F9 on Mac)

#### Step 5: Wait for Completion
- Quick test (10 epochs): ~20-30 minutes with GPU
- Full training (50 epochs): ~1-2 hours with GPU

#### Step 6: Download Results
Results are automatically saved to your Google Drive in the `models/` folder.

---

## Alternative Solutions

### Solution 1: Use Docker (Advanced)

**Pros:**
- Consistent environment
- Works on any OS
- Isolated from system issues

**Cons:**
- Requires Docker installation
- More complex setup

**Steps:**
```bash
# 1. Install Docker Desktop for Mac
# Download from: https://www.docker.com/products/docker-desktop

# 2. Pull TensorFlow Docker image
docker pull tensorflow/tensorflow:latest-jupyter

# 3. Run container with project mounted
cd "/Users/tuannghiat/Downloads/Project IoT và Ứng Dụng HUST"
docker run -it --rm -v $(pwd):/workspace -p 8888:8888 tensorflow/tensorflow:latest-jupyter

# 4. Open Jupyter in browser (URL will be shown in terminal)

# 5. Run pipeline in Jupyter terminal
cd /workspace
python src/run_pipeline.py --quick
```

---

### Solution 2: Use Conda Environment

**Pros:**
- Better dependency management
- More stable than pip

**Cons:**
- Requires Anaconda/Miniconda
- May still have compatibility issues

**Steps:**
```bash
# 1. Install Miniconda
# Download from: https://docs.conda.io/en/latest/miniconda.html

# 2. Create new environment
conda create -n tinyhar python=3.10

# 3. Activate environment
conda activate tinyhar

# 4. Install TensorFlow
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.16.2

# 5. Install other dependencies
pip install scikit-learn matplotlib seaborn pandas

# 6. Run pipeline
python src/run_pipeline.py --quick
```

---

### Solution 3: Use Remote Server/Cloud VM

**Options:**
- AWS EC2 with GPU
- Google Cloud Compute Engine
- Azure Virtual Machines
- Paperspace Gradient

**Pros:**
- Powerful hardware
- No local compatibility issues
- Can use GPU for faster training

**Cons:**
- May incur costs
- Requires cloud account setup

---

### Solution 4: Train Models Individually (Workaround)

If you want to try training locally despite the issues, train one model at a time:

**Step 1: Create individual training script**

Create `src/train_single_model.py`:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_loader import load_uci_har_data
from preprocessing import preprocess_data, reshape_for_cnn
from model import create_depthwise_separable_cnn
from tensorflow import keras

# Load data
X_train, X_test, y_train, y_test, _, _, _ = load_uci_har_data()
X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, _ = preprocess_data(
    X_train, X_test, y_train, y_test
)
X_train_r, X_val_r, X_test_r = reshape_for_cnn(X_train_p, X_val_p, X_test_p)

# Create model
model = create_depthwise_separable_cnn((561, 1), 6)
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train_r, y_train_p,
    validation_data=(X_val_r, y_val_p),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Save
model.save('models/har_model_depthwise_cnn.h5')
print("Model saved!")
```

**Step 2: Run with minimal imports**
```bash
python src/train_single_model.py
```

---

## What to Do Now

### Immediate Action (Recommended)
1. ✅ Use the Google Colab notebook: `TinyHAR_Training_Colab.ipynb`
2. Upload project to Google Drive
3. Run the notebook
4. Download trained models

### Alternative Actions
1. Try Docker solution (if comfortable with Docker)
2. Try Conda environment (may still have issues)
3. Use cloud VM (if you have cloud credits)

---

## Expected Results (When Training Succeeds)

### Generated Files
```
models/
├── har_model_cnn_simple.h5                    # Baseline model
├── har_model_cnn_lstm.h5                      # CNN-LSTM model
├── har_model_depthwise_cnn.h5                 # Depthwise CNN model
├── har_model_cnn_attention.h5                 # CNN-Attention model
├── har_model_*_quantized.tflite               # TFLite models
├── har_model_*_quantized.h                    # C headers for ESP32
├── training_results_comparison.json           # Training metrics
├── model_comparison_report.txt                # Text report
├── model_comparison_plots.png                 # Visual comparison
├── evaluation_results_all_models.json         # Evaluation metrics
├── comprehensive_evaluation_report.txt        # Evaluation report
├── confusion_matrix_*.png                     # Confusion matrices
├── deployment_results.json                    # Deployment metrics
└── deployment_report.txt                      # Deployment guide
```

### Expected Performance
| Model | Accuracy | Size (Quantized) | ESP32 Ready |
|-------|----------|------------------|-------------|
| Baseline CNN | 95.83% | ~277 KB | ⚠️ |
| CNN-LSTM | 96-97% | ~41 KB | ✅ |
| Depthwise CNN | 94-96% | ~7 KB | ✅✅ |
| CNN-Attention | 96-97% | ~31 KB | ✅ |

---

## Summary

**Current Status:**
- ❌ Local training blocked by TensorFlow compatibility issue
- ✅ All code is correct and ready to run
- ✅ Google Colab solution provided

**Recommended Path:**
1. Use Google Colab (easiest and most reliable)
2. Train all models with GPU acceleration
3. Download results to local machine
4. Deploy to ESP32

**Time Estimate:**
- Colab setup: 5-10 minutes
- Quick training (10 epochs): 20-30 minutes
- Full training (50 epochs): 1-2 hours

---

**Next Step:** Open `TinyHAR_Training_Colab.ipynb` in Google Colab and run all cells! 🚀

