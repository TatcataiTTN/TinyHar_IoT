# HAR Datasets Comparison and Download Guide

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Purpose:** Detailed comparison and download instructions for HAR datasets

---

## Quick Reference Table

| Dataset | Subjects | Activities | Sensors | Rate | Size | Best For | Priority |
|---------|----------|------------|---------|------|------|----------|----------|
| **UCI HAR** | 30 | 6 | Acc+Gyro | 50Hz | 60MB | Baseline benchmark | ⭐⭐⭐ REQUIRED |
| **WISDM** | 36 | 6 | Acc | 20Hz | 40MB | Matching sampling rate | ⭐⭐ Recommended |
| **PAMAP2** | 9 | 18 | 3×IMU | 100Hz | 500MB | Multi-sensor fusion | ⭐ Optional |
| **MotionSense** | 24 | 6 | Acc+Gyro | 50Hz | 100MB | Additional validation | ⭐ Optional |
| **HuGaDB** | Multiple | Gait | IMU | Variable | 200MB | Gait analysis | Optional |

---

## 1. UCI HAR Dataset (REQUIRED)

### Overview
- **Full Name:** Human Activity Recognition Using Smartphones
- **Year:** 2012
- **Institution:** University of California, Irvine
- **Status:** ⭐⭐⭐ REQUIRED for TinyHAR baseline

### Specifications
- **Participants:** 30 volunteers (19-48 years old)
- **Activities:** 6 classes
  1. WALKING
  2. WALKING_UPSTAIRS
  3. WALKING_DOWNSTAIRS
  4. SITTING
  5. STANDING
  6. LAYING
- **Sensors:** Accelerometer + Gyroscope (Samsung Galaxy S II)
- **Sampling Rate:** 50Hz
- **Window Size:** 2.56 seconds (128 readings)
- **Features:** 561 pre-computed features
- **Train/Test Split:** 70%/30% (21/9 subjects)

### Download Instructions

**Option 1: UCI ML Repository (Official)**
```bash
# Direct download
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

# Extract
unzip "UCI HAR Dataset.zip" -d datasets/
```

**Option 2: Kaggle (Easier)**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API key (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d uciml/human-activity-recognition-with-smartphones -p datasets/

# Extract
cd datasets
unzip human-activity-recognition-with-smartphones.zip
```

**Option 3: Manual Download**
1. Visit: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Click "Download" button
3. Extract to `datasets/UCI_HAR/`

### File Structure
```
UCI HAR Dataset/
├── README.txt
├── features_info.txt
├── features.txt (561 feature names)
├── activity_labels.txt (6 activity names)
├── train/
│   ├── X_train.txt (7352 samples × 561 features)
│   ├── y_train.txt (7352 labels)
│   └── subject_train.txt (7352 subject IDs)
└── test/
    ├── X_test.txt (2947 samples × 561 features)
    ├── y_test.txt (2947 labels)
    └── subject_test.txt (2947 subject IDs)
```

### Why UCI HAR is Essential
✅ Industry-standard benchmark  
✅ 561 features match TinyHAR architecture  
✅ Extensive research baseline (1000+ citations)  
✅ Pre-processed and ready to use  
✅ Subject-wise train/test split prevents data leakage  

### Citation
```bibtex
@inproceedings{anguita2013public,
  title={A public domain dataset for human activity recognition using smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis},
  booktitle={Esann},
  volume={3},
  pages={3},
  year={2013}
}
```

---

## 2. WISDM Dataset (RECOMMENDED)

### Overview
- **Full Name:** Wireless Sensor Data Mining
- **Year:** 2012
- **Institution:** Fordham University
- **Status:** ⭐⭐ RECOMMENDED (20Hz matches TinyHAR!)

### Specifications
- **Participants:** 36 users
- **Activities:** 6 classes (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
- **Sensors:** Accelerometer only (smartphone in pocket)
- **Sampling Rate:** 20Hz ⭐ (EXACT match with TinyHAR)
- **Format:** CSV (timestamp, user, activity, x, y, z)
- **Size:** ~40MB

### Download Instructions

**Kaggle Download:**
```bash
# Download WISDM dataset
kaggle datasets download -d uciml/wisdm-dataset -p datasets/

# Extract
cd datasets
unzip wisdm-dataset.zip -d WISDM/
```

**Manual Download:**
1. Visit: https://www.cis.fordham.edu/wisdm/dataset.php
2. Download WISDM_ar_v1.1_raw.txt
3. Place in `datasets/WISDM/`

### File Format
```csv
user,activity,timestamp,x-accel,y-accel,z-accel
33,Jogging,49105962326000,-0.694,12.68,0.50;
33,Jogging,49106062271000,5.12,11.26,-0.59;
```

### Advantages for TinyHAR
✅ 20Hz sampling rate (no downsampling needed)  
✅ Simple CSV format  
✅ Good for testing real-time collection pipeline  
✅ Complements UCI HAR with different sensor placement  

---

## 3. PAMAP2 Dataset (OPTIONAL - Advanced)

### Overview
- **Full Name:** Physical Activity Monitoring for Aging People
- **Year:** 2012
- **Institution:** University of Kaiserslautern
- **Status:** ⭐ OPTIONAL (Advanced multi-sensor)

### Specifications
- **Participants:** 9 subjects
- **Activities:** 18 activities (lying, sitting, standing, walking, running, cycling, etc.)
- **Sensors:** 3 IMU units (hand, chest, ankle)
  - Each IMU: Accelerometer (16g) + Gyroscope (2000°/s) + Magnetometer
- **Sampling Rate:** 100Hz
- **Format:** Space-separated TXT files
- **Size:** ~500MB

### Download Instructions
```bash
# UCI Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip

# Extract
unzip PAMAP2_Dataset.zip -d datasets/PAMAP2/
```

### File Structure
```
PAMAP2_Dataset/
├── Protocol/
│   ├── subject101.dat
│   ├── subject102.dat
│   └── ...
└── Optional/
    └── (additional sessions)
```

### Data Format (54 columns)
- Column 1: Timestamp
- Column 2: Activity ID (1-24)
- Column 3: Heart rate
- Columns 4-20: IMU hand (temp, 3D-acc, 3D-acc, 3D-gyro, 3D-mag, orientation)
- Columns 21-37: IMU chest
- Columns 38-54: IMU ankle

### Advantages
✅ Multi-sensor data (like GY85 with 3 sensors)  
✅ Includes magnetometer data  
✅ More diverse activities (18 vs 6)  
✅ Good for testing sensor fusion algorithms  

---

## 4. MotionSense Dataset

### Overview
- **Institution:** University of Birmingham
- **Participants:** 24 subjects
- **Activities:** 6 (downstairs, upstairs, walking, jogging, sitting, standing)
- **Sensors:** Accelerometer + Gyroscope (iPhone)
- **Sampling Rate:** 50Hz

### Download
```bash
kaggle datasets download -d malekzadeh/motionsense-dataset -p datasets/
```

---

## 5. Dataset Comparison Matrix

### Feature Comparison

| Feature | UCI HAR | WISDM | PAMAP2 | MotionSense |
|---------|---------|-------|--------|-------------|
| Pre-computed features | ✅ 561 | ❌ Raw | ❌ Raw | ❌ Raw |
| Gyroscope | ✅ | ❌ | ✅ | ✅ |
| Magnetometer | ❌ | ❌ | ✅ | ❌ |
| Multiple body positions | ❌ | ❌ | ✅ (3) | ❌ |
| Subject diversity | ✅ 30 | ✅ 36 | ⚠️ 9 | ✅ 24 |
| Activity diversity | ⚠️ 6 | ⚠️ 6 | ✅ 18 | ⚠️ 6 |
| Ready for TinyHAR | ✅ | ⚠️ | ⚠️ | ⚠️ |

### Recommended Download Priority

**Phase 1 (Essential):**
1. ✅ UCI HAR - Required for baseline

**Phase 2 (Recommended):**
2. ✅ WISDM - Same 20Hz sampling rate

**Phase 3 (Optional):**
3. ⚠️ PAMAP2 - If exploring multi-sensor fusion
4. ⚠️ MotionSense - Additional validation data

---

## 6. Storage Requirements

```
datasets/
├── UCI_HAR/           (~60 MB)
├── WISDM/             (~40 MB)
├── PAMAP2/            (~500 MB)
├── MotionSense/       (~100 MB)
└── README.md

Total: ~700 MB (all datasets)
Minimum: ~60 MB (UCI HAR only)
```

---

## 7. Quick Start Commands

### Download All Essential Datasets
```bash
#!/bin/bash
# Create datasets directory
mkdir -p datasets
cd datasets

# 1. UCI HAR (Required)
echo "Downloading UCI HAR..."
kaggle datasets download -d uciml/human-activity-recognition-with-smartphones
unzip human-activity-recognition-with-smartphones.zip -d UCI_HAR/
rm human-activity-recognition-with-smartphones.zip

# 2. WISDM (Recommended)
echo "Downloading WISDM..."
kaggle datasets download -d uciml/wisdm-dataset
unzip wisdm-dataset.zip -d WISDM/
rm wisdm-dataset.zip

echo "✅ Essential datasets downloaded!"
```

Save as `download_datasets.sh` and run:
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

---

**Next Steps:**
1. Download UCI HAR dataset (required)
2. Verify data integrity
3. Explore data structure
4. Proceed to model training pipeline

**Document maintained by:** TinyHAR Project Team
