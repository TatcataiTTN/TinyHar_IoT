# Dataset Download Instructions

**Status:** Manual download required due to network issues  
**Last Updated:** January 2026

---

## Quick Start - Manual Download

### Option 1: Direct Download from UCI Repository

1. **Visit the official UCI repository:**
   ```
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
   ```

2. **Click the "Download" button** on the page

3. **Save the file** as `UCI HAR Dataset.zip`

4. **Move to datasets folder:**
   ```bash
   mv ~/Downloads/"UCI HAR Dataset.zip" datasets/
   cd datasets
   unzip "UCI HAR Dataset.zip"
   ```

5. **Verify extraction:**
   ```bash
   ls -la "UCI HAR Dataset"
   ```

### Option 2: Kaggle Download (Recommended)

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle API credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download dataset:**
   ```bash
   cd datasets
   kaggle datasets download -d uciml/human-activity-recognition-with-smartphones
   unzip human-activity-recognition-with-smartphones.zip
   ```

### Option 3: Google Drive Mirror (Alternative)

If UCI repository is down, use this Google Drive mirror:
```
https://drive.google.com/file/d/1xaW8T7c5Uj8JYLnJVvRqciAJvDRPvtkv/view
```

---

## Verification

After downloading, verify the dataset structure:

```bash
cd datasets
python3 << EOF
import os

required_files = [
    'UCI HAR Dataset/train/X_train.txt',
    'UCI HAR Dataset/train/y_train.txt',
    'UCI HAR Dataset/test/X_test.txt',
    'UCI HAR Dataset/test/y_test.txt',
    'UCI HAR Dataset/features.txt',
    'UCI HAR Dataset/activity_labels.txt'
]

print("Verifying UCI HAR Dataset...")
all_ok = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} NOT FOUND")
        all_ok = False

if all_ok:
    print("\n✅ Dataset is ready!")
else:
    print("\n❌ Some files are missing!")
EOF
```

Expected output:
```
✅ UCI HAR Dataset/train/X_train.txt (26,561,747 bytes)
✅ UCI HAR Dataset/train/y_train.txt (14,704 bytes)
✅ UCI HAR Dataset/test/X_test.txt (11,182,699 bytes)
✅ UCI HAR Dataset/test/y_test.txt (5,894 bytes)
✅ UCI HAR Dataset/features.txt (15,785 bytes)
✅ UCI HAR Dataset/activity_labels.txt (80 bytes)

✅ Dataset is ready!
```

---

## Dataset Structure

```
datasets/
└── UCI HAR Dataset/
    ├── README.txt
    ├── features_info.txt
    ├── features.txt (561 feature names)
    ├── activity_labels.txt (6 activities)
    ├── train/
    │   ├── X_train.txt (7352 samples × 561 features)
    │   ├── y_train.txt (7352 labels)
    │   ├── subject_train.txt (7352 subject IDs)
    │   └── Inertial Signals/ (raw sensor data - optional)
    └── test/
        ├── X_test.txt (2947 samples × 561 features)
        ├── y_test.txt (2947 labels)
        ├── subject_test.txt (2947 subject IDs)
        └── Inertial Signals/ (raw sensor data - optional)
```

---

## Troubleshooting

### Problem: Download is very slow
**Solution:** Use Kaggle option or Google Drive mirror

### Problem: Zip file is corrupted
**Solution:** 
1. Delete the corrupted file: `rm UCI_HAR.zip`
2. Try downloading again with a different browser
3. Verify file size should be ~60MB

### Problem: Permission denied when extracting
**Solution:** 
```bash
chmod +x datasets
cd datasets
unzip "UCI HAR Dataset.zip"
```

---

## Next Steps

After successful download:

1. **Explore the data:**
   ```bash
   python3 scripts/explore_data.py
   ```

2. **Train the model:**
   ```bash
   python3 scripts/train_model.py
   ```

3. **Convert to TFLite:**
   ```bash
   python3 scripts/convert_to_tflite.py
   ```

---

**Note:** The automatic download script (`scripts/download_dataset.py`) may fail due to network issues with the UCI repository. Manual download is recommended for reliability.

**Maintained by:** TinyHAR Project Team

