#!/usr/bin/env python3
"""
UCI HAR Dataset Downloader with Retry Mechanism
Downloads and extracts the UCI HAR dataset for TinyHAR project
"""

import urllib.request
import zipfile
import os
import sys
import time

def download_with_retry(url, output_path, max_retries=5):
    """Download file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading from {url}")
            
            # Download with progress
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}% ({count * block_size}/{total_size} bytes)")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, output_path, reporthook)
            print("\n✅ Download complete!")
            
            # Verify file size
            file_size = os.path.getsize(output_path)
            print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            if file_size < 100000:  # Less than 100KB is suspicious
                print("⚠️ File size too small, retrying...")
                os.remove(output_path)
                time.sleep(2)
                continue
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Download failed.")
                return False
    
    return False

def extract_zip(zip_path, extract_to='.'):
    """Extract zip file"""
    try:
        print(f"\nExtracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction complete!")
        return True
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return False

def verify_dataset(base_path):
    """Verify that all required files exist"""
    required_files = [
        'train/X_train.txt',
        'train/y_train.txt',
        'train/subject_train.txt',
        'test/X_test.txt',
        'test/y_test.txt',
        'test/subject_test.txt',
        'features.txt',
        'activity_labels.txt'
    ]
    
    print("\nVerifying dataset files...")
    all_exist = True
    for file in required_files:
        full_path = os.path.join(base_path, file)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    # Configuration
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    zip_filename = 'UCI_HAR.zip'
    dataset_folder = 'UCI HAR Dataset'
    
    print("=" * 60)
    print("UCI HAR Dataset Downloader for TinyHAR Project")
    print("=" * 60)
    
    # Download
    if not os.path.exists(zip_filename):
        success = download_with_retry(url, zip_filename)
        if not success:
            print("\n❌ Failed to download dataset. Please try:")
            print("1. Check your internet connection")
            print("2. Download manually from:")
            print("   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
            print("3. Or use Kaggle:")
            print("   kaggle datasets download -d uciml/human-activity-recognition-with-smartphones")
            sys.exit(1)
    else:
        print(f"✅ {zip_filename} already exists, skipping download")
    
    # Extract
    if not os.path.exists(dataset_folder):
        success = extract_zip(zip_filename)
        if not success:
            sys.exit(1)
    else:
        print(f"✅ {dataset_folder} already exists, skipping extraction")
    
    # Verify
    if verify_dataset(dataset_folder):
        print("\n" + "=" * 60)
        print("✅ UCI HAR Dataset is ready!")
        print("=" * 60)
        print(f"\nDataset location: {os.path.abspath(dataset_folder)}")
        print("\nNext steps:")
        print("1. Run: python3 scripts/explore_data.py")
        print("2. Run: python3 scripts/train_model.py")
    else:
        print("\n❌ Dataset verification failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()

