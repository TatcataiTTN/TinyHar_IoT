# TinyHAR Implementation Plan and Roadmap

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Purpose:** Step-by-step implementation guide for replicating TinyHAR project

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Hardware Setup](#2-hardware-setup)
3. [Software Dependencies](#3-software-dependencies)
4. [Phase 1: Data and Model Development](#4-phase-1-data-and-model-development)
5. [Phase 2: ESP32 Implementation](#5-phase-2-esp32-implementation)
6. [Phase 3: Testing and Validation](#6-phase-3-testing-and-validation)
7. [Troubleshooting Guide](#7-troubleshooting-guide)

---

## 1. Project Overview

### 1.1 Timeline Estimate

| Phase | Duration | Complexity | Dependencies |
|-------|----------|------------|--------------|
| Hardware Setup | 1-2 days | Easy | Parts arrival |
| Software Setup | 1 day | Easy | Internet |
| Data Download | 2-4 hours | Easy | Internet |
| Model Training | 4-8 hours | Medium | GPU optional |
| ESP32 Firmware | 2-3 days | Medium | Hardware ready |
| Testing & Debug | 3-5 days | Hard | Full system |
| **Total** | **2-3 weeks** | **Medium** | - |

### 1.2 Skill Requirements

**Required:**
- ✅ Basic Python programming
- ✅ Arduino/ESP32 basics
- ✅ Basic electronics (breadboard, wiring)

**Helpful but Optional:**
- ⭐ Machine learning fundamentals
- ⭐ I2C protocol knowledge
- ⭐ Web development (HTML/JS)

### 1.3 Budget Estimate

| Item | Quantity | Unit Price | Total |
|------|----------|------------|-------|
| ESP32 Dev Board | 1 | $8 | $8 |
| GY85 Sensor Module | 1 | $12 | $12 |
| Breadboard | 1 | $3 | $3 |
| Jumper Wires | 1 set | $2 | $2 |
| USB Cable | 1 | $3 | $3 |
| Power Bank (optional) | 1 | $15 | $15 |
| **Total** | - | - | **$28-43** |

---

## 2. Hardware Setup

### 2.1 Parts List

#### 2.1.1 Required Components

**ESP32 Development Board:**
- Model: ESP32-DevKitC or compatible
- Specifications:
  - Dual-core Xtensa LX6 @ 240MHz
  - 520KB SRAM, 4MB Flash
  - WiFi + Bluetooth
  - GPIO pins with I2C support
- Where to buy: Amazon, AliExpress, Adafruit, SparkFun
- Price: $5-10

**GY85 9-DOF IMU Sensor Module:**
- Contains:
  - ADXL345: 3-axis accelerometer
  - ITG3200/ITG3205: 3-axis gyroscope
  - HMC5883L: 3-axis magnetometer
  - BMP085: Barometric pressure sensor
- Interface: I2C
- Voltage: 3.3V-5V
- Where to buy: Amazon, AliExpress, eBay
- Price: $10-15

**Accessories:**
- Breadboard (400 or 830 points)
- Male-to-male jumper wires (at least 4)
- USB Micro-B or USB-C cable (depending on ESP32 board)

#### 2.1.2 Optional Components

- 10000mAh Power Bank with current measurement (Cuktech or similar)
- Multimeter for debugging
- Logic analyzer for I2C debugging
- Enclosure/case for portable use

### 2.2 Wiring Diagram

```
ESP32 DevKit          GY85 Module
┌─────────────┐      ┌──────────┐
│             │      │          │
│    3.3V ────┼──────┼─── VCC   │
│             │      │          │
│     GND ────┼──────┼─── GND   │
│             │      │          │
│  GPIO21 ────┼──────┼─── SDA   │
│   (SDA)     │      │          │
│             │      │          │
│  GPIO22 ────┼──────┼─── SCL   │
│   (SCL)     │      │          │
│             │      │          │
└─────────────┘      └──────────┘
```

### 2.3 Assembly Instructions

**Step 1: Inspect Components**
- Check ESP32 for bent pins
- Verify GY85 has all sensors populated
- Test USB cable with multimeter (optional)

**Step 2: Connect Power**
- Insert ESP32 into breadboard
- Connect GY85 VCC to ESP32 3.3V (or 5V if available)
- Connect GY85 GND to ESP32 GND

**Step 3: Connect I2C**
- Connect GY85 SDA to ESP32 GPIO21
- Connect GY85 SCL to ESP32 GPIO22

**Step 4: Verify Connections**
- Double-check all connections
- Ensure no short circuits
- Keep wires short (<15cm) to reduce noise

**Step 5: Power On**
- Connect ESP32 to computer via USB
- Check if ESP32 LED lights up
- Verify GY85 power LED (if present)

### 2.4 Hardware Testing

**Test 1: ESP32 Basic Test**
```cpp
void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Test");
}

void loop() {
  Serial.println("Hello from ESP32!");
  delay(1000);
}
```

**Test 2: I2C Scanner**
```cpp
#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Serial.println("I2C Scanner");
}

void loop() {
  byte count = 0;
  for (byte i = 1; i < 127; i++) {
    Wire.beginTransmission(i);
    if (Wire.endTransmission() == 0) {
      Serial.printf("Device found at 0x%02X\n", i);
      count++;
    }
  }
  Serial.printf("Found %d devices\n\n", count);
  delay(5000);
}
```

**Expected Output:**
```
Device found at 0x1E  (HMC5883L)
Device found at 0x53  (ADXL345)
Device found at 0x68  (ITG3200)
Device found at 0x77  (BMP085)
Found 4 devices
```

---

## 3. Software Dependencies

### 3.1 Development Environment Setup

#### 3.1.1 Arduino IDE (Recommended for Beginners)

**Step 1: Install Arduino IDE**
- Download from: https://www.arduino.cc/en/software
- Version: 2.0 or later
- Install for your OS (Windows/Mac/Linux)

**Step 2: Add ESP32 Board Support**
1. Open Arduino IDE
2. Go to File → Preferences
3. Add to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to Tools → Board → Boards Manager
5. Search "esp32"
6. Install "esp32 by Espressif Systems"

**Step 3: Select Board**
- Tools → Board → ESP32 Arduino → ESP32 Dev Module
- Tools → Port → (select your ESP32 port)

#### 3.1.2 PlatformIO (Alternative - Advanced)

**Installation:**
1. Install VS Code
2. Install PlatformIO extension
3. Create new project with ESP32 board

**platformio.ini:**
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps = 
    tanakamasayuki/TensorFlowLite_ESP32@^0.9.0
    bblanchon/ArduinoJson@^6.21.0
    wire
```

### 3.2 Python Environment Setup

**Step 1: Install Python**
- Version: Python 3.8-3.11 (3.10 recommended)
- Download from: https://www.python.org/downloads/

**Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv tinyhar_env

# Activate (Windows)
tinyhar_env\Scripts\activate

# Activate (Mac/Linux)
source tinyhar_env/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install --upgrade pip

# Core ML libraries
pip install tensorflow==2.15.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0

# Data processing
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Data collection
pip install requests==2.31.0

# Jupyter (optional)
pip install jupyter==1.0.0
```

**requirements.txt:**
```
tensorflow==2.15.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
requests==2.31.0
jupyter==1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

### 3.3 Arduino Libraries

**Required Libraries:**

1. **TensorFlowLite_ESP32**
   - Library Manager: Search "TensorFlowLite_ESP32"
   - Install version 0.9.0 or later

2. **ArduinoJson**
   - Library Manager: Search "ArduinoJson"
   - Install version 6.21.0 or later

3. **Wire** (Built-in)
   - Pre-installed with ESP32 board support

4. **WiFi** (Built-in)
   - Pre-installed with ESP32 board support

5. **WebServer** (Built-in)
   - Pre-installed with ESP32 board support

**Installation via Library Manager:**
1. Sketch → Include Library → Manage Libraries
2. Search for library name
3. Click Install

---

## 4. Phase 1: Data and Model Development

### 4.1 Dataset Download

**Task Checklist:**
- [ ] Create datasets directory
- [ ] Download UCI HAR dataset
- [ ] Verify data integrity
- [ ] Explore data structure
- [ ] (Optional) Download WISDM dataset
- [ ] (Optional) Download PAMAP2 dataset

**Commands:**
```bash
# Create directory
mkdir -p datasets
cd datasets

# Option 1: Manual download
# Visit: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
# Download and extract to datasets/UCI_HAR/

# Option 2: Using wget (Linux/Mac)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"
mv "UCI HAR Dataset" UCI_HAR

# Option 3: Using Kaggle CLI
pip install kaggle
kaggle datasets download -d uciml/human-activity-recognition-with-smartphones
unzip human-activity-recognition-with-smartphones.zip -d UCI_HAR
```

**Verify Download:**
```python
import os

# Check if files exist
required_files = [
    'datasets/UCI_HAR/train/X_train.txt',
    'datasets/UCI_HAR/train/y_train.txt',
    'datasets/UCI_HAR/test/X_test.txt',
    'datasets/UCI_HAR/test/y_test.txt',
]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} NOT FOUND")
```

### 4.2 Model Training

**Task Checklist:**
- [ ] Load and explore UCI HAR data
- [ ] Preprocess and normalize features
- [ ] Build neural network model
- [ ] Train with early stopping
- [ ] Evaluate on test set
- [ ] Save trained model

**Training Script (train_model.py):**
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# Load UCI HAR data
def load_uci_har_data():
    print("Loading UCI HAR dataset...")
    X_train = np.loadtxt('datasets/UCI HAR Dataset/train/X_train.txt')
    y_train = np.loadtxt('datasets/UCI HAR Dataset/train/y_train.txt')
    X_test = np.loadtxt('datasets/UCI HAR Dataset/test/X_test.txt')
    y_test = np.loadtxt('datasets/UCI HAR Dataset/test/y_test.txt')

    # Convert labels from 1-6 to 0-5
    y_train = y_train - 1
    y_test = y_test - 1

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# Build model
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train
X_train, y_train, X_test, y_test = load_uci_har_data()

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Build and train
model = build_model(X_train.shape[1], 6)
model.summary()

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save
model.save('models/har_model.h5')
print("Model saved to models/har_model.h5")
```

### 4.3 Model Conversion to TFLite

**Task Checklist:**
- [ ] Convert Keras model to TFLite
- [ ] Apply quantization (INT8)
- [ ] Generate C header file
- [ ] Verify model size (<50KB)

**Conversion Script (convert_to_tflite.py):**
```python
import tensorflow as tf
import numpy as np
import pickle

def representative_dataset_gen():
    X_train = np.loadtxt('datasets/UCI HAR Dataset/train/X_train.txt')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_train_scaled = scaler.transform(X_train)

    for i in range(min(100, len(X_train_scaled))):
        yield [X_train_scaled[i:i+1].astype(np.float32)]

# Load model
model = tf.keras.models.load_model('models/har_model.h5')

# Convert with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save
with open('models/har_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
```

**Generate Header (generate_header.py):**
```python
import pickle

# Load model and scaler
with open('models/har_model.tflite', 'rb') as f:
    tflite_model = f.read()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Generate header
with open('firmware/har_model.h', 'w') as f:
    f.write("#ifndef HAR_MODEL_H\n#define HAR_MODEL_H\n\n")

    # Model array
    f.write("const unsigned char har_model_tflite[] = {\n")
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        f.write(f"  {hex_str},\n")
    f.write("};\n")
    f.write(f"const unsigned int har_model_tflite_len = {len(tflite_model)};\n\n")

    # Normalization parameters
    f.write("const float feature_means[561] = {\n")
    for i in range(0, 561, 8):
        chunk = scaler.mean_[i:i+8]
        f.write("  " + ", ".join([f"{v:.6f}" for v in chunk]) + ",\n")
    f.write("};\n\n")

    f.write("const float feature_stds[561] = {\n")
    for i in range(0, 561, 8):
        chunk = scaler.scale_[i:i+8]
        f.write("  " + ", ".join([f"{v:.6f}" for v in chunk]) + ",\n")
    f.write("};\n\n")

    f.write("#endif\n")

print("Generated firmware/har_model.h")
```

---

## 5. Phase 2: ESP32 Implementation

### 5.1 Firmware Structure

**Task Checklist:**
- [ ] Create Arduino project structure
- [ ] Implement I2C sensor drivers
- [ ] Integrate TFLite Micro
- [ ] Implement feature extraction
- [ ] Setup WiFi AP and web server
- [ ] Test on hardware

**Project Structure:**
```
firmware/
├── TinyHAR_ESP32/
│   ├── TinyHAR_ESP32.ino       # Main Arduino sketch
│   ├── config.h                # Configuration
│   ├── sensors.h               # Sensor drivers
│   ├── features.h              # Feature extraction
│   ├── inference.h             # TFLite inference
│   ├── wifi_server.h           # WiFi and HTTP
│   └── har_model.h             # Model data
└── README.md
```

### 5.2 Main Sketch Template

**TinyHAR_ESP32.ino:**
```cpp
#include "config.h"
#include "sensors.h"
#include "features.h"
#include "inference.h"
#include "wifi_server.h"

void setup() {
  Serial.begin(115200);
  Serial.println("TinyHAR ESP32 Starting...");

  // Initialize I2C sensors
  initSensors();

  // Calibrate sensors
  calibrateSensors();

  // Initialize TFLite
  setupTFLite();

  // Setup WiFi AP
  setupWiFiAP();

  Serial.println("System ready!");
}

void loop() {
  // Read sensors at 20Hz
  readSensors();

  // Add to buffer
  addToBuffer();

  // Check if buffer is full (64 samples = 3.2 seconds)
  if (bufferFull()) {
    // Extract features
    extractFeatures();

    // Run inference
    int activity = runInference();

    // Update display/output
    updateActivity(activity);
  }

  // Handle HTTP requests
  server.handleClient();

  // Maintain 20Hz timing
  delay(50);
}
```

### 5.3 Sensor Driver Implementation

**sensors.h:**
```cpp
#ifndef SENSORS_H
#define SENSORS_H

#include <Wire.h>

// I2C addresses
#define ADXL345_ADDR 0x53
#define ITG3200_ADDR 0x68
#define HMC5883L_ADDR 0x1E

// Global sensor data
float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;
float mag_x, mag_y, mag_z;

// Calibration offsets
float accel_offset_x = 0, accel_offset_y = 0, accel_offset_z = 0;
float gyro_bias_x = 0, gyro_bias_y = 0, gyro_bias_z = 0;

void initSensors() {
  Wire.begin(21, 22);
  Wire.setClock(100000);

  // Initialize ADXL345
  writeRegister(ADXL345_ADDR, 0x2D, 0x08);  // Measurement mode
  writeRegister(ADXL345_ADDR, 0x31, 0x0B);  // ±16g, full resolution

  // Initialize ITG3200
  writeRegister(ITG3200_ADDR, 0x3E, 0x00);  // Wake up
  writeRegister(ITG3200_ADDR, 0x15, 49);    // 20Hz sample rate
  writeRegister(ITG3200_ADDR, 0x16, 0x1B);  // ±2000°/s

  // Initialize HMC5883L
  writeRegister(HMC5883L_ADDR, 0x00, 0x70);  // 8 samples, 15Hz
  writeRegister(HMC5883L_ADDR, 0x01, 0x20);  // ±1.3 Ga
  writeRegister(HMC5883L_ADDR, 0x02, 0x00);  // Continuous mode

  Serial.println("Sensors initialized");
}

void readSensors() {
  // Read ADXL345
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x32);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 6);

  int16_t ax = Wire.read() | (Wire.read() << 8);
  int16_t ay = Wire.read() | (Wire.read() << 8);
  int16_t az = Wire.read() | (Wire.read() << 8);

  accel_x = (ax * 0.0039 * 9.81) - accel_offset_x;
  accel_y = (ay * 0.0039 * 9.81) - accel_offset_y;
  accel_z = (az * 0.0039 * 9.81) - accel_offset_z;

  // Read ITG3200
  Wire.beginTransmission(ITG3200_ADDR);
  Wire.write(0x1D);
  Wire.endTransmission(false);
  Wire.requestFrom(ITG3200_ADDR, 6);

  int16_t gx = (Wire.read() << 8) | Wire.read();
  int16_t gy = (Wire.read() << 8) | Wire.read();
  int16_t gz = (Wire.read() << 8) | Wire.read();

  gyro_x = ((gx / 14.375) * (PI / 180.0)) - gyro_bias_x;
  gyro_y = ((gy / 14.375) * (PI / 180.0)) - gyro_bias_y;
  gyro_z = ((gz / 14.375) * (PI / 180.0)) - gyro_bias_z;
}

void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

#endif
```

---

## 6. Phase 3: Testing and Validation

### 6.1 Unit Testing

**Test Checklist:**
- [ ] I2C communication test
- [ ] Sensor reading accuracy
- [ ] Feature extraction correctness
- [ ] Model inference speed
- [ ] WiFi connectivity
- [ ] HTTP API responses

### 6.2 Integration Testing

**Test Scenarios:**
1. **Walking Test:** Walk for 30 seconds, verify classification
2. **Sitting Test:** Sit still for 30 seconds
3. **Stairs Test:** Walk up/down stairs
4. **Transition Test:** Switch between activities
5. **Power Test:** Measure battery consumption

### 6.3 Performance Metrics

**Expected Results:**
- Accuracy: >90% on UCI HAR test set
- Inference time: <50ms per prediction
- Sample rate: 20Hz stable
- Power consumption: <100mA average
- WiFi range: >10 meters

---

## 7. Troubleshooting Guide

### 7.1 Hardware Issues

**Problem: I2C devices not found**
- Check wiring connections
- Verify power supply (3.3V or 5V)
- Try different I2C pins
- Add external pull-up resistors (4.7kΩ)

**Problem: Sensor readings are noisy**
- Shorten I2C wires
- Add decoupling capacitors
- Reduce I2C clock speed
- Implement software filtering

### 7.2 Software Issues

**Problem: Compilation errors**
- Update ESP32 board package
- Install required libraries
- Check Arduino IDE version
- Verify file paths

**Problem: Out of memory**
- Reduce tensor arena size
- Disable unused features
- Use PROGMEM for constants
- Optimize buffer sizes

### 7.3 Model Issues

**Problem: Low accuracy**
- Verify feature extraction matches training
- Check normalization parameters
- Ensure correct sensor calibration
- Retrain model with more data

**Problem: Slow inference**
- Enable quantization
- Reduce model size
- Increase CPU frequency
- Optimize feature extraction

---

## Appendix A: Quick Reference Commands

**Download Dataset:**
```bash
python3 scripts/download_dataset.py
```

**Train Model:**
```bash
python3 scripts/train_model.py
```

**Convert to TFLite:**
```bash
python3 scripts/convert_to_tflite.py
python3 scripts/generate_header.py
```

**Upload to ESP32:**
```bash
# Arduino IDE: Sketch → Upload
# Or use arduino-cli:
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/TinyHAR_ESP32
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/TinyHAR_ESP32
```

**Monitor Serial:**
```bash
arduino-cli monitor -p /dev/ttyUSB0 -c baudrate=115200
```

---

**Document Status:** Complete
**Next Steps:** Follow phases 1-3 sequentially
**Estimated Time:** 2-3 weeks for full implementation

**Maintained by:** TinyHAR Project Team

