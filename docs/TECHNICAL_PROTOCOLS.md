# Technical Protocols and Implementation Guide

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Purpose:** Detailed technical protocols for ESP32, sensors, TFLite Micro, and system integration

---

## Table of Contents
1. [ESP32 I2C Communication with GY85](#1-esp32-i2c-communication-with-gy85)
2. [TensorFlow Lite Micro Integration](#2-tensorflow-lite-micro-integration)
3. [WiFi Access Point and HTTP Server](#3-wifi-access-point-and-http-server)
4. [Model Conversion Pipeline](#4-model-conversion-pipeline)
5. [Power Optimization Techniques](#5-power-optimization-techniques)
6. [Data Collection Protocol](#6-data-collection-protocol)

---

## 1. ESP32 I2C Communication with GY85

### 1.1 Hardware Connections

**GY85 Module Pinout:**
```
GY85 Pin    →    ESP32 Pin    →    Function
VCC         →    3.3V or 5V   →    Power supply
GND         →    GND          →    Ground
SDA         →    GPIO 21      →    I2C Data
SCL         →    GPIO 22      →    I2C Clock
```

**Important Notes:**
- ⚠️ ESP32 GPIO pins are 3.3V tolerant
- ✅ GY85 works with both 3.3V and 5V
- ✅ Internal pull-up resistors usually sufficient (4.7kΩ on GY85 board)
- ⚠️ Keep I2C wires short (<30cm) to avoid noise

### 1.2 I2C Sensor Addresses

**GY85 contains 4 sensors with different I2C addresses:**

| Sensor | Function | I2C Address | Alternative Address |
|--------|----------|-------------|---------------------|
| ADXL345 | Accelerometer | 0x53 | 0x1D (if SDO high) |
| ITG3200/ITG3205 | Gyroscope | 0x68 | 0x69 (if AD0 high) |
| HMC5883L | Magnetometer | 0x1E | N/A |
| BMP085 | Pressure/Temp | 0x77 | N/A |

### 1.3 ESP32 I2C Initialization

**Arduino Code:**
```cpp
#include <Wire.h>

// I2C Configuration
#define SDA_PIN 21
#define SCL_PIN 22
#define I2C_FREQ 100000  // 100kHz standard mode

void setup() {
  Serial.begin(115200);
  
  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_FREQ);
  
  // Scan for I2C devices
  scanI2CDevices();
  
  // Initialize sensors
  initADXL345();
  initITG3200();
  initHMC5883L();
}

void scanI2CDevices() {
  Serial.println("Scanning I2C bus...");
  byte count = 0;
  
  for (byte i = 1; i < 127; i++) {
    Wire.beginTransmission(i);
    if (Wire.endTransmission() == 0) {
      Serial.print("Found device at 0x");
      Serial.println(i, HEX);
      count++;
    }
  }
  Serial.printf("Found %d devices\n", count);
}
```

### 1.4 ADXL345 Accelerometer Protocol

**Initialization:**
```cpp
#define ADXL345_ADDR 0x53
#define ADXL345_POWER_CTL 0x2D
#define ADXL345_DATA_FORMAT 0x31
#define ADXL345_DATAX0 0x32

void initADXL345() {
  // Set to measurement mode
  writeRegister(ADXL345_ADDR, ADXL345_POWER_CTL, 0x08);
  
  // Set range to ±16g, full resolution
  writeRegister(ADXL345_ADDR, ADXL345_DATA_FORMAT, 0x0B);
  
  delay(10);
  Serial.println("ADXL345 initialized");
}

void readADXL345(float &ax, float &ay, float &az) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(ADXL345_DATAX0);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 6);
  
  int16_t x = Wire.read() | (Wire.read() << 8);
  int16_t y = Wire.read() | (Wire.read() << 8);
  int16_t z = Wire.read() | (Wire.read() << 8);
  
  // Convert to m/s² (scale factor: 0.0039 g/LSB for ±16g)
  ax = x * 0.0039 * 9.81;
  ay = y * 0.0039 * 9.81;
  az = z * 0.0039 * 9.81;
}
```

**ADXL345 Key Registers:**
- `0x00`: DEVID (should read 0xE5)
- `0x2D`: POWER_CTL (0x08 = measurement mode)
- `0x31`: DATA_FORMAT (0x0B = ±16g, full resolution)
- `0x32-0x37`: DATAX0-DATAZ1 (6 bytes, X/Y/Z acceleration)

### 1.5 ITG3200 Gyroscope Protocol

**Initialization:**
```cpp
#define ITG3200_ADDR 0x68
#define ITG3200_PWR_MGM 0x3E
#define ITG3200_SMPLRT_DIV 0x15
#define ITG3200_DLPF_FS 0x16
#define ITG3200_GYRO_XOUT_H 0x1D

void initITG3200() {
  // Wake up (exit sleep mode)
  writeRegister(ITG3200_ADDR, ITG3200_PWR_MGM, 0x00);
  
  // Set sample rate divider (1kHz / (1 + 49) = 20Hz)
  writeRegister(ITG3200_ADDR, ITG3200_SMPLRT_DIV, 49);
  
  // Set full scale range ±2000°/s, DLPF 42Hz
  writeRegister(ITG3200_ADDR, ITG3200_DLPF_FS, 0x1B);
  
  delay(10);
  Serial.println("ITG3200 initialized");
}

void readITG3200(float &gx, float &gy, float &gz) {
  Wire.beginTransmission(ITG3200_ADDR);
  Wire.write(ITG3200_GYRO_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(ITG3200_ADDR, 6);
  
  int16_t x = (Wire.read() << 8) | Wire.read();
  int16_t y = (Wire.read() << 8) | Wire.read();
  int16_t z = (Wire.read() << 8) | Wire.read();
  
  // Convert to rad/s (scale factor: 14.375 LSB/(°/s))
  gx = (x / 14.375) * (PI / 180.0);
  gy = (y / 14.375) * (PI / 180.0);
  gz = (z / 14.375) * (PI / 180.0);
}
```

**ITG3200 Key Registers:**
- `0x00`: WHO_AM_I (should read 0x68 or 0x69)
- `0x15`: SMPLRT_DIV (sample rate divider)
- `0x16`: DLPF_FS (digital low-pass filter and full scale)
- `0x1D-0x22`: GYRO_XOUT_H to GYRO_ZOUT_L (6 bytes)
- `0x3E`: PWR_MGM (power management)

### 1.6 HMC5883L Magnetometer Protocol

**Initialization:**
```cpp
#define HMC5883L_ADDR 0x1E
#define HMC5883L_CONFIG_A 0x00
#define HMC5883L_CONFIG_B 0x01
#define HMC5883L_MODE 0x02
#define HMC5883L_DATA_X_MSB 0x03

void initHMC5883L() {
  // Set to 8 samples averaged, 15Hz output rate
  writeRegister(HMC5883L_ADDR, HMC5883L_CONFIG_A, 0x70);
  
  // Set gain to ±1.3 Ga
  writeRegister(HMC5883L_ADDR, HMC5883L_CONFIG_B, 0x20);
  
  // Set to continuous measurement mode
  writeRegister(HMC5883L_ADDR, HMC5883L_MODE, 0x00);
  
  delay(10);
  Serial.println("HMC5883L initialized");
}

void readHMC5883L(float &mx, float &my, float &mz) {
  Wire.beginTransmission(HMC5883L_ADDR);
  Wire.write(HMC5883L_DATA_X_MSB);
  Wire.endTransmission(false);
  Wire.requestFrom(HMC5883L_ADDR, 6);
  
  // Note: HMC5883L order is X, Z, Y
  int16_t x = (Wire.read() << 8) | Wire.read();
  int16_t z = (Wire.read() << 8) | Wire.read();
  int16_t y = (Wire.read() << 8) | Wire.read();
  
  // Convert to µT (scale factor: 0.92 mGa/LSB for ±1.3Ga)
  mx = x * 0.92;
  my = y * 0.92;
  mz = z * 0.92;
}
```

### 1.7 Calibration Procedure

**Accelerometer Calibration:**
```cpp
void calibrateAccelerometer(int samples = 100) {
  float ax_sum = 0, ay_sum = 0, az_sum = 0;
  
  Serial.println("Calibrating accelerometer (keep device still)...");
  
  for (int i = 0; i < samples; i++) {
    float ax, ay, az;
    readADXL345(ax, ay, az);
    ax_sum += ax;
    ay_sum += ay;
    az_sum += az;
    delay(10);
  }
  
  // Calculate offsets (assuming Z-axis = 1g when upright)
  accel_offset_x = ax_sum / samples;
  accel_offset_y = ay_sum / samples;
  accel_offset_z = (az_sum / samples) - 9.81;  // Remove gravity
  
  Serial.printf("Accel offsets: X=%.3f, Y=%.3f, Z=%.3f\n", 
                accel_offset_x, accel_offset_y, accel_offset_z);
}
```

**Gyroscope Calibration:**
```cpp
void calibrateGyroscope(int samples = 100) {
  float gx_sum = 0, gy_sum = 0, gz_sum = 0;
  
  Serial.println("Calibrating gyroscope (keep device still)...");
  
  for (int i = 0; i < samples; i++) {
    float gx, gy, gz;
    readITG3200(gx, gy, gz);
    gx_sum += gx;
    gy_sum += gy;
    gz_sum += gz;
    delay(10);
  }
  
  // Calculate bias
  gyro_bias_x = gx_sum / samples;
  gyro_bias_y = gy_sum / samples;
  gyro_bias_z = gz_sum / samples;
  
  Serial.printf("Gyro bias: X=%.4f, Y=%.4f, Z=%.4f\n", 
                gyro_bias_x, gyro_bias_y, gyro_bias_z);
}
```

### 1.8 Helper Functions

```cpp
void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.read();
}
```

---

## 2. TensorFlow Lite Micro Integration

### 2.1 Library Installation

**Arduino IDE:**
```
1. Open Arduino IDE
2. Go to Sketch → Include Library → Manage Libraries
3. Search "TensorFlowLite_ESP32"
4. Install "TensorFlowLite_ESP32" by tanakamasayuki
```

**PlatformIO:**
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps = 
    tanakamasayuki/TensorFlowLite_ESP32@^0.9.0
    bblanchon/ArduinoJson@^6.21.0
```

### 2.2 Model Header File Structure

**har_model.h:**
```cpp
#ifndef HAR_MODEL_H
#define HAR_MODEL_H

// TFLite model as byte array
const unsigned char har_model_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
  // ... (model bytes)
};
const unsigned int har_model_tflite_len = 44700;  // ~44.7KB

// Activity labels
const char* activity_labels[] = {
  "WALKING",
  "WALKING_UPSTAIRS", 
  "WALKING_DOWNSTAIRS",
  "SITTING",
  "STANDING",
  "LAYING"
};
const int num_activities = 6;

// Feature normalization parameters (561 features)
const float feature_means[561] = { /* ... */ };
const float feature_stds[561] = { /* ... */ };

#endif
```

### 2.3 TFLite Micro Setup

**Include Headers:**
```cpp
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "har_model.h"

// TFLite globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena for model operations
  constexpr int kTensorArenaSize = 60 * 1024;  // 60KB
  uint8_t tensor_arena[kTensorArenaSize];
}
```

**Initialization:**
```cpp
void setupTFLite() {
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  model = tflite::GetModel(har_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model version %d doesn't match schema %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Create ops resolver (use MutableOpResolver for smaller size)
  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLite initialized successfully");
  Serial.printf("Input shape: [%d, %d]\n",
                input->dims->data[0], input->dims->data[1]);
  Serial.printf("Output shape: [%d, %d]\n",
                output->dims->data[0], output->dims->data[1]);
}
```

### 2.4 Inference Pipeline

**Feature Normalization:**
```cpp
void normalizeFeatures(float* features, int num_features) {
  for (int i = 0; i < num_features; i++) {
    features[i] = (features[i] - feature_means[i]) / feature_stds[i];
  }
}
```

**Run Inference:**
```cpp
int runInference(float* features) {
  // Normalize features
  normalizeFeatures(features, 561);

  // Copy to input tensor
  for (int i = 0; i < 561; i++) {
    input->data.f[i] = features[i];
  }

  // Run inference
  unsigned long start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time = micros() - start;

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return -1;
  }

  // Get prediction
  int predicted_class = 0;
  float max_prob = output->data.f[0];

  for (int i = 1; i < num_activities; i++) {
    if (output->data.f[i] > max_prob) {
      max_prob = output->data.f[i];
      predicted_class = i;
    }
  }

  Serial.printf("Inference: %s (%.2f%%) in %lu µs\n",
                activity_labels[predicted_class],
                max_prob * 100,
                inference_time);

  return predicted_class;
}
```

---

## 3. WiFi Access Point and HTTP Server

### 3.1 WiFi AP Configuration

**Setup Access Point:**
```cpp
#include <WiFi.h>
#include <WebServer.h>

// AP Configuration
const char* ap_ssid = "ESP32-HAR";
const char* ap_password = "12345678";
IPAddress local_ip(192, 168, 4, 1);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

WebServer server(80);

void setupWiFiAP() {
  // Configure AP
  WiFi.softAPConfig(local_ip, gateway, subnet);
  WiFi.softAP(ap_ssid, ap_password);

  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  // Setup HTTP routes
  server.on("/", handleRoot);
  server.on("/api/sensor", handleSensorData);
  server.on("/api/activity", handleActivity);
  server.on("/api/stats", handleStats);
  server.on("/api/calibration_data", handleCalibrationData);

  server.begin();
  Serial.println("HTTP server started");
}
```

### 3.2 HTTP API Endpoints

**Root Page (Web UI):**
```cpp
void handleRoot() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
  <title>TinyHAR Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial; margin: 20px; }
    .sensor-data { background: #f0f0f0; padding: 10px; margin: 10px 0; }
    .activity { font-size: 24px; font-weight: bold; color: #007bff; }
  </style>
</head>
<body>
  <h1>TinyHAR Activity Monitor</h1>
  <div id="activity" class="activity">Loading...</div>
  <div id="sensor" class="sensor-data">Sensor data...</div>
  <script>
    setInterval(() => {
      fetch('/api/activity')
        .then(r => r.json())
        .then(d => {
          document.getElementById('activity').innerText =
            'Activity: ' + d.activity + ' (' + d.confidence + '%)';
        });
      fetch('/api/sensor')
        .then(r => r.json())
        .then(d => {
          document.getElementById('sensor').innerHTML =
            'Accel: ' + d.ax.toFixed(2) + ', ' + d.ay.toFixed(2) + ', ' + d.az.toFixed(2) + '<br>' +
            'Gyro: ' + d.gx.toFixed(2) + ', ' + d.gy.toFixed(2) + ', ' + d.gz.toFixed(2);
        });
    }, 500);
  </script>
</body>
</html>
  )";
  server.send(200, "text/html", html);
}
```

**Sensor Data Endpoint:**
```cpp
#include <ArduinoJson.h>

void handleSensorData() {
  StaticJsonDocument<256> doc;

  doc["timestamp"] = millis();
  doc["ax"] = current_ax;
  doc["ay"] = current_ay;
  doc["az"] = current_az;
  doc["gx"] = current_gx;
  doc["gy"] = current_gy;
  doc["gz"] = current_gz;
  doc["mx"] = current_mx;
  doc["my"] = current_my;
  doc["mz"] = current_mz;

  String response;
  serializeJson(doc, response);

  server.send(200, "application/json", response);
}
```

**Activity Endpoint:**
```cpp
void handleActivity() {
  StaticJsonDocument<128> doc;

  doc["activity"] = activity_labels[current_activity];
  doc["confidence"] = current_confidence * 100;
  doc["timestamp"] = millis();

  String response;
  serializeJson(doc, response);

  server.send(200, "application/json", response);
}
```

**Stats Endpoint:**
```cpp
void handleStats() {
  StaticJsonDocument<256> doc;

  doc["uptime"] = millis() / 1000;
  doc["free_heap"] = ESP.getFreeHeap();
  doc["inference_time_us"] = last_inference_time;
  doc["sample_rate"] = actual_sample_rate;
  doc["buffer_fill"] = buffer_count;

  String response;
  serializeJson(doc, response);

  server.send(200, "application/json", response);
}
```

### 3.3 CORS Headers (for external access)

```cpp
void setupCORS() {
  server.enableCORS(true);

  // Or manually add headers
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}
```

---

## 4. Model Conversion Pipeline

### 4.1 Python Training Script

**train_model.py:**
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle

# Load UCI HAR data
def load_uci_har_data():
    X_train = np.loadtxt('datasets/UCI_HAR/train/X_train.txt')
    y_train = np.loadtxt('datasets/UCI_HAR/train/y_train.txt')
    X_test = np.loadtxt('datasets/UCI_HAR/test/X_test.txt')
    y_test = np.loadtxt('datasets/UCI_HAR/test/y_test.txt')

    # Convert labels from 1-6 to 0-5
    y_train = y_train - 1
    y_test = y_test - 1

    return X_train, y_train, X_test, y_test

# Normalize features
def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for deployment
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, scaler

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

# Train model
def train_model():
    # Load data
    X_train, y_train, X_test, y_test = load_uci_har_data()

    # Normalize
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # Build model
    model = build_model(X_train.shape[1], 6)
    model.summary()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
    )

    # Train
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model
    model.save('har_model.h5')

    return model, scaler

if __name__ == '__main__':
    model, scaler = train_model()
```

### 4.2 TFLite Conversion Script

**convert_to_tflite.py:**
```python
import tensorflow as tf
import numpy as np
import pickle

def representative_dataset_gen():
    """Generate representative dataset for quantization"""
    X_train = np.loadtxt('datasets/UCI_HAR/train/X_train.txt')

    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X_train_scaled = scaler.transform(X_train)

    # Use subset for calibration
    for i in range(min(100, len(X_train_scaled))):
        yield [X_train_scaled[i:i+1].astype(np.float32)]

def convert_to_tflite():
    # Load Keras model
    model = tf.keras.models.load_model('har_model.h5')

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Full integer quantization
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert
    tflite_model = converter.convert()

    # Save
    with open('har_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

    return tflite_model

if __name__ == '__main__':
    tflite_model = convert_to_tflite()
```

### 4.3 Generate C Header File

**generate_header.py:**
```python
import pickle
import numpy as np

def generate_c_header():
    # Load TFLite model
    with open('har_model.tflite', 'rb') as f:
        tflite_model = f.read()

    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Activity labels
    labels = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
        "SITTING", "STANDING", "LAYING"
    ]

    # Generate header file
    with open('har_model.h', 'w') as f:
        f.write("#ifndef HAR_MODEL_H\n")
        f.write("#define HAR_MODEL_H\n\n")

        # Model array
        f.write("const unsigned char har_model_tflite[] = {\n")
        for i in range(0, len(tflite_model), 12):
            chunk = tflite_model[i:i+12]
            hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
            f.write(f"  {hex_str},\n")
        f.write("};\n")
        f.write(f"const unsigned int har_model_tflite_len = {len(tflite_model)};\n\n")

        # Labels
        f.write("const char* activity_labels[] = {\n")
        for label in labels:
            f.write(f'  "{label}",\n')
        f.write("};\n")
        f.write(f"const int num_activities = {len(labels)};\n\n")

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

    print("Generated har_model.h")

if __name__ == '__main__':
    generate_c_header()
```

---

## 5. Power Optimization Techniques

### 5.1 Deep Sleep Mode

**Basic Deep Sleep:**
```cpp
#define uS_TO_S_FACTOR 1000000ULL
#define TIME_TO_SLEEP 60  // seconds

void enterDeepSleep() {
  Serial.println("Entering deep sleep...");

  // Configure wakeup
  esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * uS_TO_S_FACTOR);

  // Enter deep sleep
  esp_deep_sleep_start();
}
```

**GPIO Wakeup:**
```cpp
#define BUTTON_PIN 0

void setupGPIOWakeup() {
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_0, 0);  // Wake on LOW
}
```

### 5.2 Light Sleep Mode

```cpp
void enterLightSleep(int seconds) {
  esp_sleep_enable_timer_wakeup(seconds * uS_TO_S_FACTOR);
  esp_light_sleep_start();
}
```

### 5.3 CPU Frequency Scaling

```cpp
void setCPUFrequency(uint32_t freq_mhz) {
  // Options: 240, 160, 80, 40, 20, 10 MHz
  setCpuFrequencyMhz(freq_mhz);
  Serial.printf("CPU frequency set to %d MHz\n", getCpuFrequencyMhz());
}
```

### 5.4 WiFi Power Management

```cpp
void disableWiFi() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  Serial.println("WiFi disabled");
}

void enableWiFi() {
  WiFi.mode(WIFI_AP);
  setupWiFiAP();
  Serial.println("WiFi enabled");
}
```

### 5.5 Adaptive Sampling

```cpp
int adaptive_sample_rate = 20;  // Hz

void adjustSampleRate(int activity) {
  if (activity == SITTING || activity == STANDING) {
    adaptive_sample_rate = 5;  // Reduce to 5Hz for static activities
  } else {
    adaptive_sample_rate = 20;  // Full 20Hz for dynamic activities
  }
}
```

---

## 6. Data Collection Protocol

### 6.1 Python Data Collector

**data_collector.py:**
```python
import requests
import csv
import time
from datetime import datetime

class DataCollector:
    def __init__(self, esp32_ip='192.168.4.1'):
        self.base_url = f'http://{esp32_ip}'
        self.session = requests.Session()
        self.session.timeout = 2

    def collect_sample(self):
        """Collect single sensor sample"""
        try:
            response = self.session.get(f'{self.base_url}/api/sensor')
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error: {e}")
        return None

    def collect_session(self, activity_label, duration_sec=30, filename=None):
        """Collect data for specified duration"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data_{activity_label}_{timestamp}.csv'

        samples = []
        start_time = time.time()
        target_rate = 20  # Hz
        interval = 1.0 / target_rate

        print(f"Collecting {activity_label} for {duration_sec}s...")

        while time.time() - start_time < duration_sec:
            loop_start = time.time()

            sample = self.collect_sample()
            if sample:
                sample['label'] = activity_label
                sample['local_time'] = time.time()
                samples.append(sample)
                print(f"\rSamples: {len(samples)}", end='')

            # Maintain timing
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        print(f"\nCollected {len(samples)} samples")

        # Save to CSV
        self.save_to_csv(samples, filename)
        return samples

    def save_to_csv(self, samples, filename):
        """Save samples to CSV file"""
        if not samples:
            return

        keys = samples[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(samples)

        print(f"Saved to {filename}")

# Usage
if __name__ == '__main__':
    collector = DataCollector()

    activities = ['walking', 'sitting', 'standing', 'laying']

    for activity in activities:
        input(f"Press Enter to start collecting {activity}...")
        collector.collect_session(activity, duration_sec=30)
        print("Done!\n")
```

---

**Document Status:** Complete - Part 1
**Next:** Continue with advanced topics in separate sections

**Maintained by:** TinyHAR Project Team
