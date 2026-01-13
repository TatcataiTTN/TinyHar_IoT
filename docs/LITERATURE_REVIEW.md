# Literature Review: Edge-Based Human Activity Recognition

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Purpose:** Comprehensive review of research papers, methods, and datasets for TinyHAR project replication

---

## Table of Contents
1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Datasets Comparison](#datasets-comparison)
4. [Methods and Algorithms](#methods-and-algorithms)
5. [Edge Deployment Strategies](#edge-deployment-strategies)
6. [Research Gaps and Opportunities](#research-gaps-and-opportunities)
7. [References](#references)

---

## 1. Introduction

Human Activity Recognition (HAR) using wearable sensors has become a critical research area with applications in healthcare, fitness tracking, elderly care, and smart environments. Traditional cloud-based HAR systems face challenges including:

- **Privacy concerns**: Sensitive personal data transmitted to external servers
- **Latency issues**: Network delays affect real-time applications
- **Energy consumption**: Continuous wireless transmission drains battery
- **Connectivity dependency**: Requires stable internet connection

**Edge computing** addresses these limitations by processing data locally on resource-constrained devices like microcontrollers. This literature review focuses on:
- Edge-based HAR implementations
- TensorFlow Lite Micro on ESP32/microcontrollers
- Wearable IMU sensor-based activity recognition
- Model optimization techniques (quantization, pruning)
- Sensor fusion algorithms

---

## 2. Related Work

### 2.1 Edge-Based HAR Systems

#### 2.1.1 Microcontroller-Based Implementations

**Key Papers:**

1. **"Human Activity Recognition on Microcontrollers with Quantized and Adaptive Deep Learning Models"** (ACM, 2022)
   - Authors: Multiple contributors
   - DOI: 10.1145/3542819
   - Key Findings: Demonstrates feasibility of deep learning HAR on microcontrollers using quantization
   - Relevance: Direct application to ESP32-based systems
   - Link: https://dl.acm.org/doi/full/10.1145/3542819

2. **"TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Human Activity Recognition"** (arXiv, 2025)
   - Authors: Research team
   - arXiv: 2507.07949v1
   - Key Findings: Ultra-lightweight models achieving high accuracy with minimal resources
   - Relevance: Model architecture optimization strategies
   - Link: https://arxiv.org/html/2507.07949v1

3. **"Towards Generalizable Human Activity Recognition"** (arXiv, 2025)
   - arXiv: 2508.12213v1
   - Key Findings: Survey on generalization challenges in HAR
   - Relevance: Cross-domain adaptation and robustness
   - Link: https://arxiv.org/html/2508.12213v1

4. **"Improving Out-of-distribution Human Activity Recognition via IMU-Video Cross-modal Self-supervision"** (arXiv, 2025)
   - arXiv: 2507.13482v1
   - Key Findings: Cross-modal learning for improved generalization
   - Relevance: Handling distribution shifts in real-world deployment
   - Link: https://arxiv.org/pdf/2507.13482

### 2.2 Model Optimization for Embedded Systems

#### 2.2.1 Quantization Techniques

**Key Papers:**

1. **"Quantized Neural Networks for Microcontrollers: A Comprehensive Survey"** (arXiv, 2025)
   - arXiv: 2508.15008v1
   - Key Findings: Hardware-centric survey on quantization methods
   - Techniques: Post-training quantization (PTQ), Quantization-aware training (QAT)
   - Relevance: Essential for ESP32 deployment (32-bit → 8-bit conversion)
   - Link: https://arxiv.org/html/2508.15008v1

2. **"Quantization and Deployment of Deep Neural Networks on Microcontrollers"** (MDPI Sensors, 2021)
   - DOI: 10.3390/s21092984
   - Key Findings: Practical deployment strategies with power consumption analysis
   - Relevance: Complete pipeline from training to embedded deployment
   - Link: https://www.mdpi.com/1424-8220/21/9/2984

3. **"Efficient Human Activity Recognition on Edge Devices Using Quantization"** (Nature, 2025)
   - Key Findings: Full integer quantization for microcontrollers
   - Results: 78% size reduction with <2% accuracy loss
   - Relevance: Directly applicable to TinyHAR optimization goals
   - Link: https://www.nature.com/articles/s41598-025-98571-2

#### 2.2.2 Model Compression and Pruning

**Key Papers:**

1. **"Emerging Trends and Strategic Opportunities in Tiny Machine Learning"** (ScienceDirect, 2025)
   - Key Findings: Comprehensive overview of TinyML optimization methods
   - Techniques: Quantization, pruning, clustering, knowledge distillation
   - Relevance: Multi-strategy optimization approach
   - Link: https://www.sciencedirect.com/science/article/pii/S0925231225014183

2. **"A Comprehensive Survey on Tiny Machine Learning for Human Activity Recognition"** (IEEE, 2025)
   - Key Findings: 16-bit quantization for HAR models on resource-limited devices
   - Relevance: Balancing accuracy and resource constraints
   - Link: https://ieeexplore.ieee.org/iel8/6488907/11121332/10979983.pdf

### 2.3 Wearable Sensor-Based HAR

#### 2.3.1 IMU Sensor Fusion

**Key Papers:**

1. **"A Multi-Channel Hybrid Deep Learning Framework for Multi-Sensor HAR"** (ScienceDirect, 2024)
   - Key Findings: Fusion of accelerometer, gyroscope, and magnetometer data
   - Architecture: Multi-channel CNN for sensor fusion
   - Relevance: GY85 sensor module integration strategy
   - Link: https://www.sciencedirect.com/science/article/pii/S1110016824000425

2. **"Sensor Data Acquisition and Multimodal Sensor Fusion for HAR"** (MDPI Sensors, 2019)
   - DOI: 10.3390/s19071716
   - Key Findings: Systematic comparison of sensor modalities
   - Relevance: Optimal sensor selection and fusion strategies
   - Link: https://www.mdpi.com/1424-8220/19/7/1716

3. **"An Enhanced Human Activity Recognition Algorithm with Positional Awareness"** (PMLR, 2023)
   - Key Findings: IMU data processing with accelerometer, gyroscope, magnetometer
   - Relevance: Feature extraction from multi-sensor data
   - Link: https://proceedings.mlr.press/v189/xu23a/xu23a.pdf

#### 2.3.2 Deep Learning Architectures for HAR

**Key Papers:**

1. **"WISNet: A Deep Neural Network Based HAR System"** (ScienceDirect, 2024)
   - Evaluation: WISDM, UCI-HAR 2012, OPPORTUNITY, PAMAP2 datasets
   - Architecture: Deep neural network optimized for wearable sensors
   - Relevance: Benchmark comparison across multiple datasets
   - Link: https://www.sciencedirect.com/science/article/pii/S0957417424018669

2. **"An Enhanced Dual Inception-Attention-BiGRU-Attention Model for HAR"** (Nature, 2025)
   - Datasets: PAMAP2, UCI-HAR, WISDM
   - Architecture: Hybrid CNN-RNN with attention mechanisms
   - Relevance: State-of-the-art accuracy benchmarks
   - Link: https://www.nature.com/articles/s41598-025-32859-1

3. **"Convolutional Attention-Based Bidirectional Recurrent Neural Network for HAR"** (Wiley, 2025)
   - Datasets: WISDM, UCI-HAR, PAMAP2
   - Key Findings: Attention mechanisms improve temporal pattern recognition
   - Relevance: Advanced architecture considerations
   - Link: https://onlinelibrary.wiley.com/doi/10.1111/coin.70049

### 2.4 TensorFlow Lite Micro on ESP32

#### 2.4.1 Implementation Guides and Tutorials

**Key Resources:**

1. **"TinyML with ESP32 Tutorial"** (TeachMeMicro, 2024)
   - Comprehensive guide for TFLite Micro on ESP32
   - Topics: Model conversion, deployment, sensor integration
   - Relevance: Step-by-step implementation reference
   - Link: https://www.teachmemicro.com/tinyml-with-esp32-tutorial/

2. **"Iris Species Dataset Classification Model in ESP32 + TensorFlow Lite Micro"** (Medium, 2024)
   - Practical example with HTTP server integration
   - Code: Complete working implementation
   - Relevance: Template for ESP32 + TFLite + Web Server
   - Link: https://medium.com/@eduardo.bl/iris-dataset-classification-model-in-esp32-tensorflow-lite-micro-http-server-5aa5a66f7543

3. **"TensorFlow Lite Micro GitHub Repository"**
   - Official TFLite Micro implementation
   - Platform: DSPs, microcontrollers, embedded devices
   - Relevance: Core library and examples
   - Link: https://github.com/tensorflow/tflite-micro

#### 2.4.2 Model Conversion Pipeline

**Key Resources:**

1. **"Embedded AI Systems Part 11: Porting TensorFlow Model to Embedded Device"** (Medium, 2024)
   - Process: TFLite model → C header file conversion
   - Tools: Bash scripts and Python utilities
   - Relevance: Automated conversion pipeline
   - Link: https://medium.com/@johnos3747/embedded-ai-systems-part-11-ebd18aceb4cf

2. **"Deploying CNNs on Microcontrollers — A TinyML Blog"** (Medium, 2024)
   - Complete workflow: Keras → TFLite → C header
   - Optimization: Quantization and model compression
   - Relevance: End-to-end deployment guide
   - Link: https://nathanbaileyw.medium.com/deploying-convolutional-neural-networks-on-microcontrollers-a-tinyml-blog-5f9b4fa37864

---

## 3. Datasets Comparison

### 3.1 Primary HAR Datasets

| Dataset | Source | Subjects | Activities | Sensors | Sampling Rate | Size | Format | License |
|---------|--------|----------|------------|---------|---------------|------|--------|---------|
| **UCI HAR** | UCI ML Repository | 30 | 6 | Acc + Gyro | 50Hz | ~60MB | TXT/CSV | Open |
| **WISDM** | Fordham University | 36 | 6 | Accelerometer | 20Hz | ~40MB | CSV | Open |
| **PAMAP2** | UCI ML Repository | 9 | 18 | 3x IMU (Acc+Gyro+Mag) | 100Hz | ~500MB | TXT | Open |
| **MotionSense** | Kaggle | 24 | 6 | Acc + Gyro | 50Hz | ~100MB | CSV | Open |
| **HuGaDB** | Kaggle | Multiple | Gait activities | IMU | Variable | ~200MB | CSV | Open |

### 3.2 Dataset Details

#### 3.2.1 UCI HAR Dataset (Primary - REQUIRED)

**Full Name:** Human Activity Recognition Using Smartphones Dataset

**Citation:**
```
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
A Public Domain Dataset for Human Activity Recognition Using Smartphones.
21th European Symposium on Artificial Neural Networks, Computational Intelligence
and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
```

**Details:**
- **Subjects:** 30 volunteers (19-48 years old)
- **Activities:** 6 classes
  1. Walking
  2. Walking Upstairs
  3. Walking Downstairs
  4. Sitting
  5. Standing
  6. Laying
- **Sensors:** Accelerometer + Gyroscope (smartphone-based)
- **Sampling Rate:** 50Hz
- **Features:** 561 pre-computed features (time and frequency domain)
- **Train/Test Split:** 70%/30% (subject-wise split)
- **Download:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Kaggle:** https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

**Why UCI HAR is Essential:**
- Industry-standard benchmark for HAR research
- Pre-computed 561 features align with TinyHAR architecture
- 50Hz sampling rate (TinyHAR uses 20Hz - downsampling compatible)
- Extensive research baseline for comparison
- Well-documented and widely replicated

#### 3.2.2 WISDM Dataset (Recommended)

**Full Name:** Wireless Sensor Data Mining Activity Prediction Dataset

**Details:**
- **Subjects:** 36 users
- **Activities:** 6 classes (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
- **Sensors:** Accelerometer (smartphone in pocket)
- **Sampling Rate:** 20Hz (matches TinyHAR exactly!)
- **Format:** CSV with timestamp, user, activity, x, y, z
- **Size:** ~40MB
- **Download:** https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **Advantage:** Same 20Hz sampling rate as TinyHAR implementation

#### 3.2.3 PAMAP2 Dataset (Advanced)

**Full Name:** Physical Activity Monitoring Dataset

**Details:**
- **Subjects:** 9 participants
- **Activities:** 18 different physical activities
- **Sensors:** 3 IMU units (hand, chest, ankle) - each with Acc+Gyro+Mag
- **Sampling Rate:** 100Hz
- **Format:** TXT files
- **Size:** ~500MB
- **Download:** https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- **Advantage:** Rich multi-sensor data, includes magnetometer (like GY85)

#### 3.2.4 MotionSense Dataset

**Details:**
- **Subjects:** 24 participants
- **Activities:** 6 classes
- **Sensors:** Accelerometer + Gyroscope (smartphone)
- **Sampling Rate:** 50Hz
- **Format:** CSV
- **Download:** https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset

#### 3.2.5 HuGaDB - Human Gait Database

**Details:**
- **Focus:** Gait analysis and walking patterns
- **Sensors:** Wearable IMU sensors
- **Format:** CSV
- **Download:** https://www.kaggle.com/datasets/romanchereshnev/hugadb-human-gait-database
- **Use Case:** Specialized for walking/gait activities

### 3.3 Dataset Selection Recommendations

**For TinyHAR Replication:**

1. **Primary (Required):** UCI HAR
   - Reason: Baseline comparison, 561 features, standard benchmark

2. **Secondary (Recommended):** WISDM
   - Reason: 20Hz sampling rate matches implementation exactly

3. **Tertiary (Optional):** PAMAP2
   - Reason: Multi-sensor data including magnetometer, more activities

---

## 4. Methods and Algorithms

### 4.1 Feature Extraction Techniques

#### 4.1.1 Time-Domain Features
- **Statistical measures:** Mean, variance, standard deviation, min, max
- **Signal characteristics:** Zero-crossing rate, signal magnitude area
- **Correlation features:** Inter-axis correlation coefficients

#### 4.1.2 Frequency-Domain Features
- **FFT-based:** Dominant frequency, spectral energy, power spectral density
- **Wavelet transforms:** Multi-resolution time-frequency analysis

#### 4.1.3 UCI HAR Feature Set (561 Features)
The UCI HAR dataset provides pre-computed features:
- Time-domain: 128 features per signal
- Frequency-domain: FFT coefficients
- Statistical: Mean, std, mad, max, min, energy, entropy, etc.
- **Advantage:** Ready for neural network input without manual feature engineering

### 4.2 Machine Learning Architectures

#### 4.2.1 Traditional ML Methods
- **Support Vector Machines (SVM):** Good for small datasets, interpretable
- **Random Forest:** Robust, handles non-linear relationships
- **k-Nearest Neighbors (k-NN):** Simple, no training phase

#### 4.2.2 Deep Learning Architectures

**For TinyHAR:**
```
Input Layer: 561 features
Hidden Layer 1: 64 neurons (ReLU) + Dropout(0.3)
Hidden Layer 2: 32 neurons (ReLU) + Dropout(0.3)
Hidden Layer 3: 16 neurons (ReLU)
Output Layer: 6 neurons (Softmax)
Total Parameters: ~38.6K
```

**Optimization:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch Size: 32
- Early Stopping: Patience=10
- Learning Rate Reduction: Factor=0.2, Patience=5

**Alternative Architectures:**
- **CNN-based:** 1D Convolutions for temporal patterns
- **LSTM/GRU:** Recurrent networks for sequence modeling
- **Hybrid CNN-LSTM:** Combines spatial and temporal features
- **Attention mechanisms:** Focus on important time steps

### 4.3 Model Optimization Strategies

#### 4.3.1 Quantization
- **Post-Training Quantization (PTQ):** 32-bit → 8-bit after training
- **Quantization-Aware Training (QAT):** Simulate quantization during training
- **TinyHAR Approach:** PTQ with representative dataset
- **Result:** 78% size reduction (205KB → 44.7KB)

#### 4.3.2 Pruning
- **Structured Pruning:** Remove entire neurons/filters
- **Unstructured Pruning:** Remove individual weights
- **TinyHAR Approach:** 40% parameter pruning
- **Result:** Minimal accuracy loss (<1%)

#### 4.3.3 Knowledge Distillation
- **Teacher Model:** Large, accurate model
- **Student Model:** Small, efficient model
- **Process:** Student learns from teacher's soft predictions
- **Benefit:** Maintain accuracy with smaller model

---

## 5. Edge Deployment Strategies

### 5.1 Hardware Platforms

#### 5.1.1 ESP32 Specifications
- **Processor:** Dual-core Xtensa LX6, 240MHz
- **RAM:** 520KB SRAM
- **Flash:** 4MB (typical)
- **Connectivity:** WiFi, Bluetooth
- **Power:** 80-260mA active, 10μA deep sleep
- **Cost:** $5-10 USD

#### 5.1.2 GY85 Sensor Module
- **ADXL345:** 3-axis accelerometer (±16g)
- **ITG3200/ITG3205:** 3-axis gyroscope (±2000°/s)
- **HMC5883L:** 3-axis magnetometer
- **BMP085:** Barometric pressure + temperature
- **Interface:** I2C (SDA=GPIO21, SCL=GPIO22)
- **Voltage:** 3.3V-5V
- **Cost:** $10-15 USD

### 5.2 Software Stack

#### 5.2.1 Development Environment
- **Arduino IDE:** Version 2.x with ESP32 board support
- **PlatformIO:** Alternative IDE with better library management
- **ESP-IDF:** Official Espressif framework (advanced)

#### 5.2.2 Key Libraries
- **TensorFlow Lite Micro:** ML inference engine
- **Wire.h:** I2C communication
- **WiFi.h:** Network connectivity
- **WebServer.h:** HTTP server
- **ArduinoJson:** JSON parsing/serialization

### 5.3 System Architecture

#### 5.3.1 Data Flow Pipeline
```
Sensor Reading (20Hz) → Calibration → Circular Buffer (64 samples)
→ Feature Extraction → Normalization → ML Inference (1-5Hz)
→ Activity Classification → WiFi Transmission (optional)
```

#### 5.3.2 Memory Management
- **Static Allocation:** Pre-allocate buffers to avoid fragmentation
- **Tensor Arena:** Fixed memory pool for TFLite operations
- **Circular Buffer:** Efficient windowing without memory copies

#### 5.3.3 Power Optimization
- **Adaptive Sampling:** Reduce rate during static activities
- **WiFi Management:** Turn off when not needed
- **Deep Sleep:** Use between activity sessions
- **Sensor Control:** Disable unused sensors (magnetometer, pressure)

### 5.4 Communication Protocols

#### 5.4.1 I2C Communication
- **Clock Speed:** 100kHz (standard mode)
- **Pull-up Resistors:** 4.7kΩ (usually on-board)
- **Address Scanning:** Detect connected sensors
- **Error Handling:** Timeout and retry mechanisms

#### 5.4.2 WiFi Access Point Mode
- **SSID:** ESP32-HAR (configurable)
- **Password:** 12345678 (configurable)
- **IP Address:** 192.168.4.1 (default gateway)
- **DHCP:** Automatic client IP assignment

#### 5.4.3 HTTP API Endpoints
- `GET /api/sensor` - Real-time sensor data (JSON)
- `GET /api/activity` - Current activity classification
- `GET /api/stats` - System statistics (memory, uptime)
- `GET /api/calibration_data` - Bulk data export for training

---

## 6. Research Gaps and Opportunities

### 6.1 Identified Gaps

1. **Population-Specific Models**
   - Most datasets focus on adults (19-48 years)
   - Limited research on children, elderly, disabled populations
   - TinyHAR addresses this with children pilot study

2. **Real-World Deployment Challenges**
   - Lab datasets don't reflect real-world variability
   - Environmental factors (temperature, vibration) understudied
   - Long-term stability and drift not well documented

3. **Privacy-Preserving Data Collection**
   - Edge processing solves inference privacy
   - Training data collection still requires data sharing
   - Need frameworks for federated learning on edge devices

4. **Energy Efficiency**
   - Most papers report accuracy, few analyze power consumption
   - Battery life critical for wearable applications
   - Need comprehensive power profiling methodologies

5. **Model Adaptation and Personalization**
   - One-size-fits-all models have limitations
   - Online learning on microcontrollers challenging
   - Transfer learning from general to specific users needed

### 6.2 Opportunities for TinyHAR

1. **Complete End-to-End System**
   - Most research focuses on individual components
   - TinyHAR provides integrated solution: hardware + firmware + training

2. **Reproducible Implementation**
   - Open-source code and detailed documentation
   - Step-by-step replication guide
   - Accessible hardware (<$20 total cost)

3. **Children Activity Recognition**
   - Novel application domain
   - Ethical framework for pediatric research
   - Foundation for educational applications (P.E. monitoring)

4. **Privacy-First Design**
   - Local processing by default
   - Optional data transmission for monitoring only
   - No cloud dependency

5. **Practical Power Analysis**
   - Real-world battery life measurements
   - Power optimization strategies documented
   - Trade-offs between accuracy and energy consumption

---

## 7. References

### 7.1 Primary Research Papers

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN 2013*.

2. Human Activity Recognition on Microcontrollers with Quantized and Adaptive Deep Learning Models. (2022). *ACM Transactions*. DOI: 10.1145/3542819

3. TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Human Activity Recognition. (2025). *arXiv:2507.07949v1*.

4. Quantized Neural Networks for Microcontrollers: A Comprehensive Survey. (2025). *arXiv:2508.15008v1*.

5. Quantization and Deployment of Deep Neural Networks on Microcontrollers. (2021). *Sensors*, 21(9), 2984. DOI: 10.3390/s21092984

### 7.2 Dataset Sources

1. UCI HAR Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. WISDM Dataset: https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
3. PAMAP2 Dataset: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
4. MotionSense Dataset: https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
5. HuGaDB Dataset: https://www.kaggle.com/datasets/romanchereshnev/hugadb-human-gait-database

### 7.3 Technical Documentation

1. TensorFlow Lite Micro: https://github.com/tensorflow/tflite-micro
2. ESP32 Technical Reference: https://www.espressif.com/en/products/socs/esp32
3. ADXL345 Datasheet: https://www.analog.com/en/products/adxl345.html
4. ITG3200 Datasheet: https://www.invensense.com/products/motion-tracking/3-axis/itg-3200/
5. HMC5883L Datasheet: https://www.honeywell.com/

### 7.4 Tutorials and Guides

1. TinyML with ESP32 Tutorial: https://www.teachmemicro.com/tinyml-with-esp32-tutorial/
2. TensorFlow Lite Micro on ESP32: https://medium.com/@eduardo.bl/iris-dataset-classification-model-in-esp32-tensorflow-lite-micro-http-server-5aa5a66f7543
3. GY-85 IMU Tutorial: https://www.instructables.com/Tutorial-to-Interface-GY-85-IMU-9DOF-Sensor-With-A/
4. ESP32 Access Point Web Server: https://www.dfrobot.com/blog-851.html

---

## Appendix A: Search Queries Used

For reproducibility, here are the search queries used to compile this literature review:

1. "wearable sensor IMU accelerometer gyroscope activity recognition IEEE 2022-2026"
2. "Edge-based Human Activity Recognition ESP32 TensorFlow Lite Micro 2020-2026"
3. "UCI HAR dataset Kaggle WISDM PAMAP2 human activity recognition download"
4. "model quantization optimization embedded systems microcontroller HAR"
5. "arXiv human activity recognition deep learning wearable sensors 2023 2024 2025"
6. "IMU sensor fusion accelerometer gyroscope magnetometer activity recognition algorithm"
7. "TensorFlow Lite Micro ESP32 tutorial implementation guide 2023 2024"

---

**Document Status:** Complete
**Next Steps:**
1. Download UCI HAR, WISDM, and PAMAP2 datasets
2. Create TECHNICAL_PROTOCOLS.md for implementation details
3. Create IMPLEMENTATION_PLAN.md for step-by-step guide

**Maintained by:** TinyHAR Project Team
**License:** CC BY 4.0 (Documentation), datasets follow their respective licenses


