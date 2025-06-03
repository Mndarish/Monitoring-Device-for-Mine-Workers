# Monitoring-Device-for-Mine-Workers
This project is a smart wearable monitoring device designed for mine workers. It continuously tracks vital signs such as heart rate, body temperature, and detects hazardous gases in the environment. The system provides real-time alerts to ensure safety and enables timely intervention in case of emergencies underground.




# AEIFNet: Hybrid Autoencoder–Isolation Forest Network for Miners’ Safety

## Overview

This project implements a real-time multi-layered safety monitoring system for mine workers using IoT sensor networks, machine learning, and web-based remote monitoring. It detects hazardous environmental and physiological anomalies using a novel hybrid model — AEIFNet (Autoencoder + Isolation Forest).

Designed to improve worker safety in hazardous underground mining environments, the system enables immediate local and remote alerts for proactive intervention.

---

## Highlights

- Real-time anomaly detection using AEIFNet (Autoencoder + Isolation Forest)
- ESP32-based IoT architecture for sensor data acquisition and transmission
- FastAPI + WebSocket dashboard for live visualization and alerts
- Custom synthetic dataset simulating realistic mining scenarios (based on OSHA/NIOSH standards)
- Model achieves 99% accuracy, 96% F1-score, 91% precision, and 100% recall

---

## AEIFNet Model Architecture

AEIFNet is a hybrid anomaly detection model composed of:
- Autoencoder (AE): Learns compressed representations of normal behavior; detects deviation via reconstruction error
- Isolation Forest (IF): Flags statistical outliers using tree-based ensemble learning
- Manual Thresholding: Compliance check using hard safety thresholds
- Final Decision Layer: Triggers alert only if all conditions confirm a hazardous anomaly

> Final Output = AE detects anomaly OR IF detects anomaly AND Thresholds violated

---

## Hardware Stack

| Component         | Function                                      |
|------------------|-----------------------------------------------|
| ESP32-Wroom-32    | Central controller + WiFi transmission       |
| MQ-135            | CO2 detection                                |
| MQ-136            | H₂S detection                                |
| MQ-7              | CO detection                                 |
| MQ-137            | NH₃ detection                                |
| AD8232            | ECG/Heart rate monitoring                    |
| Buzzer            | On-site alert system                         |
| Custom PCB        | Integrated sensor mounting & routing         |

---

## Dataset & Simulation

Due to lack of open mining datasets, a synthetic dataset was generated using:
- Toxic gas ranges from OSHA/NIOSH (PEL, LTEL, STEL)
- ECG parameter ranges (PR, QT, ST intervals)
- Simulated Combined Cardiac Load (CCL) and PR/QT ratio
- Anomaly flags: `gas_flag`, `cardiac_flag` used for evaluation (not training)

---

## Performance Metrics

| Model               | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| Autoencoder         | 0.73      | 0.79   | 0.76     | 0.97     |
| Isolation Forest    | 0.88      | 0.72   | 0.79     | 0.98     |
| Hybrid AEIFNet      | 0.91      | 1.00   | 0.96     | 0.99     |

AUC (AEIFNet): 0.93

---

## Software Stack

- Language: Python 3.10+
- Microcontroller: C++ (Arduino IDE for ESP32)
- Libraries:
  - `scikit-learn`, `tensorflow/keras`, `pandas`, `numpy`
  - `FastAPI`, `uvicorn`, `websockets`, `matplotlib`

---

## System Architecture

```text
[Sensor Array] → [ESP32 Preprocessing + WiFi] → [Local Server]
                    ↓                              ↓
               [Local Alert]                 [AEIFNet Model]
                                               ↓
                                      [FastAPI Web Dashboard]
                                               ↓
                                      [Remote Alert & Logging]
