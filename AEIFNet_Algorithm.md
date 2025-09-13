# AEIFNet Hybrid Anomaly Detection Framework

This document describes the detailed step-by-step logic of the **AEIFNet (Autoencoder + Isolation Forest) based Hybrid Anomaly Detection Framework** used for real-time safety monitoring in mining environments.

The goal of this algorithm is to detect anomalies in environmental and physiological data collected from sensors using a combination of deep learning (Autoencoders) and traditional machine learning (Isolation Forest), along with threshold-based compliance checks.

---

## Algorithm 1: Hybrid Anomaly Detection Framework

### Step-by-Step Process

```text
Step 0: Normalize Sensor Readings
   - All input data from environmental and physiological sensors are normalized.
   - Normalization ensures feature uniformity and stabilizes training.

Step 1: Train Autoencoder
   - Input: Multivariate sensor data
   - Architecture:
       Encoder: [Input → 16 → 8]
       Decoder: [8 → 16 → Output]
   - Output: Reconstructed sensor data
   - Loss Function: Mean Squared Error (MSE)
   - Training Parameters: 100 epochs, batch size = 256

Step 2: Compute Reconstruction Errors
   - Each input sample is passed through the trained Autoencoder.
   - Reconstruction error for each sample is calculated as:
       MSE = mean((input - reconstructed_input)^2)

Step 3: Train Isolation Forest
   - Input Features: Concatenated [sensor data + reconstruction error]
   - Contamination Parameter: 0.007 (fraction of anomalies in data)
   - Learns anomaly boundaries by isolating outliers via random splits.

Step 4: Hybrid Anomaly Prediction
   - Define Autoencoder anomaly threshold:
       THybrid = 0.009
   - Rules:
       • Autoencoder flags sample if reconstruction error > THybrid
       • Isolation Forest flags sample if classified as outlier
   - Hybrid_Anomaly = 1 

Step 5: Final Anomaly Decision (Rule-Based Verification)
   - If (Hybrid_Anomaly = 1) AND (sensor values exceed manual safety thresholds)
         → Mark sample as **Anomaly**
     Else
         → Mark sample as **Normal**

