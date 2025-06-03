# AEIFNet Hybrid Anomaly Detection Framework

This document describes the detailed step-by-step logic of the **AEIFNet (Autoencoder + Isolation Forest) based Hybrid Anomaly Detection Framework** used for real-time safety monitoring in mining environments.

The goal of this algorithm is to detect anomalies in environmental and physiological data collected from sensors using a combination of deep learning (Autoencoders) and traditional machine learning (Isolation Forest), along with threshold-based compliance checks.

---

## Algorithm 1: Hybrid Anomaly Detection Framework

### Step-by-Step Process

```text
Step 0: Normalize sensor readings
   - All input data from gas and physiological sensors are scaled to ensure uniformity.
   - This step prepares the data for training and inference using ML/DL models.

Step 1: Train Autoencoder
   - Input: Multivariate sensor data
   - Architecture:
       Encoder: [Input → 16 → 8]
       Decoder: [8 → 16 → Output]
   - Output: Reconstructed version of the original input
   - Loss Function: Mean Squared Error (MSE)
   - Training Parameters: 100 epochs, batch size = 256

Step 2: Compute Reconstruction Errors
   - After training, each input sample is passed through the Autoencoder.
   - The reconstruction error is calculated as:
       MSE = mean((input - reconstructed_input)^2)
   - This error serves as an indicator of how "normal" or "abnormal" the sample is.

Step 3: Train Isolation Forest
   - Input: Concatenated vector of original sensor data + reconstruction errors
   - Contamination Parameter: 0.007 (estimated proportion of anomalies in training data)
   - The model isolates outliers using randomly selected features and thresholds.

Step 4: Predict Anomalies (Hybrid Detection)
   - Autoencoder: Flag a sample if its reconstruction error exceeds a threshold:
       THybrid = 0.009
   - Isolation Forest: Independently flag outliers based on tree partitioning
   - Hybrid_Anomaly is triggered if:
       (Reconstruction Error > THybrid) OR (Isolation Forest flags as outlier)

Step 5: Final Anomaly Decision
   - Manual thresholds are evaluated to ensure critical safety limits are enforced:
       If Hybrid_Anomaly = 1 AND any sensor reading exceeds predefined safety bounds:
           Mark the sample as anomalous
       Else:
           Mark as normal
