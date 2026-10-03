# Real-Time Fraud & Anomaly Detection Pipeline

An end-to-end data processing and machine learning pipeline for detecting fraudulent transactions and anomalies in streaming financial data.

## Business Problem
Financial institutions lose billions annually to fraudulent activities. Traditional rule-based systems fail to detect complex fraud patterns. This automated pipeline leverages machine learning anomaly detection to flag high-risk transactions in real time.

## Technical Stack
* **Language & Libraries:** Python (`pandas`, `numpy`, `scikit-learn`)
* **Machine Learning:** `IsolationForest` & Random Forest Classifiers
* **Data Ingestion:** Synthetic Financial Transaction Stream
* **Visualization:** Seaborn / Matplotlib

## Project Architecture & Workflow
1. **Data Ingestion:** Generate synthetic credit card transaction streams with custom risk features.
2. **Feature Engineering:** Calculate transaction frequency, amount z-scores, and deviation metrics.
3. **Model Inference:** Run transactions through an anomaly detection model to produce a risk score.
4. **Alerting System:** Filter high-risk alerts into a flagged transaction log for fraud analysts.

## Project Structure
```text
fraud-detection-pipeline/
├── README.md
├── requirements.txt
└── src/
    └── fraud_pipeline.py