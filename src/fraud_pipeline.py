import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Simulate Financial Transaction Data
def generate_transactions(n_samples=1000):
    np.random.seed(42)
    normal_amounts = np.random.normal(loc=50, scale=15, size=int(n_samples * 0.95))
    fraud_amounts = np.random.normal(loc=500, scale=100, size=int(n_samples * 0.05))
    
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    np.random.shuffle(amounts)
    
    df = pd.DataFrame({
        'transaction_id': [f"TXN-{i+1000}" for i in range(n_samples)],
        'amount': np.round(amounts, 2),
        'tx_frequency_1h': np.random.poisson(lam=2, size=n_samples),
        'distance_from_home_km': np.round(np.random.exponential(scale=10, size=n_samples), 2)
    })
    
    # Add outlier distance for extreme amounts
    df.loc[df['amount'] > 300, 'distance_from_home_km'] *= 5
    return df

# 2. Fraud & Anomaly Detection Pipeline
def detect_fraud(df):
    features = ['amount', 'tx_frequency_1h', 'distance_from_home_km']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Train Isolation Forest for Anomaly Detection
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = model.fit_predict(scaled_features)
    
    # Convert prediction: -1 = Anomaly (Fraud), 1 = Normal
    df['is_flagged'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
    return df

if __name__ == "__main__":
    print("--- 1. Generating Streaming Transaction Data ---")
    raw_data = generate_transactions(n_samples=1000)
    print(raw_data.head())
    
    print("\n--- 2. Running Fraud Detection Engine ---")
    processed_data = detect_fraud(raw_data)
    
    flagged_txns = processed_data[processed_data['is_flagged'] == True]
    print(f"Total Transactions Processed: {len(processed_data)}")
    print(f"Flagged Suspicious Transactions: {len(flagged_txns)}")
    
    print("\n--- 3. Top Flagged Fraud Alerts ---")
    print(flagged_txns[['transaction_id', 'amount', 'tx_frequency_1h', 'distance_from_home_km']].head())