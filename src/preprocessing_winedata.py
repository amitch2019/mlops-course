# src/preprocessing_winedata.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def main():
    # Load the prepared data
    wine_data = pd.read_csv('data/wine_quality.csv')
    
    # Let's separate features from the target
    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']
    
    # Convert 'type' to numeric if it's still categorical
    if 'type' in X.columns and X['type'].dtype == 'object':
        X['type'] = X['type'].map({'red': 0, 'white': 1})
    
    # Create train, validation, and test sets (70%, 15%, 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Save the datasets for logging with MLFlow
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    os.makedirs('data/splits', exist_ok=True)
    train_data.to_csv('data/splits/train_data.csv', index=False)
    val_data.to_csv('data/splits/val_data.csv', index=False)
    test_data.to_csv('data/splits/test_data.csv', index=False)
    
    # Basic preprocessing - scaling
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create a scaler
    scaler = StandardScaler()
    
    # Fit and transform on training data, transform validation and test
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save the preprocessed datasets
    os.makedirs('data/processed', exist_ok=True)
    pd.concat([X_train_scaled, y_train], axis=1).to_csv('data/processed/train_scaled.csv', index=False)
    pd.concat([X_val_scaled, y_val], axis=1).to_csv('data/processed/val_scaled.csv', index=False)
    pd.concat([X_test_scaled, y_test], axis=1).to_csv('data/processed/test_scaled.csv', index=False)
    
    # Save the scaler for future use
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Data preparation complete. Ready for modeling.")

if __name__ == "__main__":
    main()
