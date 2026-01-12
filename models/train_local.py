import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.loader import DataLoader

# Config
DATASET_DIR = r"g:\Projects\AI-BASED-NIDS\dataset"
MODEL_DIR = r"g:\Projects\AI-BASED-NIDS\models"
AUTOENCODER_DIR = os.path.join(MODEL_DIR, "autoencoder")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(AUTOENCODER_DIR, exist_ok=True)

# GPU Header
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Training will be slow on CPU.")

def build_cnn(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_local():
    print("=== STARTING LOCAL TRAINING ===")
    
    # 1. Load Data
    loader = DataLoader(DATASET_DIR)
    print("Loading Dataset (Limit to 50k samples for local speed)...")
    
    # Load 2017 data (Parquet)
    # loading pattern matches recursively for parquets too
    df = loader.load_files(pattern="*.parquet") 
    
    if df.empty:
        print("Error: No data found in dataset directory.")
        return

    # Sampling for speed
    if len(df) > 50000:
        df = df.sample(50000)
    
    print(f"Preprocessing {len(df)} samples...")
    # Preprocess (Scale + Align)
    # Note: loader.preprocess handles label encoding if 'Label' or 'Label' column exists
    # CIC-IDS2017 parquet usually has 'Label'
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1] # Guess label if not named rigidly
    
    X, y = loader.preprocess(df, label_col=label_col)
    
    # Save Scaler (Critical for Detector)
    # Loader saves it automatically during preprocess, but good to verify
    
    # 2. Train Supervised (CNN)
    print("Training CNN Model...")
    X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2)
    
    cnn = build_cnn(input_shape=(X.shape[1], 1))
    cnn.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))
    
    cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
    cnn.save(cnn_path)
    print(f"Saved CNN to {cnn_path}")
    
    # 3. Train Autoencoder (Unsupervised - Benign Only)
    print("Training Autoencoder...")
    # Filter only Benign for training one-class
    if y is not None:
        benign_indices = (y == 0)
        X_benign = X[benign_indices]
    else:
        X_benign = X # Assume all normal if no labels? Dangerous.
    
    # Limit samples
    X_benign_train, X_benign_test = train_test_split(X_benign, test_size=0.2)
    
    autoencoder = build_autoencoder(input_dim=X.shape[1])
    autoencoder.fit(X_benign_train, X_benign_train, epochs=2, batch_size=32, validation_data=(X_benign_test, X_benign_test))
    
    ae_path = os.path.join(AUTOENCODER_DIR, "autoencoder.h5")
    autoencoder.save(ae_path)
    
    # Calculate Threshold
    reconstructions = autoencoder.predict(X_benign_test)
    mse = np.mean(np.power(X_benign_test - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 99)
    print(f"Autoencoder Threshold: {threshold}")
    # We might want to save threshold to a file too, or hardcode/config it.
    
    print("=== LOCAL TRAINING COMPLETE ===")
    print("Models and Scaler are ready for realtime_detector.")

if __name__ == "__main__":
    train_local()
