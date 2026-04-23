import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import os
import shutil

# Paths
DATASET_DIR = r"g:\Projects\AI-BASED-NIDS\dataset"
MODEL_DIR = r"g:\Projects\AI-BASED-NIDS\models"

def retrain_model():
    print("=== STARTING RETRAINING PIPELINE ===")
    
    # 1. Load Confirmed Feedback Data
    feedback_path = os.path.join(DATASET_DIR, "verified_new.csv")
    if not os.path.exists(feedback_path):
        print("No verified feedback data found.")
        return
    
    df_new = pd.read_csv(feedback_path)
    print(f"Loaded {len(df_new)} verified samples.")
    
    if len(df_new) < 50:
        print("Not enough samples to retrain. Waiting for more data.")
        return

    # 2. Preprocess (Reuse Loader Logic conceptually)
    # Ideally, we import loader here. For now, assuming data is already roughly aligned or we align it
    try:
        scaler = joblib.load(os.path.join(DATASET_DIR, "scaler.save"))
    except:
        print("Scaler not found. Cannot normalize.")
        return

    # Separate X, y
    y_new = df_new['label'].values
    X_new = df_new.drop(columns=['label', 'timestamp'], errors='ignore')
    
    # Align cols (Simplified)
    # X_new = aligner.align(X_new) 
    
    X_new_scaled = scaler.transform(X_new)
    # Reshape for CNN (samples, features, 1)
    X_new_reshaped = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)

    # 3. Load Current Model
    model_path = os.path.join(MODEL_DIR, "cnn_model.h5")
    if not os.path.exists(model_path):
        print("Base model not found.")
        return
        
    print("Loading current CNN model...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Incremental Training (Fine-tuning)
    # We train on the new data with a small learning rate to avoid catastrophic forgetting
    print("Fine-tuning model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_new_reshaped, y_new, test_size=0.2)
    
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=16)
    
    # 5. Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy on New Data: {val_acc:.4f}")
    
    # 6. Safety Check & Save
    # In a real system, we would also test against a 'holdout' set of old data to ensure we didn't break old detections.
    if val_acc > 0.75: # Arbitrary safety threshold
        backup_path = model_path + ".bak"
        shutil.copy(model_path, backup_path)
        print(f"Backed up old model to {backup_path}")
        
        model.save(model_path)
        print(f"SUCCESS: Model updated and saved to {model_path}")
        
        # Clear/Archive feedback file so we don't overtrain on it next time?
        # Or keep it for full retraining. For incremental, maybe archive.
        shutil.move(feedback_path, feedback_path + f".processed.{int(time.time())}")
    else:
        print("New model performance suspicious. Aborting update.")

if __name__ == "__main__":
    retrain_model()
