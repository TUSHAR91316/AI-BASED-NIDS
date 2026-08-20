import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import joblib
import time
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from data_pipeline.loader import DataLoader

# Config
DATASET_DIR = Path(r"g:\Projects\AI-BASED-NIDS\dataset")
MODEL_DIR = r"g:\Projects\AI-BASED-NIDS\models"
AUTOENCODER_DIR = os.path.join(MODEL_DIR, "autoencoder")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(AUTOENCODER_DIR, exist_ok=True)

# GPU Header with memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Limit GPU memory usage to 80%
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
            )
        print(f"✅ GPU Detected & Configured: {gpus}")
        print("✅ Mixed Precision Enabled")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Training will be slow on CPU.")

# --- Custom Layers ---
class Attention(layers.Layer):
    def __init__(self, step_dim, **kwargs):
        self.supports_masking = True
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 name='{}_W'.format(self.name))
        self.features_dim = input_shape[-1]
        self.b = self.add_weight(shape=(input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name))
        self.u = self.add_weight(shape=(input_shape[1],),
                                 initializer='glorot_uniform',
                                 name='{}_u'.format(self.name))
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = keras.backend.reshape(keras.backend.dot(keras.backend.reshape(x, (-1, features_dim)),
                        keras.backend.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        eij += self.b
        eij = keras.backend.tanh(eij)
        a = keras.backend.exp(eij)
        if mask is not None:
            a *= keras.backend.cast(mask, keras.backend.floatx())
        a /= keras.backend.cast(keras.backend.sum(a, axis=1, keepdims=True) + keras.backend.epsilon(), keras.backend.floatx())
        a = keras.backend.expand_dims(a)
        weighted_input = x * a
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "step_dim": self.step_dim
        })
        return config

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 1x1 conv if shapes don't match or just add
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_advanced_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual Blocks
    x = residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = residual_block(x, 128)
    x = layers.MaxPooling1D(2)(x)
    x = residual_block(x, 256)
    
    # Attention Mechanism
    # We flatten implicitly via GlobalAverage or use a custom Attention layer
    # Let's use GlobalAveragePooling for robustness + a dense head
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        attn_output = tf.cast(self.att(inputs, inputs), tf.float32)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = tf.cast(self.ffn(out1), tf.float32)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

def build_lstm(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Transformers expect (batch, seq_len, features)
    # Our data is (batch, features, 1) effectively from reshape?
    # Actually for tabular transformer:
    # If we treat each feature as a "token" in a sequence, we need embedding.
    # But here we just have a vector. 
    # Let's treat the feature vector as the sequence of length 'features' and dim 1.
    
    x = TransformerBlock(embed_dim=1, num_heads=2, ff_dim=32)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_deep_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(16, activation='relu')(x) # Latent space
    
    # Decoder
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)
    
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def load_and_merge_datasets(loader, max_samples_per_dataset=1000000):
    """
    Load and merge multiple datasets with intelligent sampling.
    Uses stratified sampling to maintain class balance across datasets.
    """
    print("=== LOADING AND MERGING MULTIPLE DATASETS ===")
    
    # Load both CSV and Parquet files
    df_list = []
    
    # Find all dataset files
    dataset_files = []
    for ext in ['*.parquet', '*.csv']:
        dataset_files.extend(glob.glob(str(DATASET_DIR / "**" / ext), recursive=True))
    
    print(f"Found {len(dataset_files)} dataset files")
    
    for f in dataset_files:
        print(f"\nProcessing: {f}")
        try:
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            
            initial_len = len(df)
            
            # Check for label column variations
            label_col = None
            for col in df.columns:
                if col.lower() in ['label', 'labels', 'target', 'class', 'attack', 'category']:
                    label_col = col
                    break
            
            if label_col:
                # Standardize label column name
                df = df.rename(columns={label_col: 'Label'})
                
                # Show class distribution
                label_counts = df['Label'].value_counts()
                print(f"  Records: {initial_len}, Classes: {len(label_counts)}")
                print(f"  Top classes: {dict(label_counts.head(3))}")
                
                # Smart sampling per dataset to balance memory usage
                if len(df) > max_samples_per_dataset:
                    # Stratified sampling to preserve class distribution
                    try:
                        from sklearn.model_selection import StratifiedShuffleSplit
                        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples_per_dataset, random_state=42)
                        for _, sample_idx in sss.split(df, df['Label']):
                            df = df.iloc[sample_idx]
                    except:
                        # Fallback to random sampling if stratified fails
                        df = df.sample(max_samples_per_dataset, random_state=42)
                    print(f"  Sampled to: {len(df)} records")
            else:
                print(f"  Warning: No label column found, skipping")
                continue
            
            df_list.append(df)
            
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    if not df_list:
        print("Error: No valid datasets found!")
        return pd.DataFrame()
    
    # Merge all datasets
    print(f"\n=== MERGING {len(df_list)} DATASETS ===")
    
    # Find common columns across all datasets
    common_cols = set(df_list[0].columns)
    for df in df_list[1:]:
        common_cols &= set(df.columns)
    
    print(f"Common columns: {len(common_cols)}")
    
    # Select only common columns and merge
    merged_df = pd.concat([df[list(common_cols)] for df in df_list], ignore_index=True)
    
    print(f"\nTotal records: {len(merged_df)}")
    print(f"Class distribution:")
    print(merged_df['Label'].value_counts())
    
    return merged_df

def balance_dataset(df, method='hybrid'):
    """
    Balance dataset using multiple strategies.
    method: 'undersample', 'oversample', or 'hybrid' (default)
    """
    from sklearn.utils import resample
    
    label_counts = df['Label'].value_counts()
    max_count = label_counts.max()
    min_count = label_counts.min()
    
    print(f"\n=== BALANCING DATASET (method={method}) ===")
    print(f"Before: {dict(label_counts)}")
    
    if method == 'undersample':
        # Downsample majority classes
        target_size = min_count * 2  # Keep at least 2x minority class
        balanced_dfs = []
        for label in label_counts.index:
            df_class = df[df['Label'] == label]
            if len(df_class) > target_size:
                df_class = resample(df_class, replace=False, n_samples=target_size, random_state=42)
            balanced_dfs.append(df_class)
        df = pd.concat(balanced_dfs)
        
    elif method == 'oversample':
        # Upsample minority classes using SMOTE-like approach
        try:
            from imblearn.over_sampling import SMOTE
            # This requires numeric features only
            X = df.drop('Label', axis=1).select_dtypes(include=[np.number])
            y = df['Label']
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='Label')], axis=1)
        except Exception as e:
            print(f"SMOTE failed ({e}), using random oversampling")
            target_size = max_count // 2
            balanced_dfs = []
            for label in label_counts.index:
                df_class = df[df['Label'] == label]
                if len(df_class) < target_size:
                    df_class = resample(df_class, replace=True, n_samples=target_size, random_state=42)
                balanced_dfs.append(df_class)
            df = pd.concat(balanced_dfs)
            
    else:  # hybrid
        # Smart balancing: neither fully undersample nor fully oversample
        target_size = int(np.median(label_counts))
        balanced_dfs = []
        for label in label_counts.index:
            df_class = df[df['Label'] == label]
            if len(df_class) > target_size * 2:
                # Undersample very large classes
                df_class = resample(df_class, replace=False, n_samples=target_size * 2, random_state=42)
            elif len(df_class) < target_size // 2:
                # Oversample very small classes
                df_class = resample(df_class, replace=True, n_samples=target_size // 2, random_state=42)
            balanced_dfs.append(df_class)
        df = pd.concat(balanced_dfs)
    
    print(f"After: {dict(df['Label'].value_counts())}")
    return df

def train_local():
    print("=== STARTING ADVANCED GPU TRAINING WITH ENHANCED DATASETS ===")
    
    # 1. Load and Merge Multiple Datasets
    loader = DataLoader(DATASET_DIR)
    df = load_and_merge_datasets(loader, max_samples_per_dataset=800000)
    
    if df.empty:
        print("Error: No data found in dataset directory.")
        print("Run: python data_pipeline/dataset_manager.py --recommended")
        return
    
    # 2. Balance the dataset for better training
    df = balance_dataset(df, method='hybrid')
    
    # 3. Final sampling for memory management (but keep more data)
    MAX_TRAINING_SAMPLES = 1000000  # Increased from 500k
    if len(df) > MAX_TRAINING_SAMPLES:
        print(f"\nStratified sampling to {MAX_TRAINING_SAMPLES} for training...")
        # Use stratified sampling to preserve class distribution
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_TRAINING_SAMPLES, random_state=42)
            for _, sample_idx in sss.split(df, df['Label']):
                df = df.iloc[sample_idx]
        except:
            df = df.sample(MAX_TRAINING_SAMPLES, random_state=42)
    
    print(f"Preprocessing {len(df)} samples...")
    # Preprocess (Scale + Align)
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1] 
    
    X, y = loader.preprocess(df, label_col=label_col)
    
    # 2. Train Supervised (ResNet CNN)
    cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
    if os.path.exists(cnn_path):
        print(f"✅ CNN Model already exists at {cnn_path}. Skipping training.")
        # Load it to be sure? No need if we trust the file.
        X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], X.shape[1], 1), y, test_size=0.2, random_state=42)
    else:
        print("\n--- Training Residual CNN Model ---")
        X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
        X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)
        
        cnn = build_advanced_cnn(input_shape=(X.shape[1], 1))
        cnn.summary()
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_cnn.h5"), save_best_only=True)
        ]
        
        cnn.fit(X_train, y_train, 
                epochs=10, 
                batch_size=64, 
                validation_data=(X_test, y_test),
                callbacks=callbacks)
        
        cnn.save(cnn_path)
        print(f"✅ Saved Advanced CNN to {cnn_path}")
    
    # 3. Train Autoencoder (Unsupervised - Benign Only)
    ae_path = os.path.join(AUTOENCODER_DIR, "autoencoder.h5")
    if os.path.exists(ae_path):
        print(f"✅ Autoencoder already exists at {ae_path}. Skipping training.")
        if y is not None:
             benign_indices = (y == 0)
             X_benign = X[benign_indices]
        else:
             X_benign = X
        
        # We still need X_benign_test for threshold calculation
        X_benign_train, X_benign_test = train_test_split(X_benign, test_size=0.2, random_state=42)
        
        # Load the model
        autoencoder = keras.models.load_model(ae_path)
    else:
        print("\n--- Training Deep Autoencoder ---")
        # Filter only Benign for training one-class
        if y is not None:
            benign_indices = (y == 0)
            X_benign = X[benign_indices]
        else:
            X_benign = X 
        
        print(f"Training on {len(X_benign)} Benign samples.")
        X_benign_train, X_benign_test = train_test_split(X_benign, test_size=0.2, random_state=42)
        
        autoencoder = build_deep_autoencoder(input_dim=X.shape[1])
        
        ae_callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(os.path.join(AUTOENCODER_DIR, "best_ae.h5"), save_best_only=True)
        ]
        
        autoencoder.fit(X_benign_train, X_benign_train, 
                        epochs=10, 
                        batch_size=64, 
                        validation_data=(X_benign_test, X_benign_test),
                        callbacks=ae_callbacks)
        
        autoencoder.save(ae_path)
    
    # Calculate Threshold
    print("Calculating Anomaly Threshold...")
    reconstructions = autoencoder.predict(X_benign_test)
    mse = np.mean(np.power(X_benign_test - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 99) # 99th percentile of benign reconstruction error
    print(f"✅ Autoencoder Threshold (99%): {threshold}")
    
    # Save threshold
    with open(os.path.join(AUTOENCODER_DIR, "threshold.txt"), "w") as f:
        f.write(str(threshold))
        
    # 4. Train Isolation Forest (Anomaly Detection - Alternative)
    print("\n--- Training Isolation Forest ---")
    from sklearn.ensemble import IsolationForest
    
    # Isolation Forest is unsupervised, but works best when trained on "normal" data 
    # to learn the distribution of normality.
    print(f"Training Isolation Forest on {len(X_benign)} Benign samples...")
    
    # Subsample for IF (it can be memory intensive and slow on large datasets)
    if len(X_benign) > 100000:
        X_iso = X_benign[np.random.choice(len(X_benign), 100000, replace=False)]
    else:
        X_iso = X_benign
        
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_iso)
    
    iso_path = os.path.join(MODEL_DIR, "isolation_forest", "isolation_forest.joblib")
    os.makedirs(os.path.dirname(iso_path), exist_ok=True)
    joblib.dump(iso_forest, iso_path)
    print(f"✅ Saved Isolation Forest to {iso_path}")

    # 5. Train LSTM (Sequential/Time-Series focus)
    print("\n--- Training LSTM Model ---")
    lstm = build_lstm(input_shape=(X.shape[1], 1))
    
    lstm_callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "lstm", "best_lstm.h5"), save_best_only=True)
    ]
    os.makedirs(os.path.join(MODEL_DIR, "lstm"), exist_ok=True)
    
    lstm.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=lstm_callbacks)
    
    lstm_path = os.path.join(MODEL_DIR, "lstm", "lstm_model.h5")
    lstm.save(lstm_path)
    print(f"✅ Saved LSTM to {lstm_path}")

    # 6. Train Transformer (Tabular Transformer)
    print("\n--- Training Transformer Model ---")
    transformer = build_transformer(input_shape=(X.shape[1], 1))
    
    trans_callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "transformer", "best_transformer.h5"), save_best_only=True)
    ]
    os.makedirs(os.path.join(MODEL_DIR, "transformer"), exist_ok=True)
    
    transformer.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=trans_callbacks)
    
    trans_path = os.path.join(MODEL_DIR, "transformer", "transformer_model.h5")
    transformer.save(trans_path)
    print(f"✅ Saved Transformer to {trans_path}")
    
    print("\n=== TRAINING COMPLETE ===")
    print("All models (CNN, Autoencoder, Isolation Forest, LSTM, Transformer) are ready.")

if __name__ == "__main__":
    train_local()
