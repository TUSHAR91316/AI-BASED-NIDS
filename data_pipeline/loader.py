import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

class FeatureAligner:
    """
    Ensures that input data (from different datasets or live traffic)
    always maps to the same fixed feature vector expected by the models.
    """
    def __init__(self, target_features):
        self.target_features = target_features

    def align(self, df):
        """
        Aligns the dataframe columns to self.target_features.
        - Fills missing columns with 0.
        - Removes extra columns.
        - Reorders columns to match target.
        """
        # 1. Create missing columns with 0
        missing_cols = set(self.target_features) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
            
        # 2. Select only target columns in correct order
        aligned_df = df[self.target_features]
        return aligned_df

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        # Define the "Gold Standard" feature set (Based on CIC-IDS2017/2018 intersection)
        self.required_features = [
            'Destination Port', 'Flow Duration', 'Total Fwd Packets',
            'Total Backward Packets', 'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max',
            'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
            'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
            'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
            'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Header Length.1', 'Subflow Fwd Packets',
            'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
            'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
            'Idle Std', 'Idle Max', 'Idle Min'
        ]
        self.aligner = FeatureAligner(self.required_features)

    def load_files_optimized(self, pattern="*", max_workers=4, chunk_size=100000):
        """Loads and merges all CSVs/Parquets with parallel processing and chunking."""
        # Check both csv and parquet
        files = glob.glob(str(self.dataset_path / "**" / "*.csv"), recursive=True)
        files += glob.glob(str(self.dataset_path / "**" / "*.parquet"), recursive=True)

        if not files:
            print("No files found.")
            return pd.DataFrame()

        print(f"Loading {len(files)} files with {max_workers} workers...")

        # Parallel file loading
        df_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._load_single_file_chunked, f, chunk_size): f for f in files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    df_chunks.extend(chunks)
                    print(f"✅ Loaded {file_path}")
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")

        if not df_chunks:
            return pd.DataFrame()

        # Concatenate all chunks
        print("Concatenating chunks...")
        full_df = pd.concat(df_chunks, ignore_index=True)

        # Force garbage collection
        del df_chunks
        gc.collect()

        return full_df

    def _load_single_file_chunked(self, file_path, chunk_size):
        """Load a single file in chunks to manage memory."""
        chunks = []
        try:
            if file_path.endswith('.parquet'):
                # Parquet files can be loaded directly (usually compressed)
                df = pd.read_parquet(file_path)
                chunks.append(df)
            else:
                # CSV files - load in chunks
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

        return chunks

    def preprocess_optimized(self, df, label_col='Label', batch_size=50000):
        """
        Cleans and aligns the data with memory optimization.
        """
        print(f"Preprocessing {len(df)} rows...")

        # Handle Labels (Encode Benign=0, Attack=1 for Supervised)
        if label_col in df.columns:
            # Standardize label names (CIC datasets use 'BENIGN')
            df['target'] = df[label_col].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)

        # Clean infinite/NaN values in batches
        print("Cleaning data...")
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            batch = df.iloc[i:end_idx]
            batch.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Fill NaN with 0 for numerical columns
            numeric_cols = batch.select_dtypes(include=[np.number]).columns
            batch[numeric_cols] = batch[numeric_cols].fillna(0)
            df.iloc[i:end_idx] = batch

        # Align features
        print("Aligning features...")
        X = self.aligner.align(df)

        # Normalize in batches to save memory
        print("Normalizing data...")
        scaler = MinMaxScaler()
        X_scaled = np.zeros_like(X, dtype=np.float32)

        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            X_scaled[i:end_idx] = scaler.fit_transform(X.iloc[i:end_idx])

        # Save Scaler for Real-Time use
        import joblib
        scaler_path = self.dataset_path / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

        # Clean up memory
        del X
        gc.collect()

        return X_scaled, df['target'] if 'target' in df.columns else None

    def load_scaler(self):
        import joblib
        scaler_path = self.dataset_path / "scaler.joblib"
        if scaler_path.exists():
            return joblib.load(scaler_path)
        return None

if __name__ == "__main__":
    # Test
    loader = DataLoader(r"g:\Projects\AI-BASED-NIDS\dataset")
    print("Features defined:", len(loader.required_features))
