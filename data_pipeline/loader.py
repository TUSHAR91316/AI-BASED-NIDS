import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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

    def load_files(self, pattern="*"):
        """Loads and merges all CSVs/Parquets matching the pattern."""
        # Check both csv and parquet
        files = glob.glob(str(self.dataset_path / "**" / "*.csv"), recursive=True)
        files += glob.glob(str(self.dataset_path / "**" / "*.parquet"), recursive=True)
        
        df_list = []
        for f in files:
            print(f"Loading {f}...")
            try:
                if f.endswith('.parquet'):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                
                # Align columns immediately to save memory if possible, or just append
                # For now append raw, align later
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not df_list:
            print("No files found.")
            return pd.DataFrame()

        full_df = pd.concat(df_list, ignore_index=True)
        return full_df

    def preprocess(self, df, label_col='Label'):
        """
        Cleans and aligns the data.
        """
        # Handle Labels (Encode Benign=0, Attack=1 for Supervised)
        if label_col in df.columns:
            # Standardize label names (CIC datasets use 'BENIGN')
            df['target'] = df[label_col].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        
        # Clean infinite/NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Align features
        X = self.aligner.align(df)
        
        # Normalize
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save Scaler for Real-Time use
        import joblib
        scaler_path = self.dataset_path / "scaler.save"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        return X_scaled, df['target'] if 'target' in df.columns else None

    def load_scaler(self):
        import joblib
        scaler_path = self.dataset_path / "scaler.save"
        if scaler_path.exists():
            return joblib.load(scaler_path)
        return None

if __name__ == "__main__":
    # Test
    loader = DataLoader(r"g:\Projects\AI-BASED-NIDS\dataset")
    print("Features defined:", len(loader.required_features))
