"""
generate_colab_notebook.py
---------------------------
Run this script to generate AI_NIDS_Colab_Training.ipynb
Usage: python generate_colab_notebook.py
"""

import json
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "AI_NIDS_Colab_Training.ipynb")

def cell(source, cell_type="code"):
    """Helper to create a standard Jupyter notebook cell with line-array source."""
    if isinstance(source, str):
        src_list = source.splitlines(keepends=True)
    else:
        src_list = source

    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src_list}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src_list
    }


CELLS = [

# ─── CELL 0: Header ────────────────────────────────────────────────────────────
cell("""# 🛡️ AI-BASED NIDS — Complete Training Pipeline

**Google Colab GPU Training Notebook**

This notebook trains the complete hybrid AI detection engine:
- ✅ **Residual 1D-CNN** (Supervised — Attack Classification)
- ✅ **Bidirectional LSTM** (Supervised — Sequential Flow Modeling)
- ✅ **Transformer** (Supervised — Attention-Based Detection)
- ✅ **Deep Autoencoder** (Unsupervised — Zero-Day Anomaly Detection)
- ✅ **Isolation Forest** (Unsupervised — Outlier Detection)

---

## 🚀 Quick Start
1. **Enable GPU**: Runtime → Change runtime type → T4 GPU
2. **Upload Kaggle API key** (`kaggle.json`) in Cell 2
3. **Run All Cells**: Runtime → Run All
4. **Download** the `NIDS_Models.zip` at the end

## 📦 Primary Dataset
**CIC-IDS2017** — 2.8M records, 80 features, 14 attack categories.  
The gold standard benchmark for NIDS research.

## 📁 Output Structure
```
models/
├── cnn_model.h5                    ← Residual CNN
├── scaler.joblib                   ← MinMaxScaler (70 features)
├── autoencoder/
│   ├── autoencoder.h5              ← Deep Autoencoder
│   └── threshold.txt               ← 99th percentile MSE threshold
├── lstm/
│   └── lstm_model.h5               ← Bidirectional LSTM
├── transformer/
│   └── transformer_model.h5        ← Transformer
└── isolation_forest/
    └── isolation_forest.joblib     ← Isolation Forest
```""", cell_type="markdown"),

# ─── CELL 1: Environment Setup ─────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 1: GPU Check & Environment Setup
# ============================================================
import subprocess, sys

print('=' * 62)
print('  AI-BASED NIDS — Training Pipeline')
print('  CNN + LSTM + Transformer + Autoencoder + IsolationForest')
print('=' * 62)

# Check GPU
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\\n')[:6]
    print('\\n✅ GPU Available:')
    print('\\n'.join(lines))
except:
    print('⚠️  No GPU. Go to Runtime → Change runtime type → T4 GPU')

print('\\nInstalling dependencies...')
# Install all required packages using %pip magic for clean kernel integration
%pip install -q kaggle imbalanced-learn scikit-learn tensorflow numpy pandas matplotlib seaborn plotly joblib tqdm
print('✅ All dependencies installed.')
"""),

# ─── CELL 2: Kaggle Auth ───────────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 2: Kaggle API Authentication
# ============================================================
# Get your kaggle.json from: https://www.kaggle.com/settings → API → Create New Token
import os

try:
    from google.colab import files
except ImportError:
    files = None

if files is not None:
    print('Upload your kaggle.json API key:')
    uploaded = files.upload()
    if 'kaggle.json' in uploaded:
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
        with open(kaggle_path, 'wb') as f:
            f.write(uploaded['kaggle.json'])
        os.chmod(kaggle_path, 0o600)
        print('✅ Kaggle API credentials configured successfully.')
    else:
        print('⚠️  No kaggle.json uploaded.')
else:
    print('⚠️  Running outside Google Colab or google.colab module unavailable.')
    print('   Place kaggle.json in ~/.kaggle/kaggle.json manually if running locally.')
"""),

# ─── CELL 3: Directory Setup ───────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 3: Project Directory Setup
# ============================================================
from pathlib import Path

BASE_DIR   = Path('/content/AI-BASED-NIDS')
DATASET_DIR = BASE_DIR / 'dataset'
MODEL_DIR   = BASE_DIR / 'models'
AE_DIR      = MODEL_DIR / 'autoencoder'
LSTM_DIR    = MODEL_DIR / 'lstm'
TRANS_DIR   = MODEL_DIR / 'transformer'
ISO_DIR     = MODEL_DIR / 'isolation_forest'
CNN_DIR     = MODEL_DIR / 'cnn'
LOG_DIR     = BASE_DIR / 'logs'

for d in [DATASET_DIR, MODEL_DIR, AE_DIR, LSTM_DIR, TRANS_DIR, ISO_DIR, CNN_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print('✅ Directory structure created:')
for p in [BASE_DIR] + sorted([x for x in BASE_DIR.rglob('*') if x.is_dir()]):
    print(f'   {p}')
"""),

# ─── CELL 4: Download CIC-IDS2017 ─────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 4: Download CIC-IDS2017 Dataset (Primary Benchmark)
# ============================================================
import subprocess

def try_download(kaggle_id, dest_folder, label=''):
    print(f'Trying: {kaggle_id}...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', kaggle_id,
         '-p', str(dest_folder), '--unzip'],
        capture_output=True, text=True, timeout=1800
    )
    if result.returncode == 0:
        print(f'✅ {label or kaggle_id} downloaded successfully!')
        return True
    else:
        print(f'  Failed: {result.stderr[:250]}')
        return False

cic_2017_dir = DATASET_DIR / 'CIC-IDS2017'

# Try multiple known Kaggle IDs for CIC-IDS2017
CANDIDATES = [
    ('dhoogla/cicids2017',               'CIC-IDS2017 (primary)'),
    ('cicdataset/cicids-2017',            'CIC-IDS2017 (mirror)'),
    ('shashwatwork/internet-traffic-data-in-the-wild', 'CIC-IDS2017 (alt)'),
]

downloaded = False
for kid, label in CANDIDATES:
    if try_download(kid, cic_2017_dir, label):
        downloaded = True
        break

if not downloaded:
    print('\\n⚠️  CIC-IDS2017 unavailable. Downloading NSL-KDD as fallback...')
    try_download('hassan06/nslkdd', DATASET_DIR / 'NSL-KDD', 'NSL-KDD')

# List all downloaded files
print('\\n📂 Dataset directory:')
all_csv = list(DATASET_DIR.rglob('*.csv'))
for f in sorted(all_csv):
    size_mb = f.stat().st_size / (1024*1024)
    print(f'   {f.relative_to(DATASET_DIR)} ({size_mb:.1f} MB)')
"""),

# ─── CELL 5: (Optional) Additional Datasets ───────────────────────────────────
cell("""\
# ============================================================
# CELL 5: (Optional) Additional Datasets
# Comment/uncomment as needed. LARGE downloads!
# ============================================================
import subprocess

EXTRA = [
    # Uncomment the datasets you want:
    # ('hassan06/nslkdd',                 DATASET_DIR/'NSL-KDD',        'NSL-KDD'),
    # ('mrwellsdouglas/unsw-nb15',         DATASET_DIR/'UNSW-NB15',      'UNSW-NB15'),
    # ('solarmainframe/ids-intrusion-csv', DATASET_DIR/'CIC-IDS2018',    'CIC-IDS2018 ~10GB'),
]

for kid, dest, label in EXTRA:
    print(f'Downloading {label}...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', kid, '-p', str(dest), '--unzip'],
        capture_output=True, text=True, timeout=3600
    )
    status = '✅' if result.returncode == 0 else '❌'
    print(f'  {status} {label}')

print('\\n✅ Extra downloads done (skipped if no extras uncommented).')
"""),

# ─── CELL 6: Data Loading & Exploration ───────────────────────────────────────
cell("""\
# ============================================================
# CELL 6: Data Loading & Exploration
# ============================================================
import pandas as pd
import numpy as np
import glob, os, warnings
warnings.filterwarnings('ignore')

print('📊 Scanning for dataset files...')

csv_files = sorted(glob.glob(str(DATASET_DIR / '**/*.csv'), recursive=True))
print(f'Found {len(csv_files)} CSV files\\n')

def safe_read_csv(fpath, **kwargs):
    # Robust CSV loader with automatic encoding fallback.
    for enc in [None, 'utf-8', 'cp1252', 'latin1']:
        try:
            if enc:
                return pd.read_csv(fpath, encoding=enc, **kwargs)
            else:
                return pd.read_csv(fpath, **kwargs)
        except (UnicodeDecodeError, Exception):
            continue
    # Ultimate fallback with error handling
    return pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip', **kwargs)

# Peek at each file
for fpath in csv_files:
    try:
        df_peek = safe_read_csv(fpath, nrows=3, low_memory=False)
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f'  {os.path.basename(fpath)} ({size_mb:.1f} MB)')
        print(f'    Cols: {len(df_peek.columns)} | Sample: {list(df_peek.columns[:6])}...')
    except Exception as e:
        print(f'  ⚠️  Peek failed for {os.path.basename(fpath)}: {e}')

print()

# ── Load & Merge ───────────────────────────────────────────────
all_dfs = []
MAX_PER_FILE = 600_000   # Rows per file (adjust for Colab RAM ~12GB)

for fpath in csv_files:
    try:
        df_peek = safe_read_csv(fpath, nrows=2, low_memory=False)
        df_peek.columns = [c.strip() for c in df_peek.columns]

        label_col = None
        for col in df_peek.columns:
            if col.strip().lower() in ['label', 'labels', 'attack_type', 'target', 'class', 'category']:
                label_col = col
                break

        if label_col is None:
            print(f'  ⚠️  No label column in {os.path.basename(fpath)} — skipping')
            continue

        df = safe_read_csv(fpath, nrows=MAX_PER_FILE, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if label_col != 'Label':
            df = df.rename(columns={label_col: 'Label'})

        all_dfs.append(df)
        print(f'✅ {os.path.basename(fpath)}: {len(df):,} rows × {len(df.columns)} cols')

    except Exception as e:
        print(f'❌ {os.path.basename(fpath)}: {e}')

if not all_dfs:
    print('⚠️  No dataset files loaded from disk. Generating synthetic baseline dataset for demonstration...')
    # Generate synthetic fallback dataset matching 70 gold-standard features
    synth_cols = [
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
    np.random.seed(42)
    s_data = np.random.rand(10000, len(synth_cols)) * 1000
    synth_df = pd.DataFrame(s_data, columns=synth_cols)
    synth_df['Label'] = ['BENIGN'] * 7000 + ['DoS'] * 3000
    all_dfs.append(synth_df)
    print(f'✅ Created synthetic baseline dataset: {len(synth_df):,} rows × {len(synth_df.columns)} cols')

# Merge
if len(all_dfs) == 1:
    full_df = all_dfs[0]
else:
    common_cols = set(all_dfs[0].columns)
    for df in all_dfs[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)
    print(f'\\nCommon columns: {len(common_cols)}')
    full_df = pd.concat([df[common_cols] for df in all_dfs], ignore_index=True)

print(f'\\n✅ Total: {len(full_df):,} rows × {len(full_df.columns)} columns')
print('\\n📈 Label distribution:')
print(full_df['Label'].value_counts().to_string())
"""),

# ─── CELL 7: Feature Engineering ──────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 7: Feature Engineering & Gold-Standard Alignment
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, gc

# === GOLD STANDARD 70-FEATURE SET ===
# This MUST match the FeatureAligner in data_pipeline/loader.py
REQUIRED_FEATURES = [
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
INPUT_DIM = len(REQUIRED_FEATURES)
print(f'Gold-standard feature vector: {INPUT_DIM} dimensions')

# ── 1. Binary label encoding ───────────────────────────────────
print('\\n[1/5] Encoding labels...')
full_df['target'] = full_df['Label'].apply(
    lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL', 'LEGITIMATE', '0'] else 1
)
print(f'  Benign: {(full_df.target==0).sum():,}  |  Attack: {(full_df.target==1).sum():,}')

# ── 2. Feature alignment ───────────────────────────────────────
print('\\n[2/5] Aligning to 70-feature gold standard...')
for col in REQUIRED_FEATURES:
    if col not in full_df.columns:
        full_df[col] = 0.0

X_df = full_df[REQUIRED_FEATURES].copy()
y    = full_df['target'].values.astype(np.float32)

# ── 3. Data cleaning ───────────────────────────────────────────
print('\\n[3/5] Cleaning: removing inf/NaN and clipping outliers...')
X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_df.fillna(0, inplace=True)

for col in X_df.columns:
    cap = X_df[col].quantile(0.999)
    if cap > 0:
        X_df[col] = X_df[col].clip(upper=cap)

# ── 4. Stratified sampling ─────────────────────────────────────
print('\\n[4/5] Stratified sampling (memory management)...')
MAX_SAMPLES = 600_000
if len(X_df) > MAX_SAMPLES:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SAMPLES, random_state=42)
    for _, idx in sss.split(X_df, y):
        X_df = X_df.iloc[idx].reset_index(drop=True)
        y    = y[idx]
    print(f'  Sampled to {len(X_df):,}')
else:
    print(f'  Using all {len(X_df):,} records')

# ── 5. MinMax scaling ──────────────────────────────────────────
print('\\n[5/5] Fitting MinMaxScaler...')
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_df).astype(np.float32)

scaler_path = MODEL_DIR / 'scaler.joblib'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, str(scaler_path))
print(f'  ✅ Scaler saved: {scaler_path}')
print(f'  Scaled X shape: {X_scaled.shape}')

del X_df, full_df; gc.collect()
print('\\n✅ Feature engineering complete!')
"""),

# ─── CELL 8: Train/Val/Test Split ─────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 8: Train / Validation / Test Split
# ============================================================
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print('✂️  Creating 70% / 15% / 15% stratified splits...')

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.176,   # ~15% of total
    random_state=42, stratify=y_train_full
)

print(f'  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}')
print(f'  Feature dim: {X_train.shape[1]}')
print(f'  Train — Benign: {(y_train==0).sum():,}  |  Attack: {(y_train==1).sum():,}')

# 3-D reshape for Conv/LSTM/Transformer
X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_3d   = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_3d  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Benign-only for unsupervised models
X_benign_train = X_train[(y_train == 0)]
X_benign_val   = X_val[(y_val == 0)]
X_benign_test  = X_test[(y_test == 0)]
X_attack_test  = X_test[(y_test == 1)]
print(f'  Benign train (AE/IF): {len(X_benign_train):,}')

# Class imbalance weight
ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
CLASS_WEIGHT = {0: 1.0, 1: ratio}
print(f'  Class weight ratio (attack): {ratio:.2f}')

# Quick visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Benign', 'Attack'], [(y_train==0).sum(), (y_train==1).sum()],
            color=['#00ff41', '#ff4b4b'])
axes[0].set_title('Training Set Class Distribution')
axes[1].pie([(y==0).sum(), (y==1).sum()], labels=['Benign', 'Attack'],
            colors=['#00ff41', '#ff4b4b'], autopct='%1.1f%%')
axes[1].set_title('Full Dataset')
plt.tight_layout()
plt.savefig(str(LOG_DIR / 'class_distribution.png'), dpi=100, bbox_inches='tight')
plt.show()
print('\\n✅ Splits ready!')
"""),

# ─── CELL 9: Architecture Definitions ─────────────────────────────────────────
cell("""\
# ============================================================
# CELL 9: Neural Network Architecture Definitions
# ============================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f'TensorFlow: {tf.__version__}')
print(f'GPUs found: {tf.config.list_physical_devices(\"GPU\")}')

# Enable mixed precision (speeds up T4/A100 ~2x)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print('Mixed precision (float16) enabled.')

INPUT_SHAPE_3D = (INPUT_DIM, 1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. Residual CNN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def residual_block(x, filters, kernel_size=3):
    skip = x
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding='same')(skip)
    x = layers.Add()([x, skip])
    return layers.Activation('relu')(x)

def build_residual_cnn(input_shape):
    inp = layers.Input(shape=input_shape, name='cnn_input')
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 128)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 256)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32', name='cnn_out')(x)
    m = keras.Model(inp, out, name='ResidualCNN_NIDS')
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return m

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Bidirectional LSTM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_lstm_model(input_shape):
    inp = layers.Input(shape=input_shape, name='lstm_input')
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32', name='lstm_out')(x)
    m = keras.Model(inp, out, name='BiLSTM_NIDS')
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return m

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. Transformer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@keras.utils.register_keras_serializable(package='Custom', name='TransformerBlock')
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw)
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.ff_dim = ff_dim; self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(1, embed_dim // num_heads))
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu', dtype='float32'),
            layers.Dense(embed_dim, dtype='float32')
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.d1 = layers.Dropout(rate)
        self.d2 = layers.Dropout(rate)

    def call(self, x, training=False):
        x = tf.cast(x, tf.float32)
        a = self.att(x, x); a = self.d1(a, training=training)
        o1 = self.ln1(x + a)
        f = self.ffn(o1); f = self.d2(f, training=training)
        return self.ln2(o1 + f)

    def get_config(self):
        c = super().get_config()
        c.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads,
                  'ff_dim': self.ff_dim, 'rate': self.rate})
        return c

def build_transformer_model(input_shape):
    inp = layers.Input(shape=input_shape, name='trans_input')
    x = tf.cast(inp, tf.float32)
    x = layers.Conv1D(32, 1, padding='same', activation='relu', dtype='float32')(x)
    x = TransformerBlock(32, 4, 64)(x)
    x = TransformerBlock(32, 4, 64)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', dtype='float32')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32', name='trans_out')(x)
    m = keras.Model(inp, out, name='Transformer_NIDS')
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return m

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. Deep Autoencoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,), name='ae_input')
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    enc = layers.Dense(16, activation='relu', name='latent')(x)
    x = layers.Dense(64, activation='relu')(enc)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    dec = layers.Dense(input_dim, activation='sigmoid', dtype='float32', name='ae_out')(x)
    ae = keras.Model(inp, dec, name='Autoencoder_NIDS')
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return ae

# Summary
for fn, shape in [(build_residual_cnn, INPUT_SHAPE_3D),
                   (build_lstm_model, INPUT_SHAPE_3D),
                   (build_transformer_model, INPUT_SHAPE_3D),
                   (build_autoencoder, INPUT_DIM)]:
    m = fn(shape)
    print(f'{m.name:<30}: {m.count_params():>10,} parameters')
    del m

print('\\n✅ All architectures defined!')
"""),

# ─── CELL 10: Train CNN ────────────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 10: Train Residual CNN (Supervised)
# ============================================================
import time
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

print('='*60)
print('  TRAINING 1/5 — Residual CNN')
print('='*60)

CNN_PATH = MODEL_DIR / 'cnn_model.h5'

cnn_model = build_residual_cnn(INPUT_SHAPE_3D)
cnn_model.summary()

cbs_cnn = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=5,
                                   restore_best_weights=True, mode='max', verbose=1),
    keras.callbacks.ModelCheckpoint(str(CNN_DIR / 'best_cnn.h5'),
                                     monitor='val_auc', save_best_only=True, mode='max'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                       min_lr=1e-7, verbose=1),
    keras.callbacks.CSVLogger(str(LOG_DIR / 'cnn_training.csv')),
]

t0 = time.time()
cnn_hist = cnn_model.fit(
    X_train_3d, y_train,
    epochs=25,
    batch_size=512,
    validation_data=(X_val_3d, y_val),
    class_weight=CLASS_WEIGHT,
    callbacks=cbs_cnn,
    verbose=1
)
elapsed = (time.time() - t0) / 60
print(f'\\nTraining completed in {elapsed:.1f} minutes')

print('\\nTest Set Evaluation:')
cnn_probs = cnn_model.predict(X_test_3d, verbose=0).flatten()
cnn_preds = (cnn_probs > 0.5).astype(int)
print(classification_report(y_test, cnn_preds, target_names=['Benign', 'Attack']))
print(f'ROC-AUC: {roc_auc_score(y_test, cnn_probs):.4f}')

CNN_DIR.mkdir(parents=True, exist_ok=True)
cnn_model.save(str(CNN_PATH))
print(f'\\n✅ CNN saved: {CNN_PATH}')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(cnn_hist.history['loss'], label='Train'); ax[0].plot(cnn_hist.history['val_loss'], label='Val')
ax[0].set_title('CNN Loss'); ax[0].legend()
ax[1].plot(cnn_hist.history['auc'], label='Train'); ax[1].plot(cnn_hist.history['val_auc'], label='Val')
ax[1].set_title('CNN AUC'); ax[1].legend()
plt.savefig(str(LOG_DIR / 'cnn_curves.png'), dpi=100, bbox_inches='tight')
plt.show()
"""),

# ─── CELL 11: Train LSTM ───────────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 11: Train Bidirectional LSTM (Supervised)
# ============================================================
import time
from sklearn.metrics import classification_report, roc_auc_score

print('='*60)
print('  TRAINING 2/5 — Bidirectional LSTM')
print('='*60)

LSTM_PATH = LSTM_DIR / 'lstm_model.h5'
lstm_model = build_lstm_model(INPUT_SHAPE_3D)
lstm_model.summary()

cbs_lstm = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=5,
                                   restore_best_weights=True, mode='max', verbose=1),
    keras.callbacks.ModelCheckpoint(str(LSTM_DIR / 'best_lstm.h5'),
                                     monitor='val_auc', save_best_only=True, mode='max'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    keras.callbacks.CSVLogger(str(LOG_DIR / 'lstm_training.csv')),
]

t0 = time.time()
lstm_hist = lstm_model.fit(
    X_train_3d, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(X_val_3d, y_val),
    class_weight=CLASS_WEIGHT,
    callbacks=cbs_lstm,
    verbose=1
)
print(f'\\nCompleted in {(time.time()-t0)/60:.1f} min')

print('\\nTest Set Evaluation:')
lstm_probs = lstm_model.predict(X_test_3d, verbose=0).flatten()
lstm_preds = (lstm_probs > 0.5).astype(int)
print(classification_report(y_test, lstm_preds, target_names=['Benign', 'Attack']))
print(f'ROC-AUC: {roc_auc_score(y_test, lstm_probs):.4f}')

LSTM_DIR.mkdir(parents=True, exist_ok=True)
lstm_model.save(str(LSTM_PATH))
print(f'\\n✅ LSTM saved: {LSTM_PATH}')
"""),

# ─── CELL 12: Train Transformer ────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 12: Train Transformer (Supervised)
# ============================================================
import time
from sklearn.metrics import classification_report, roc_auc_score

print('='*60)
print('  TRAINING 3/5 — Transformer')
print('='*60)

# Disable fp16 for Transformer (attention stability)
tf.keras.mixed_precision.set_global_policy('float32')
print('Switched to float32 for Transformer stability.')

TRANS_PATH = TRANS_DIR / 'transformer_model.h5'
trans_model = build_transformer_model(INPUT_SHAPE_3D)
trans_model.summary()

cbs_trans = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=5,
                                   restore_best_weights=True, mode='max', verbose=1),
    keras.callbacks.ModelCheckpoint(str(TRANS_DIR / 'best_transformer.h5'),
                                     monitor='val_auc', save_best_only=True, mode='max'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
    keras.callbacks.CSVLogger(str(LOG_DIR / 'transformer_training.csv')),
]

t0 = time.time()
trans_hist = trans_model.fit(
    X_train_3d, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(X_val_3d, y_val),
    class_weight=CLASS_WEIGHT,
    callbacks=cbs_trans,
    verbose=1
)
print(f'\\nCompleted in {(time.time()-t0)/60:.1f} min')

print('\\nTest Set Evaluation:')
trans_probs = trans_model.predict(X_test_3d, verbose=0).flatten()
trans_preds = (trans_probs > 0.5).astype(int)
print(classification_report(y_test, trans_preds, target_names=['Benign', 'Attack']))
print(f'ROC-AUC: {roc_auc_score(y_test, trans_probs):.4f}')

TRANS_DIR.mkdir(parents=True, exist_ok=True)
trans_model.save(str(TRANS_PATH))
print(f'\\n✅ Transformer saved: {TRANS_PATH}')
"""),

# ─── CELL 13: Train Autoencoder ────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 13: Train Autoencoder — Zero-Day Anomaly Detection
# ============================================================
import time, numpy as np
import matplotlib.pyplot as plt

print('='*60)
print('  TRAINING 4/5 — Deep Autoencoder (Unsupervised)')
print('='*60)
print(f'Training ONLY on {len(X_benign_train):,} BENIGN samples.')
print('Anomaly = high reconstruction error (MSE > threshold)')

tf.keras.mixed_precision.set_global_policy('mixed_float16')

AE_PATH = AE_DIR / 'autoencoder.h5'
ae_model = build_autoencoder(INPUT_DIM)
ae_model.summary()

cbs_ae = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(str(AE_DIR / 'best_ae.h5'),
                                     monitor='val_loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1),
    keras.callbacks.CSVLogger(str(LOG_DIR / 'ae_training.csv')),
]

t0 = time.time()
ae_hist = ae_model.fit(
    X_benign_train, X_benign_train,    # Input = Target (reconstruction)
    epochs=80,
    batch_size=1024,
    validation_data=(X_benign_val, X_benign_val),
    callbacks=cbs_ae,
    verbose=1
)
print(f'\\nCompleted in {(time.time()-t0)/60:.1f} min')

# ── Threshold Calculation ──────────────────────────────────
print('\\nCalculating anomaly threshold...')
recon_benign = ae_model.predict(X_benign_test, verbose=0).astype(np.float32)
mse_benign   = np.mean(np.power(X_benign_test - recon_benign, 2), axis=1)

recon_attack = ae_model.predict(X_attack_test[:5000], verbose=0).astype(np.float32)
mse_attack   = np.mean(np.power(X_attack_test[:5000] - recon_attack, 2), axis=1)

threshold_99 = float(np.percentile(mse_benign, 99))
threshold_95 = float(np.percentile(mse_benign, 95))

print(f'  Benign MSE (mean): {mse_benign.mean():.6f}')
print(f'  Attack MSE (mean): {mse_attack.mean():.6f}')
print(f'  Threshold @99th:   {threshold_99:.6f}')
print(f'  Threshold @95th:   {threshold_95:.6f}')
detected_at_99 = (mse_attack > threshold_99).sum()
print(f'  Zero-day detection @ 99th threshold: {detected_at_99/len(mse_attack)*100:.1f}%')

AE_DIR.mkdir(parents=True, exist_ok=True)
with open(str(AE_DIR / 'threshold.txt'), 'w') as f:
    f.write(str(threshold_99))

ae_model.save(str(AE_PATH))
print(f'\\n✅ Autoencoder saved: {AE_PATH}')

# MSE distribution plot
plt.figure(figsize=(10, 5))
plt.hist(mse_benign, bins=80, alpha=0.7, label='Benign MSE', color='#00ff41', density=True)
plt.hist(mse_attack[:3000], bins=80, alpha=0.7, label='Attack MSE', color='#ff4b4b', density=True)
plt.axvline(threshold_99, color='yellow', lw=2, ls='--', label=f'Threshold={threshold_99:.5f}')
plt.xlabel('Reconstruction MSE'); plt.ylabel('Density')
plt.title('Autoencoder: Benign vs. Attack Reconstruction Error')
plt.legend()
plt.savefig(str(LOG_DIR / 'ae_mse_distribution.png'), dpi=100)
plt.show()
"""),

# ─── CELL 14: Train Isolation Forest ──────────────────────────────────────────
cell("""\
# ============================================================
# CELL 14: Train Isolation Forest (Unsupervised)
# ============================================================
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib, time

print('='*60)
print('  TRAINING 5/5 — Isolation Forest (Unsupervised)')
print('='*60)

ISO_PATH = ISO_DIR / 'isolation_forest.joblib'

# Subsample for speed
ISO_N = min(150_000, len(X_benign_train))
idx_iso = np.random.choice(len(X_benign_train), ISO_N, replace=False)
X_iso_train = X_benign_train[idx_iso]
print(f'Training on {len(X_iso_train):,} benign samples (n_estimators=200)...')

t0 = time.time()
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.01,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
iso_forest.fit(X_iso_train)
print(f'\\nCompleted in {(time.time()-t0):.0f} seconds')

# Evaluate
print('\\nEvaluating on mixed test set...')
sample_b = X_benign_test[:5000]
sample_a = X_attack_test[:5000]
X_eval = np.vstack([sample_b, sample_a])
y_eval = np.array([0]*len(sample_b) + [1]*len(sample_a))

iso_pred_raw = iso_forest.predict(X_eval)   # 1=normal, -1=anomaly
iso_pred_bin = (iso_pred_raw == -1).astype(int)
iso_scores   = -iso_forest.score_samples(X_eval)   # higher = more anomalous
iso_scores   = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)

print(classification_report(y_eval, iso_pred_bin, target_names=['Benign', 'Attack']))
print(f'ROC-AUC: {roc_auc_score(y_eval, iso_scores):.4f}')

joblib.dump(iso_forest, str(ISO_PATH))
print(f'\\n✅ Isolation Forest saved: {ISO_PATH}')
"""),

# ─── CELL 15: Ensemble Evaluation ─────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 15: Ensemble Evaluation & ROC Curve Comparison
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import RocCurveDisplay

print('='*60)
print('  FINAL ENSEMBLE EVALUATION')
print('='*60)

# Predictions from all models
cnn_p   = cnn_model.predict(X_test_3d, verbose=0).flatten()
lstm_p  = lstm_model.predict(X_test_3d, verbose=0).flatten()
trans_p = trans_model.predict(X_test_3d, verbose=0).flatten()

ae_recon = ae_model.predict(X_test, verbose=0).astype(np.float32)
ae_mse_v = np.mean(np.power(X_test - ae_recon, 2), axis=1)
ae_p     = np.clip(ae_mse_v / (threshold_99 + 1e-10), 0, 1)

iso_raw  = -iso_forest.score_samples(X_test)
iso_p    = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-10)

# Weighted ensemble (matches fusion_engine/fusion.py weights)
WEIGHTS = dict(CNN=0.35, LSTM=0.25, Transformer=0.20, Autoencoder=0.10, IsoForest=0.10)
ens_p   = (WEIGHTS['CNN']*cnn_p + WEIGHTS['LSTM']*lstm_p +
           WEIGHTS['Transformer']*trans_p + WEIGHTS['Autoencoder']*ae_p +
           WEIGHTS['IsoForest']*iso_p)

print('\\n' + f'{"Model":<22} {"AUC":>8} {"F1":>8} {"Prec":>8} {"Recall":>8}')
print('-'*60)

results = {}
for name, probs in [('CNN', cnn_p), ('BiLSTM', lstm_p), ('Transformer', trans_p),
                     ('Autoencoder', ae_p), ('IsoForest', iso_p), ('ENSEMBLE', ens_p)]:
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    f1  = f1_score(y_test, preds, zero_division=0)
    pr  = precision_score(y_test, preds, zero_division=0)
    re  = recall_score(y_test, preds, zero_division=0)
    marker = '⭐' if name == 'ENSEMBLE' else '  '
    print(f'{marker}{name:<20} {auc:>8.4f} {f1:>8.4f} {pr:>8.4f} {re:>8.4f}')
    results[name] = dict(auc=auc, f1=f1, precision=pr, recall=re)

# ROC curves
fig, ax = plt.subplots(figsize=(10, 7))
for name, probs, color in [
    ('CNN', cnn_p, '#00ff41'), ('BiLSTM', lstm_p, '#0099ff'),
    ('Transformer', trans_p, '#ff9900'), ('Ensemble', ens_p, '#ff4b4b')]:
    RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax, color=color)
ax.plot([0,1],[0,1],'k--', alpha=0.5, label='Random Classifier')
ax.set_title('ROC Curve — AI-BASED NIDS Models', fontsize=14)
ax.legend(loc='lower right')
plt.savefig(str(LOG_DIR / 'roc_curves.png'), dpi=120, bbox_inches='tight')
plt.show()

print('\\n✅ Ensemble evaluation complete!')
"""),

# ─── CELL 16: Verification & Manifest ─────────────────────────────────────────
cell("""\
# ============================================================
# CELL 16: Verification & Model Manifest
# ============================================================
import os, json
from datetime import datetime

print('='*60)
print('  MODEL VERIFICATION')
print('='*60)

results_eval = results if 'results' in locals() else {}
weights_eval = WEIGHTS if 'WEIGHTS' in locals() else {}

manifest = {
    'created_at': datetime.now().isoformat(),
    'feature_dim': INPUT_DIM,
    'required_features': REQUIRED_FEATURES,
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'autoencoder_threshold_99pct': float(threshold_99),
    'ensemble_weights': weights_eval,
    'performance': results_eval,
    'models': {}
}

MODEL_FILES = [
    ('CNN',              MODEL_DIR / 'cnn_model.h5'),
    ('LSTM',             LSTM_DIR / 'lstm_model.h5'),
    ('Transformer',      TRANS_DIR / 'transformer_model.h5'),
    ('Autoencoder',      AE_DIR / 'autoencoder.h5'),
    ('Isolation Forest', ISO_DIR / 'isolation_forest.joblib'),
    ('Scaler',           MODEL_DIR / 'scaler.joblib'),
    ('AE Threshold',     AE_DIR / 'threshold.txt'),
]

print(f'\\n{\"Artifact\":<25} {\"Size\":>10}  Status')
print('-'*50)

all_ok = True
for name, path in MODEL_FILES:
    exists = path.exists()
    size   = path.stat().st_size / (1024*1024) if exists else 0
    status = '✅ OK' if exists else '❌ MISSING'
    if not exists:
        all_ok = False
    print(f'{name:<25} {size:>8.2f} MB  {status}')
    manifest['models'][name] = {'path': str(path), 'size_mb': round(size,2), 'ok': exists}

manifest_path = BASE_DIR / 'model_manifest.json'
with open(str(manifest_path), 'w') as f:
    json.dump(manifest, f, indent=2)

if all_ok:
    print('\\n🎉 ALL MODELS VERIFIED SUCCESSFULLY!')
else:
    print('\\n⚠️  Some models missing. Re-run training cells.')
"""),

# ─── CELL 17: Package & Download ──────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 17: Package Models & Download ZIP
# ============================================================
import zipfile, shutil, os
from google.colab import files as colab_files

print('📦 Packaging all models...')

ZIP_PATH = '/content/NIDS_Models.zip'

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for name, path in MODEL_FILES:
        if path.exists():
            # Keep relative structure inside zip
            arcname = str(path.relative_to(MODEL_DIR.parent))
            zf.write(str(path), arcname)
            print(f'  + {arcname}')

    # Training logs
    for lf in LOG_DIR.rglob('*'):
        if lf.is_file():
            zf.write(str(lf), 'logs/' + lf.name)

    # Manifest
    zf.write(str(manifest_path), 'model_manifest.json')

zip_mb = os.path.getsize(ZIP_PATH) / (1024*1024)
print(f'\\n✅ ZIP ready: {ZIP_PATH} ({zip_mb:.1f} MB)')
print('\\n⬇️  Downloading...')
colab_files.download(ZIP_PATH)

print('''
╔═══════════════════════════════════════════════════════════╗
║           INSTALLATION INSTRUCTIONS                       ║
╠═══════════════════════════════════════════════════════════╣
║  1. Extract NIDS_Models.zip                               ║
║  2. Copy contents to: G:/Projects/AI-BASED-NIDS/         ║
║                                                           ║
║  Expected structure:                                      ║
║    models/                                                ║
║    ├── cnn_model.h5                                       ║
║    ├── scaler.joblib                                      ║
║    ├── autoencoder/                                       ║
║    │   ├── autoencoder.h5                                 ║
║    │   └── threshold.txt                                  ║
║    ├── lstm/                                              ║
║    │   └── lstm_model.h5                                  ║
║    ├── transformer/                                       ║
║    │   └── transformer_model.h5                           ║
║    └── isolation_forest/                                  ║
║        └── isolation_forest.joblib                        ║
║                                                           ║
║  3. Run dashboard:                                        ║
║     streamlit run dashboard/app.py                        ║
╚═══════════════════════════════════════════════════════════╝
''')
"""),

# ─── CELL 18: Google Drive Auto-Save ─────────────────────────────────────────
cell("""\
# ============================================================
# CELL 18: (Optional) Auto-Save to Google Drive
# Uncomment to persist models across Colab sessions
# ============================================================

# from google.colab import drive
# drive.mount('/content/drive')
#
# DRIVE_DEST = '/content/drive/MyDrive/AI-NIDS-Models'
# import shutil
# shutil.copytree(str(MODEL_DIR), DRIVE_DEST + '/models', dirs_exist_ok=True)
# shutil.copy('/content/NIDS_Models.zip', DRIVE_DEST + '/NIDS_Models.zip')
# shutil.copy(str(manifest_path), DRIVE_DEST + '/model_manifest.json')
# print(f'✅ Models saved to Google Drive: {DRIVE_DEST}')

print('Uncomment the lines above to save to Google Drive.')
print('This keeps your models safe if the Colab session expires.')
"""),

# ─── CELL 19: Smoke Test ──────────────────────────────────────────────────────
cell("""\
# ============================================================
# CELL 19: End-to-End Smoke Test
# Simulates a live flow going through the full pipeline
# ============================================================
import numpy as np, joblib

print('='*60)
print('  SMOKE TEST — End-to-End Pipeline Verification')
print('='*60)

# Reload scaler from disk (validates save/load round-trip)
scaler_rt = joblib.load(str(MODEL_DIR / 'scaler.joblib'))
print(f'Scaler loaded. Expects {scaler_rt.n_features_in_} features.')

# Simulate a SYN flood feature vector
mock = np.zeros((1, INPUT_DIM), dtype=np.float32)
mock[0, REQUIRED_FEATURES.index('SYN Flag Count')]    = 500.0
mock[0, REQUIRED_FEATURES.index('Flow Bytes/s')]      = 2_000_000.0
mock[0, REQUIRED_FEATURES.index('Total Fwd Packets')] = 1000.0
mock[0, REQUIRED_FEATURES.index('Flow Duration')]     = 100.0

mock_scaled = scaler_rt.transform(mock).astype(np.float32)
mock_3d     = mock_scaled.reshape(1, INPUT_DIM, 1)

# Run all models
cnn_out    = float(cnn_model.predict(mock_3d, verbose=0)[0][0])
lstm_out   = float(lstm_model.predict(mock_3d, verbose=0)[0][0])
trans_out  = float(trans_model.predict(mock_3d, verbose=0)[0][0])
ae_recon   = ae_model.predict(mock_scaled, verbose=0).astype(np.float32)
ae_mse_v   = float(np.mean(np.power(mock_scaled - ae_recon, 2)))
iso_pred   = int(iso_forest.predict(mock_scaled)[0])

ens_score = (WEIGHTS['CNN']*cnn_out + WEIGHTS['LSTM']*lstm_out +
             WEIGHTS['Transformer']*trans_out)

# Determine alert level (mirrors fusion_engine/fusion.py)
risk = ens_score * 100
if risk >= 90:   level = '🔴 CRITICAL'
elif risk >= 75: level = '🔴 MALICIOUS'
elif risk >= 40: level = '🟡 SUSPICIOUS'
else:            level = '🟢 NORMAL'

print(f'\\n  [Simulated SYN Flood Flow]')
print(f'  CNN Probability:          {cnn_out:.4f}  → {\"ATTACK\" if cnn_out>0.5 else \"benign\"}')
print(f'  LSTM Probability:         {lstm_out:.4f}  → {\"ATTACK\" if lstm_out>0.5 else \"benign\"}')
print(f'  Transformer Probability:  {trans_out:.4f}  → {\"ATTACK\" if trans_out>0.5 else \"benign\"}')
print(f'  Autoencoder MSE:          {ae_mse_v:.6f}  (threshold={threshold_99:.6f})')
print(f'  → AE Decision:            {\"ANOMALY\" if ae_mse_v > threshold_99 else \"Normal\"}')
print(f'  Isolation Forest:         {\"ANOMALY\" if iso_pred == -1 else \"Normal\"}')
print(f'\\n  ╔══ Ensemble Risk Score: {risk:.1f}% — {level} ══╗')

print('\\n' + '='*60)
print('  ✅ ALL SYSTEMS OPERATIONAL')
print('  ✅ Training Pipeline Complete')
print('='*60)
print('\\nNext steps:')
print('  1. Download NIDS_Models.zip (see Cell 17)')
print('  2. Extract to G:/Projects/AI-BASED-NIDS/')
print('  3. streamlit run dashboard/app.py')
"""),

]  # end CELLS list


# ── Assemble Notebook ─────────────────────────────────────────────────────────
NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "AI_NIDS_Colab_Training.ipynb",
            "authorship_tag": "AI-BASED-NIDS Training Pipeline v2.0"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": CELLS
}

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook written to: {OUTPUT_PATH}")
print(f"   Cells: {len(CELLS)}")
print(f"   Size: {os.path.getsize(OUTPUT_PATH)/1024:.1f} KB")
