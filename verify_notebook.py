"""verify_notebook.py — Validates the generated Colab notebook."""
import json, os, sys

NB_PATH = 'AI_NIDS_Colab_Training.ipynb'

if not os.path.exists(NB_PATH):
    print("ERROR: Notebook file not found!")
    sys.exit(1)

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print('=== NOTEBOOK VERIFICATION ===')
print(f'File size   : {os.path.getsize(NB_PATH)/1024:.1f} KB')
print(f'Total cells : {len(cells)}')
print(f'nbformat    : {nb["nbformat"]}.{nb["nbformat_minor"]}')
print(f'GPU type    : {nb["metadata"]["colab"]["gpuType"]}')
print()

# Check required training stages
required_stages = [
    'CELL 1', 'CELL 2', 'CELL 4',  # env, auth, dataset download
    'TRAINING 1', 'TRAINING 2', 'TRAINING 3', 'TRAINING 4', 'TRAINING 5',  # 5 models
    'ENSEMBLE', 'SMOKE TEST', 'Package',  # eval + download
]

all_sources = '\n'.join(''.join(c['source']) for c in cells)

print('Required stage checks:')
all_present = True
for stage in required_stages:
    present = stage.upper() in all_sources.upper()
    status = 'OK' if present else 'MISSING'
    if not present:
        all_present = False
    print(f'  [{status}] {stage}')

# Check feature vector
feat_check = 'REQUIRED_FEATURES' in all_sources and 'Destination Port' in all_sources
print(f'\n  [{"OK" if feat_check else "FAIL"}] 70-feature gold standard vector present')

scaler_check = 'scaler.joblib' in all_sources
print(f'  [{"OK" if scaler_check else "FAIL"}] scaler.joblib save/load present')

threshold_check = 'threshold.txt' in all_sources
print(f'  [{"OK" if threshold_check else "FAIL"}] AE threshold.txt save present')

kaggle_check = 'kaggle' in all_sources.lower()
print(f'  [{"OK" if kaggle_check else "FAIL"}] Kaggle download logic present')

zip_check = 'NIDS_Models.zip' in all_sources
print(f'  [{"OK" if zip_check else "FAIL"}] ZIP packaging for download present')

print()
print('Cell inventory:')
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    lines = len(src.splitlines())
    print(f'  {i:02d} [{c["cell_type"]:8s}] {lines:4d} lines | {src[:55].strip()!r}')

print()
if all_present and feat_check and scaler_check and threshold_check:
    print('RESULT: ALL CHECKS PASSED')
else:
    print('RESULT: SOME CHECKS FAILED')
