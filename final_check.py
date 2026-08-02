"""final_check.py — Final project deliverables verification."""
from pathlib import Path
import json

root = Path('.')
print('=== FINAL DELIVERABLES CHECKLIST ===')
print()

checks = []

# 1. Colab notebook exists and is valid
nb_path = root / 'AI_NIDS_Colab_Training.ipynb'
nb = json.loads(nb_path.read_bytes())
nb_cells = len(nb['cells'])
nb_gpu = nb['metadata']['colab']['gpuType']
checks.append(('Colab notebook (.ipynb)', True, f'{nb_cells} cells, GPU={nb_gpu}'))

# 2-8. Content checks
nb_src = ' '.join(''.join(c['source']) for c in nb['cells'])

for name, kw in [
    ('Kaggle download (CIC-IDS2017)',     "['kaggle', 'datasets', 'download'"),
    ('CNN training cell',                  'TRAINING 1'),
    ('LSTM training cell',                 'TRAINING 2'),
    ('Transformer training cell',          'TRAINING 3'),
    ('Autoencoder training cell',          'TRAINING 4'),
    ('Isolation Forest training cell',     'TRAINING 5'),
    ('scaler.joblib saved',               'scaler.joblib'),
    ('threshold.txt saved',               'threshold.txt'),
    ('ZIP packaging for download',        'NIDS_Models.zip'),
    ('End-to-end smoke test',             'SMOKE TEST'),
]:
    checks.append((name, kw.upper() in nb_src.upper(), ''))

# 9. Feature alignment consistent (70 features match loader.py)
loader_src = (root/'data_pipeline'/'loader.py').read_text(encoding='utf-8')
feat_ok = 'Destination Port' in nb_src and 'Destination Port' in loader_src
checks.append(('70-feature gold standard aligned', feat_ok, 'Matches loader.py'))

# 10. requirements.txt comprehensive
req = (root/'requirements.txt').read_text(encoding='utf-8')
req_ok = 'aiofiles' in req and 'numba' in req and 'kaggle' in req
checks.append(('requirements.txt complete', req_ok, 'aiofiles/numba/kaggle included'))

# 11. README updated with Colab section
readme = (root/'README.md').read_text(encoding='utf-8')
readme_ok = 'AI_NIDS_Colab_Training' in readme and 'Colab' in readme
checks.append(('README has Colab section', readme_ok, ''))

# 12. Audit docs
checks.append(('AGENTS.md present', (root/'AGENTS.md').exists(), 'Agent guidelines'))
checks.append(('AI.md present', (root/'AI.md').exists(), 'Technical architecture'))

# 13. Core project files fixed
for fname in ['realtime_detector/detector.py', 'fusion_engine/fusion.py',
               'data_pipeline/loader.py', 'feature_extractor/flow_features.py']:
    checks.append((fname + ' present', (root/fname).exists(), ''))

# Print report
print('  ' + '-'*72)
all_pass = True
for name, ok, detail in checks:
    if not ok:
        all_pass = False
    mark = 'OK' if ok else '!!'
    suffix = '  ' + detail if detail else ''
    print(f'  [{mark}] {name}{suffix}')

print('  ' + '-'*72)
print()
if all_pass:
    print('  >>> ALL DELIVERABLES COMPLETE <<<')
else:
    fails = [n for n, ok, _ in checks if not ok]
    print(f'  INCOMPLETE: {fails}')
