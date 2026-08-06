"""
test_notebook_execution.py
--------------------------
Simulates execution of all code cells in AI_NIDS_Colab_Training.ipynb to find any runtime errors or undefined variable references.
"""
import json, sys, os

with open('AI_NIDS_Colab_Training.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("=== MOCK TESTING NOTEBOOK EXECUTION ===")

global_scope = {}

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    # Skip pip magics or shell lines
    clean_lines = []
    for line in source.splitlines():
        if line.strip().startswith(('%', '!')):
            continue
        clean_lines.append(line)
    
    clean_code = "\n".join(clean_lines)
    print(f"\n---> Executing Cell {i}...")
    try:
        exec(clean_code, global_scope)
        print(f"✅ Cell {i} executed without syntax or import errors in scope.")
    except Exception as e:
        print(f"⚠️ Cell {i} raised error: {type(e).__name__}: {e}")
