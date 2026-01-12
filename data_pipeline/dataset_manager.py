import os
import requests
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_DIR = Path(r"g:\Projects\AI-BASED-NIDS\dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs / Kaggle IDs
datasets = {
    "CIC-IDS2017": {
        "type": "kaggle",
        "id": "dhoogla/cicids2017", # Mirror to avoid 403 on original
        "folder": "CIC-IDS2017"
    },
    "CSE-CIC-IDS2018": {
        "type": "kaggle",
        "id": "solarmainframe/ids-intrusion-csv", # A reliable processed version on Kaggle
        "folder": "CSE-CIC-IDS2018"
    },
    "CIC-DDoS2019": {
        "type": "kaggle",
        "id": "cicdataset/cicddos2019",
        "folder": "CIC-DDoS2019"
    },
     "UNSW-NB15": {
        "type": "kaggle",
        "id": "mrwellsdouglas/unsw-nb15",
        "folder": "UNSW-NB15"
    }
    # CTU-13 is tricky as it's often hosted on university servers with varying links.
    # We will prioritize the main CIC datasets first via Kaggle API which is reliable.
}

def download_file(url, dest_path):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def download_kaggle_dataset(dataset_id, dest_folder):
    """Download dataset using Kaggle API."""
    print(f"Downloading {dataset_id} from Kaggle...")
    try:
        # Check if kaggle is installed
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'kaggle' command not found. Please install: pip install kaggle")
        print("Also ensure you have 'kaggle.json' in C:\\Users\\...\\.kaggle\\")
        return

    dest_path = DATASET_DIR / dest_folder
    dest_path.mkdir(exist_ok=True)
    
    command = f"kaggle datasets download -d {dataset_id} -p \"{dest_path}\" --unzip"
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully downloaded {dataset_id} to {dest_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {dataset_id}: {e}")

def main():
    print("=== AI-NIDS Dataset Manager ===")
    print(f"Target Directory: {DATASET_DIR}")
    
    for name, info in datasets.items():
        if (DATASET_DIR / info["folder"]).exists() and any((DATASET_DIR / info["folder"]).iterdir()):
             print(f"Skipping {name} (Folder exists and is not empty)")
             continue

        if info["type"] == "kaggle":
            download_kaggle_dataset(info["id"], info["folder"])
        elif info["type"] == "direct":
            # Placeholder for direct generic downloads if needed
            pass
            
    print("\nCheck CTU-13 manual download if needed: https://www.stratosphereips.org/datasets-ctu13")
    print("=== Done ===")

if __name__ == "__main__":
    main()
