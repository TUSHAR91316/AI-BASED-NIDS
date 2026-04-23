import os
import requests
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_DIR = Path(r"g:\Projects\AI-BASED-NIDS\dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs / Kaggle IDs - Enhanced with better, more recent datasets
datasets = {
    # === PRIMARY DATASETS (High Quality, CIC-IDS Compatible) ===
    "CIC-IDS2017": {
        "type": "kaggle",
        "id": "dhoogla/cicids2017",
        "folder": "CIC-IDS2017",
        "priority": 1,
        "description": "Classic CIC-IDS2017 with 14 attacks"
    },
    "CSE-CIC-IDS2018": {
        "type": "kaggle",
        "id": "solarmainframe/ids-intrusion-csv",
        "folder": "CSE-CIC-IDS2018",
        "priority": 1,
        "description": "Large-scale 10-day dataset with modern attacks"
    },
    "CIC-DDoS2019": {
        "type": "kaggle",
        "id": "cicdataset/cicddos2019",
        "folder": "CIC-DDoS2019",
        "priority": 2,
        "description": "Specialized DDoS attacks (LDoS, DrDoS)"
    },
    
    # === MODERN IOT DATASETS (Critical for 2024+ NIDS) ===
    "CIC-TON-IoT": {
        "type": "kaggle",
        "id": "mahmouddaadan/cictoniot-network-intrusion-dataset",
        "folder": "CIC-TON-IoT",
        "priority": 1,
        "description": "IoT/IIoT focused with 9 attack categories"
    },
    "CIC-IoT-2023": {
        "type": "kaggle",
        "id": "cicdataset/cic-iot-2023-dataset",
        "folder": "CIC-IoT-2023",
        "priority": 1,
        "description": "Latest 2023 dataset with 33 IoT devices, 100+ attack scenarios"
    },
    
    # === BALANCED CLASSIC DATASET ===
    "NSL-KDD": {
        "type": "kaggle",
        "id": "hassan06/nslkdd",
        "folder": "NSL-KDD",
        "priority": 2,
        "description": "Improved KDD'99, balanced classes, no redundant records"
    },
    
    # === SPECIALIZED ATTACK DATASETS ===
    "CIC-Bell-DNS-2021": {
        "type": "kaggle",
        "id": "cicdataset/cic-bell-dns-2021",
        "folder": "CIC-Bell-DNS-2021",
        "priority": 3,
        "description": "DNS tunneling and DGA-based attacks"
    },
    "CIC-Darknet2020": {
        "type": "kaggle",
        "id": "cicdataset/cicdarknet2020",
        "folder": "CIC-Darknet2020",
        "priority": 3,
        "description": "Darknet traffic classification (Tor, VPN, non-VPN)"
    },
    
    # === EXISTING ===
    "UNSW-NB15": {
        "type": "kaggle",
        "id": "mrwellsdouglas/unsw-nb15",
        "folder": "UNSW-NB15",
        "priority": 2,
        "description": "UNSW-NB15 with 9 modern attack categories"
    }
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

def download_priority_datasets(max_priority=2, skip_existing=True):
    """Download datasets by priority level (1=essential, 2=important, 3=specialized)"""
    print(f"=== AI-NIDS Dataset Manager (Priority <= {max_priority}) ===")
    print(f"Target Directory: {DATASET_DIR}")
    print()
    
    # Sort by priority
    sorted_datasets = sorted(datasets.items(), key=lambda x: x[1].get("priority", 99))
    
    downloaded = []
    skipped = []
    failed = []
    
    for name, info in sorted_datasets:
        priority = info.get("priority", 99)
        if priority > max_priority:
            continue
            
        print(f"[{priority}] {name}: {info.get('description', 'No description')}")
        
        if skip_existing and (DATASET_DIR / info["folder"]).exists() and any((DATASET_DIR / info["folder"]).iterdir()):
            print(f"   ⚡ Skipping (already exists)")
            skipped.append(name)
            continue

        if info["type"] == "kaggle":
            download_kaggle_dataset(info["id"], info["folder"])
            downloaded.append(name)
        elif info["type"] == "direct":
            pass
    
    print("\n" + "="*50)
    print(f"Downloaded: {len(downloaded)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
    print("\nRECOMMENDED MINIMUM: Priority 1 datasets (CIC-IDS2017, CIC-IDS2018, CIC-TON-IoT, CIC-IoT-2023)")
    print("BEST RESULTS: Priority 1 + Priority 2 datasets")
    print("\nFor manual CTU-13: https://www.stratosphereips.org/datasets-ctu13")
    print("="*50)

def main():
    """Run interactive or default download"""
    import sys
    
    # Check for command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            download_priority_datasets(max_priority=3)
        elif sys.argv[1] == "--essential":
            download_priority_datasets(max_priority=1)
        elif sys.argv[1] == "--recommended":
            download_priority_datasets(max_priority=2)
        else:
            print("Usage: python dataset_manager.py [--essential|--recommended|--all]")
            print("  --essential: Priority 1 datasets only (minimum for good results)")
            print("  --recommended: Priority 1+2 datasets (best balance)")
            print("  --all: All datasets including specialized ones")
    else:
        # Default: recommended priority level
        download_priority_datasets(max_priority=2)

if __name__ == "__main__":
    main()
