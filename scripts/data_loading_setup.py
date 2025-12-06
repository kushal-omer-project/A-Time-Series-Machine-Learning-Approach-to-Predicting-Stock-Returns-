#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data_loader import DataLoader
from loguru import logger

def main():
    print("Stock Market Prediction Engine - Setup")
    print("=" * 50)
    
    Config.create_directories()
    print("Project directories created")
    
    loader = DataLoader()
    print("Data loader initialized")
    
    available_datasets = loader.list_available_datasets()
    print(f"\nAvailable datasets: {', '.join(available_datasets)}")
    
    downloaded_datasets = loader.list_downloaded_datasets()
    print(f"Downloaded datasets: {', '.join(downloaded_datasets) if downloaded_datasets else 'None'}")
    
    print("\nSetup complete. Next steps:")
    print("1. Setup Kaggle API credentials")
    print("2. Run data_download_exploration.py to download data")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download-data":
        print("Data download functionality available in data_download_exploration.py")
    else:
        main()