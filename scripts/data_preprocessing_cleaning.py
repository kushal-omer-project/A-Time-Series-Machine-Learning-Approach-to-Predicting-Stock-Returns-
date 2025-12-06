#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from loguru import logger
import pandas as pd
from datetime import datetime

def process_and_clean_data():
    print("\nStarting Data Processing and Cleaning...")
    processor = DataProcessor()
    print("\nLoading and cleaning world stocks dataset...")
    cleaned_df = processor.load_and_clean_world_stocks()
    
    if cleaned_df.empty:
        print("Failed to load or clean the dataset")
        return None
    
    print(f"Dataset cleaned: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
    
    print("\nAnalyzing stock data coverage...")
    analysis_df = processor.analyze_stock_coverage(cleaned_df)
    
    print("\nStock Analysis Summary:")
    print(f"   Total stocks analyzed: {len(analysis_df)}")
    print(f"   Average records per stock: {analysis_df['total_records'].mean():.0f}")
    print(f"   Date range: {analysis_df['date_range_start'].min()} to {analysis_df['date_range_end'].max()}")
    
    print("\nTop 5 Stocks by Data Volume:")
    top_5 = analysis_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"   {row['ticker']} ({row['brand_name']}): {row['total_records']:,} records")
    
    # Select target stocks for modeling
    print("\nSelecting target stocks for prediction modeling...")
    target_stocks = processor.select_target_stocks(analysis_df)
    
    if not target_stocks:
        print("WARNING: No stocks met the minimum criteria")
        return None
    
    print(f"Selected {len(target_stocks)} target stocks for modeling")
    
    # Create comprehensive visualizations
    print("\nCreating comprehensive visualizations...")
    processor.create_comprehensive_visualizations(cleaned_df, target_stocks)
    
    # Save all processed data
    print("\nSaving processed data...")
    saved_files = processor.save_cleaned_data(cleaned_df, analysis_df, target_stocks)
    
    print("Saved files:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    return {
        'cleaned_data': cleaned_df,
        'analysis': analysis_df,
        'target_stocks': target_stocks,
        'saved_files': saved_files
    }

def main():
    print("Stock Market Prediction Engine - Data Preprocessing")
    print("=" * 50)
    
    loader = DataLoader()
    downloaded_datasets = loader.list_downloaded_datasets()
    
    if 'world_stocks' not in downloaded_datasets:
        print("\nWorld stocks dataset not found!")
        print("Please run data_download_exploration.py first")
        return
    
    results = process_and_clean_data()
    if results is None:
        print("\nData processing failed!")
        return
    
    cleaned_df = results['cleaned_data']
    target_stocks = results['target_stocks']
    
    print("\nData preprocessing completed!")
    print(f"\nFinal Dataset Summary:")
    print(f"   Records: {len(cleaned_df):,}")
    print(f"   Stocks: {cleaned_df['Ticker'].nunique()}")
    print(f"   Date range: {cleaned_df['Date'].min().date()} to {cleaned_df['Date'].max().date()}")
    print(f"   Target stocks: {', '.join(target_stocks[:5])}{'...' if len(target_stocks) > 5 else ''}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download-data":
        print("Data already downloaded in Day 2. Running Day 3 processing...")
    
    main()