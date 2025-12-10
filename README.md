# A Time-Series Machine Learning Approach to Predicting Stock Returns​

> **Machine Learning System for Stock Market Analysis and Prediction using XGBoost and LightGBM**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Repository**: [https://github.com/kushalX13/A-Time-Series-Machine-Learning-Approach-to-Predicting-Stock-Returns-](https://github.com/kushalX13/A-Time-Series-Machine-Learning-Approach-to-Predicting-Stock-Returns-)

## Project Overview

A comprehensive machine learning system designed to predict stock market movements using advanced feature engineering and gradient boosting models. This project focuses on comparing XGBoost and LightGBM models for stock return prediction, demonstrating end-to-end data science capabilities from data acquisition to model deployment.

### Key Features

- **73 Optimized Features** - Technical indicators and custom engineered features
- **XGBoost & LightGBM Models** - Hyperparameter-optimized gradient boosting models
- **Time-Series Validation** - Walk-forward validation and out-of-sample testing
- **Risk Management** - Comprehensive risk metrics including Sharpe Ratio, Max Drawdown, and VaR
- **Interactive Dashboard** - Streamlit-based dashboard for predictions and performance analysis


## System Performance

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Sharpe Ratio** | 1.497 | Excellent (>1.0) |
| **Annual Return** | 14.4% | Market Average ~10% |
| **Win Rate** | 59-65% | Above Random (50%) |
| **Max Drawdown** | -14.6% | Acceptable (<20%) |
| **Prediction Speed** | <3 seconds | Real-time capable |

## Installation

### Prerequisites

- **Python 3.11+** installed on your system
- **4GB+ RAM** for ML model loading
- **Kaggle API credentials** (for data download)

### Step 1: Clone the Repository

```bash
git clone https://github.com/kushalX13/A-Time-Series-Machine-Learning-Approach-to-Predicting-Stock-Returns-.git
cd A-Time-Series-Machine-Learning-Approach-to-Predicting-Stock-Returns-

```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Kaggle API (Required for Data Download)

1. **Create a Kaggle account** at [kaggle.com](https://www.kaggle.com)

2. **Get your API credentials**:
   - Go to your Kaggle account settings
   - Scroll to the "API" section
   - Click "Create New API Token"
   - This downloads a `kaggle.json` file

3. **Set up Kaggle credentials**:
   
   **Option A: Environment Variables (Recommended)**
   ```bash
   export KAGGLE_USERNAME="your_kaggle_username"
   export KAGGLE_KEY="your_kaggle_api_key"
   ```
   
   **Option B: Create .env file**
   ```bash
   # Create .env file in project root
   echo "KAGGLE_USERNAME=your_kaggle_username" > .env
   echo "KAGGLE_KEY=your_kaggle_api_key" >> .env
   ```

### Step 4: Download Required Datasets

The project requires stock market data from Kaggle. Run the data download script:

```bash
python -c "from src.data_loader import DataLoader; loader = DataLoader(); loader.download_all_datasets()"
```

**Or manually download datasets:**

The project uses the following Kaggle datasets:
- **World Stock Prices**: `nelgiriyewithana/world-stock-prices-daily-updating`

**To download manually:**
1. Visit the dataset pages on Kaggle
2. Click "Download" (requires accepting dataset terms)
3. Extract the CSV files to `data/raw/world_stocks/`

**Expected data structure after download:**
```
data/
├── raw/
│   ├── world_stocks/
│       └── World-Stock-Prices-Dataset.csv
│   
│ 
├── processed/  (will be generated)
└── features/   (will be generated)
```

## Usage

### Running the Complete Pipeline

Run the scripts in the `scripts/` directory in the following order:

1. **Initial Setup & Data Loading Configuration**:
   ```bash
   python scripts/data_loading_setup.py
   ```
   This script:
   - Creates necessary project directories
   - Initializes the data loader
   - Verifies Kaggle API setup
   - Lists available datasets

2. **Download Data from Kaggle**:
   ```bash
   python scripts/data_download_exploration.py
   ```
   This script:
   - Downloads required datasets from Kaggle (world_stocks)
   - Explores the downloaded data
   - Identifies target stocks
   - Creates initial visualizations
   - Saves data summary

3. **Data Preprocessing and Cleaning**:
   ```bash
   python scripts/data_preprocessing_cleaning.py
   ```
   This script:
   - Loads and cleans the raw stock data
   - Handles missing values and outliers
   - Analyzes stock data coverage
   - Selects target stocks for modeling
   - Saves cleaned dataset

4. **Feature Engineering**:
   ```bash
   python scripts/feature_engineering.py
   ```
   This script:
   - Creates technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Generates 73 optimized features
   - Performs feature selection
   - Creates feature correlation analysis
   - Saves engineered features

5. **Baseline ML Models**:
   ```bash
   python scripts/baseline_ml_models.py
   ```
   This script:
   - Trains baseline regression and classification models
   - Performs time-series cross-validation
   - Evaluates model performance
   - Saves baseline model results

6. **Advanced ML Models with Hyperparameter Optimization**:
   ```bash
   python scripts/advanced_ml_hyperparameter_optimization.py
   ```
   This script:
   - Trains XGBoost and LightGBM models
   - Performs hyperparameter optimization using Optuna
   - Compares model performance
   - Saves optimized models

7. **Model Validation and Backtesting**:
   ```bash
   python scripts/model_validation_backtesting.py
   ```
   This script:
   - Performs comprehensive model validation
   - Runs backtesting on historical data
   - Calculates performance metrics
   - Generates validation reports

8. **Risk Management Analysis**:
   ```bash
   python scripts/risk_management.py
   ```
   This script:
   - Calculates risk metrics (Sharpe Ratio, Max Drawdown, VaR)
   - Performs portfolio optimization
   - Generates risk analysis reports


9. **Run Integration Tests**:
   ```bash
   python main.py
   ```
   This script:
   - Runs comprehensive system integration tests
   - Validates system health
   - Performs system optimization
   - Generates final report

### Running the Dashboard

```bash
streamlit run src/streamlit_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Running the API Server

```bash
python src/api_server.py
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

## Project Structure

```
stock-market-prediction-engine/
├── src/                  # Core source code
│   ├── config.py        # Configuration settings
│   ├── data_loader.py   # Data acquisition from Kaggle
│   ├── data_processor.py # Data preprocessing
│   ├── feature_engineer.py # Feature engineering
│   ├── ml_models.py     # Baseline models
│   ├── advanced_models.py # XGBoost & LightGBM
│   ├── validation_framework.py # Model validation
│   ├── risk_management.py # Risk analysis
│   ├── streamlit_dashboard.py # Interactive dashboard
│   └── api_server.py    # REST API
├── data/                 # Data storage (must be downloaded)
│   ├── raw/            # Raw datasets from Kaggle
│   ├── processed/      # Processed data (generated)
│   └── features/       # Engineered features (generated)
├── models/              # Trained models (generated)
│   └── advanced/       # Optimized models
├── scripts/             # Pipeline execution scripts
├── logs/                # Application logs
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technology Stack

### Machine Learning & Data Science
- **Python 3.11+** - Core development language
- **XGBoost** - Gradient boosting framework with hyperparameter optimization
- **LightGBM** - Light gradient boosting machine for efficient training
- **Optuna** - Bayesian hyperparameter optimization
- **scikit-learn** - Model evaluation and preprocessing utilities
- **pandas & NumPy** - Data manipulation and numerical computing
- **TA-Lib** - Technical analysis indicators

### Data Sources
- **Kaggle API** - Historical dataset acquisition

## Model Performance

### Model Comparison
- **XGBoost**: Optimized gradient boosting model
- **LightGBM**: Light gradient boosting machine
- **Training Data**: 307K+ records, multiple years of market data
- **Validation Method**: Time-series cross-validation with walk-forward analysis
- **Feature Selection**: 73 optimal features from 124 engineered indicators

### Risk-Adjusted Performance
- **Sharpe Ratio**: 1.497 (excellent performance)
- **Maximum Drawdown**: -14.6% (controlled risk)
- **Win Rate**: 59-65% (consistent profitability)
- **Annual Return**: 14.4% (above market average)

## Features

### Model Predictions
- Real-time stock price predictions with confidence intervals
- 5-day return forecasting
- XGBoost and LightGBM model comparison
- Feature importance analysis and model interpretability

### Risk Management
- Value at Risk (VaR) and Conditional VaR calculations
- Maximum drawdown analysis and monitoring
- Sharpe and Sortino ratio calculations
- Risk-adjusted performance metrics

### Interactive Dashboard
- Real-time prediction interface with live market data
- Performance analytics and model comparison
- Risk metrics visualization and monitoring
- Model insights and feature importance



## Troubleshooting

### Data Download Issues

**Problem**: Kaggle API authentication fails
- **Solution**: Verify your `KAGGLE_USERNAME` and `KAGGLE_KEY` are set correctly
- Check that you've accepted the dataset terms on Kaggle

**Problem**: Dataset not found
- **Solution**: Ensure dataset names match exactly:
  - `nelgiriyewithana/world-stock-prices-daily-updating`

### Model Training Issues

**Problem**: Models fail to load
- **Solution**: Ensure you've run the complete pipeline:
  1. Data preprocessing
  2. Feature engineering
  3. Model training

**Problem**: Out of memory errors
- **Solution**: Reduce batch size or use a machine with more RAM (4GB+ recommended)

## Important Notes

### Data Files
- **Data files are NOT included** in this repository due to size limitations
- All data must be downloaded from Kaggle using the instructions above
- Processed data and features will be generated when you run the pipeline
- Model files will be generated during training

### Model Files
- Trained model files are large and are NOT included in the repository
- Models must be trained by running the training scripts
- Training typically takes 30-60 minutes depending on your hardware

## License and Disclaimer

This project is for educational and portfolio demonstration purposes.

**Important Notice**: 
- This is not financial advice
- Past performance doesn't guarantee future results
- Always consult with financial professionals before making investment decisions
- Use this system at your own risk

---

<div align="center">

**Built for the data science and finance community**

[View on GitHub](https://github.com/kushalX13/A-Time-Series-Machine-Learning-Approach-to-Predicting-Stock-Returns-)

</div>
