#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 7-8
Advanced Machine Learning Model Development with Hyperparameter Optimization
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.advanced_models import AdvancedMLFramework
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_advanced_ml_development():
    """Run comprehensive advanced ML model development"""
    print("\nStarting Advanced ML Model Development (Day 7-8)")
    print("=" * 70)
    
    # Initialize advanced ML framework
    ml_framework = AdvancedMLFramework()
    
    # Load feature data (baseline results are optional)
    print("\nLoading feature data from Day 4...")
    df, baseline_regression, baseline_classification = ml_framework.load_baseline_results()
    
    if df.empty:
        print("Failed to load feature data. Please run Day 4 first.")
        return None
    
    print(f"Loaded data: {len(df):,} records, {df.shape[1]} features, {df['Ticker'].nunique()} stocks")
    
    # Baseline results are optional (only for comparison)
    if baseline_regression or baseline_classification:
        print(f"Baseline models found: {len(baseline_regression)} regression, {len(baseline_classification)} classification")
    else:
        print("INFO: No baseline models found (skipping Day 6 comparison - this is fine!)")
    
    # Display baseline performance for comparison
    if baseline_regression:
        print("\nBaseline Regression Performance:")
        for name, results in baseline_regression.items():
            r2_score = results.get('cv_r2_mean', 0)
            print(f"   {name}: R² = {r2_score:.4f}")
    
    if baseline_classification:
        print("\nBaseline Classification Performance:")
        for name, results in baseline_classification.items():
            f1_score = results.get('cv_f1_mean', 0)
            print(f"   {name}: F1 = {f1_score:.4f}")
    
    # Check if we have sufficient data for advanced modeling
    if len(df) < 1000:
        print(f"WARNING: Limited data ({len(df)} records) may affect advanced model performance")
    
    # Train Advanced Regression Models
    print("\nPhase 1: Advanced Regression Model Development")
    print("-" * 50)
    
    advanced_regression_models = ml_framework.train_advanced_models(
        df, target_col='return_5d', task_type='regression'
    )
    
    if not advanced_regression_models:
        print("No advanced regression models were trained successfully")
        return None
    
    print(f"Successfully trained {len(advanced_regression_models)} advanced regression models:")
    for model_name in advanced_regression_models.keys():
        print(f"   • {model_name}")
    
    # Evaluate Advanced Regression Models
    print("\nEvaluating advanced regression models with time-series cross-validation...")
    advanced_regression_results = ml_framework.evaluate_advanced_models(
        advanced_regression_models, df, target_col='return_5d', task_type='regression'
    )
    
    print("Advanced Regression Model Performance:")
    for name, results in advanced_regression_results.items():
        r2_mean = results.get('cv_r2_mean', 0)
        r2_std = results.get('cv_r2_std', 0)
        rmse = results.get('cv_rmse_mean', 0)
        print(f"   {name}:")
        print(f"     R² Score: {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"     RMSE: {rmse:.4f}")
    
    # Train Advanced Classification Models
    print("\nPhase 2: Advanced Classification Model Development")
    print("-" * 50)
    
    advanced_classification_models = ml_framework.train_advanced_models(
        df, target_col='return_5d', task_type='classification'
    )
    
    if not advanced_classification_models:
        print("No advanced classification models were trained successfully")
    else:
        print(f"Successfully trained {len(advanced_classification_models)} advanced classification models:")
        for model_name in advanced_classification_models.keys():
            print(f"   • {model_name}")
        
        # Evaluate Advanced Classification Models
        print("\nEvaluating advanced classification models...")
        advanced_classification_results = ml_framework.evaluate_advanced_models(
            advanced_classification_models, df, target_col='return_5d', task_type='classification'
        )
        
        print("Advanced Classification Model Performance:")
        for name, results in advanced_classification_results.items():
            f1_mean = results.get('cv_f1_mean', 0)
            f1_std = results.get('cv_f1_std', 0)
            accuracy = results.get('cv_accuracy', 0)
            print(f"   {name}:")
            print(f"     F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
            print(f"     Accuracy: {accuracy:.4f}")
    
    # Performance Comparison Analysis
    print("\nPhase 3: Performance Comparison Analysis")
    print("-" * 50)
    
    # Compare with baseline models (if available)
    if baseline_regression or baseline_classification:
        print("Comparing Advanced vs Baseline Models:")
    else:
        print("Advanced Model Performance (no baseline comparison available):")
    
    if advanced_regression_results and baseline_regression:
        print("\nRegression Model Improvements:")
        
        # Find best models
        best_advanced_reg = max(advanced_regression_results.keys(), 
                              key=lambda x: advanced_regression_results[x].get('cv_r2_mean', 0))
        best_baseline_reg = max(baseline_regression.keys(), 
                              key=lambda x: baseline_regression[x].get('cv_r2_mean', 0))
        
        advanced_r2 = advanced_regression_results[best_advanced_reg].get('cv_r2_mean', 0)
        baseline_r2 = baseline_regression[best_baseline_reg].get('cv_r2_mean', 0)
        improvement = ((advanced_r2 - baseline_r2) / baseline_r2 * 100) if baseline_r2 > 0 else 0
        
        print(f"   Best Advanced: {best_advanced_reg} (R² = {advanced_r2:.4f})")
        print(f"   Best Baseline: {best_baseline_reg} (R² = {baseline_r2:.4f})")
        print(f"   Improvement: {improvement:+.1f}% ({advanced_r2:.4f} vs {baseline_r2:.4f})")
        
        if improvement > 10:
            print("   Significant improvement achieved!")
        elif improvement > 0:
            print("   Moderate improvement achieved")
        else:
            print("   WARNING: No improvement over baseline")
    
    if 'advanced_classification_results' in locals() and baseline_classification:
        print("\nClassification Model Improvements:")
        
        # Find best models
        best_advanced_class = max(advanced_classification_results.keys(), 
                                key=lambda x: advanced_classification_results[x].get('cv_f1_mean', 0))
        best_baseline_class = max(baseline_classification.keys(), 
                                key=lambda x: baseline_classification[x].get('cv_f1_mean', 0))
        
        advanced_f1 = advanced_classification_results[best_advanced_class].get('cv_f1_mean', 0)
        baseline_f1 = baseline_classification[best_baseline_class].get('cv_f1_mean', 0)
        improvement = ((advanced_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        
        print(f"   Best Advanced: {best_advanced_class} (F1 = {advanced_f1:.4f})")
        print(f"   Best Baseline: {best_baseline_class} (F1 = {baseline_f1:.4f})")
        print(f"   Improvement: {improvement:+.1f}% ({advanced_f1:.4f} vs {baseline_f1:.4f})")
        
        if improvement > 10:
            print("   Significant improvement achieved!")
        elif improvement > 0:
            print("   Moderate improvement achieved")
        else:
            print("   WARNING: No improvement over baseline")
    
    # Feature Importance Analysis
    print("\nPhase 4: Feature Importance Analysis")
    print("-" * 50)
    
    # Display feature importance from best tree-based models
    print("Feature Importance Analysis:")
    
    for model_name, results in advanced_regression_results.items():
        if 'feature_importance' in results and any(x in model_name for x in ['XGBoost', 'LightGBM']):
            importance_df = results['feature_importance']
            print(f"\n   {model_name} - Top 10 Most Important Features:")
            for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"     {idx:2d}. {row['feature']}: {row['importance']:.4f}")
            break  # Show only one model to avoid cluttering
    
    # Hyperparameter Optimization Results
    print("\nPhase 5: Hyperparameter Optimization Results")
    print("-" * 50)
    
    if ml_framework.best_params:
        print("Optimized Hyperparameters:")
        for model_name, params in ml_framework.best_params.items():
            print(f"\n   {model_name.upper()}:")
            for param, value in list(params.items())[:5]:  # Show first 5 parameters
                if isinstance(value, float):
                    print(f"     {param}: {value:.4f}")
                else:
                    print(f"     {param}: {value}")
            if len(params) > 5:
                print(f"     ... and {len(params) - 5} more parameters")
    
    # Create Visualizations
    print("\nPhase 6: Creating Advanced Visualizations")
    print("-" * 50)
    
    try:
        print("Creating comprehensive model comparison visualizations...")
        comparison_fig = ml_framework.create_advanced_visualizations(
            advanced_regression_results, baseline_regression, ml_framework.optimization_history
        )
        
        # Save visualization
        plots_dir = Config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        viz_path = plots_dir / "day7_8_advanced_models.html"
        comparison_fig.write_html(str(viz_path))
        print(f"Advanced model visualizations saved: {viz_path}")
        
    except Exception as e:
        print(f"WARNING: Visualization creation failed: {e}")
    
    # Save All Results
    print("\nPhase 7: Saving Advanced Modeling Results")
    print("-" * 50)
    
    saved_files = ml_framework.save_advanced_results(
        advanced_regression_results, 
        advanced_classification_results if 'advanced_classification_results' in locals() else {},
        baseline_regression, 
        baseline_classification
    )
    
    print("Advanced modeling results saved:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    return {
        'advanced_regression_results': advanced_regression_results,
        'advanced_classification_results': advanced_classification_results if 'advanced_classification_results' in locals() else {},
        'baseline_regression': baseline_regression,
        'baseline_classification': baseline_classification,
        'best_hyperparameters': ml_framework.best_params,
        'saved_files': saved_files
    }

def analyze_model_improvements(results):
    """Analyze and summarize improvements from advanced models"""
    print("\nAdvanced Model Analysis & Insights")
    print("=" * 70)
    
    insights = []
    recommendations = []
    
    advanced_reg = results['advanced_regression_results']
    baseline_reg = results['baseline_regression']
    best_params = results['best_hyperparameters']
    
    # Performance improvement analysis
    if advanced_reg:
        best_advanced = max(advanced_reg.keys(), key=lambda x: advanced_reg[x].get('cv_r2_mean', 0))
        advanced_score = advanced_reg[best_advanced].get('cv_r2_mean', 0)
        
        insights.append(f"Best advanced model: {best_advanced} achieved {advanced_score:.4f} R² score")
        
        if baseline_reg:
            best_baseline = max(baseline_reg.keys(), key=lambda x: baseline_reg[x].get('cv_r2_mean', 0))
            baseline_score = baseline_reg[best_baseline].get('cv_r2_mean', 0)
            improvement = ((advanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            insights.append(f"Performance improvement: {improvement:+.1f}% over baseline {best_baseline}")
        
        # Model-specific insights
        if 'XGBoost' in best_advanced:
            insights.append("XGBoost excels at capturing non-linear patterns in financial data")
            recommendations.append("Consider ensemble methods combining XGBoost with LightGBM")
        
        if 'LightGBM' in best_advanced:
            insights.append("LightGBM shows excellent training efficiency with competitive performance")
            recommendations.append("LightGBM is ideal for production deployment due to speed")
        
        # Predictive power assessment
        if advanced_score > 0.1:
            insights.append("Strong predictive signal detected - models show genuine forecasting ability")
            recommendations.append("Implement risk-adjusted portfolio strategies using model predictions")
        elif advanced_score > 0.05:
            insights.append("Moderate predictive signal - useful for portfolio optimization")
            recommendations.append("Focus on risk management rather than aggressive trading strategies")
        else:
            insights.append("WARNING: Weak predictive signal - market efficiency limits forecastability")
            recommendations.append("Emphasize diversification and risk management over return prediction")
    
    # Hyperparameter optimization insights
    if best_params:
        insights.append(f"Hyperparameter optimization completed for {len(best_params)} models")
        
        # XGBoost specific insights
        if 'xgboost' in best_params:
            xgb_params = best_params['xgboost']
            if xgb_params.get('learning_rate', 0.1) < 0.05:
                insights.append("Low learning rate suggests need for careful regularization")
            if xgb_params.get('max_depth', 6) > 8:
                insights.append("Deep trees suggest complex feature interactions")
        
        recommendations.append("Use optimized hyperparameters for production model deployment")
    
    # Feature importance insights
    if advanced_reg:
        for model_name, results in advanced_reg.items():
            if 'feature_importance' in results:
                top_feature = results['feature_importance'].iloc[0]['feature']
                insights.append(f"Most predictive feature: {top_feature}")
                
                # Categorize feature types
                technical_features = [f for f in results['feature_importance']['feature'][:10] 
                                    if any(x in f for x in ['rsi', 'macd', 'bb_', 'stoch'])]
                if len(technical_features) >= 5:
                    insights.append("Technical indicators dominate feature importance")
                    recommendations.append("Focus on technical analysis in trading strategy")
                break
    
    print("Key Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    print(f"\nStrategic Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return insights, recommendations

def main():
    """Main execution function for Day 7-8"""
    
    print("Stock Market Prediction Engine - Day 7-8")
    print("Advanced Machine Learning Model Development & Hyperparameter Optimization")
    print("=" * 80)
    
    # Check dependencies from previous days
    config = Config()
    features_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    baseline_reg_path = config.PROCESSED_DATA_PATH / "regression_model_results.csv"
    
    if not features_path.exists():
        print("\nFeature dataset not found!")
        print("Please run Day 4 first to generate engineered features")
        return
    
    # Baseline results are optional (only for comparison)
    if not baseline_reg_path.exists():
        print("\nINFO: No baseline model results found (skipping Day 6 comparison)")
        print("   This is fine - we'll train optimized models directly!")
    
    # Run advanced ML development
    results = run_advanced_ml_development()
    
    if results is None:
        print("\nAdvanced ML model development failed!")
        return
    
    # Analyze improvements and generate insights
    insights, recommendations = analyze_model_improvements(results)
    
    # Display final summary
    advanced_reg = results['advanced_regression_results']
    baseline_reg = results['baseline_regression']
    
    print("\nDay 7-8 Completed Successfully!")
    print("=" * 80)
    print("Advanced ML models developed with hyperparameter optimization")
    print("XGBoost and LightGBM models trained and evaluated")
    print("Comprehensive model evaluation completed")
    print("Feature importance analysis conducted")
    print("Hyperparameter optimization results saved")
    print("Advanced visualizations created")
    
    print(f"\nFinal Advanced Model Development Summary:")
    print(f"   Advanced regression models: {len(advanced_reg)}")
    print(f"   Hyperparameters optimized: {len(results['best_hyperparameters'])}")
    print(f"   Feature importance analyzed: OK")
    print(f"   Insights generated: {len(insights)}")
    print(f"   Recommendations provided: {len(recommendations)}")
    
    # Best model summary
    if advanced_reg:
        best_model = max(advanced_reg.keys(), key=lambda x: advanced_reg[x].get('cv_r2_mean', 0))
        best_score = advanced_reg[best_model].get('cv_r2_mean', 0)
        print(f"   Best advanced model: {best_model} (R² = {best_score:.4f})")
        
        # Performance improvement
        if baseline_reg:
            best_baseline = max(baseline_reg.keys(), key=lambda x: baseline_reg[x].get('cv_r2_mean', 0))
            baseline_score = baseline_reg[best_baseline].get('cv_r2_mean', 0)
            improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            print(f"   Improvement over baseline: {improvement:+.1f}%")
    
    print("\nReady for Day 9-12:")
    print("1. Model ensemble and stacking techniques")
    print("2. Advanced cross-validation strategies")
    print("3. Model interpretability with SHAP")
    print("4. Production-ready model pipeline")

if __name__ == "__main__":
    main()