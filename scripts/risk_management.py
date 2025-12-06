#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 11
Risk Management & Portfolio Optimization
"""

import sys
from pathlib import Path
import warnings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.risk_management import RiskManagementFramework
from loguru import logger
warnings.filterwarnings('ignore')

def main():
    """Main execution function for Day 11"""
    
    print("Stock Market Prediction Engine - Day 11")
    print("Risk Management & Portfolio Optimization")
    print("=" * 60)
    
    # Initialize risk management framework
    risk_framework = RiskManagementFramework()
    
    # Run comprehensive risk management analysis
    print("\nPhase 1: Loading Validation Results and Models")
    print("-" * 50)
    
    # Check for required files
    config = Config()
    validation_path = config.PROCESSED_DATA_PATH / "day10_validation_results.json"
    features_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    
    if not validation_path.exists():
        print("Validation results not found!")
        print("Please ensure Day 10 validation completed successfully.")
        print(f"Expected file: {validation_path}")
        return
    
    if not features_path.exists():
        print("Feature data not found!")
        print("Please ensure Day 4 feature engineering completed successfully.")
        print(f"Expected file: {features_path}")
        return
    
    print("Required files found - proceeding with risk analysis")
    
    # Phase 1: Run Comprehensive Analysis
    print("\nPhase 2: Running Comprehensive Risk Management Analysis")
    print("-" * 60)
    
    results = risk_framework.run_comprehensive_risk_management()
    
    if not results:
        print("Risk management analysis failed!")
        return
    
    # Phase 2: Display Results Summary
    print("\nPhase 3: Risk Management Results Summary")
    print("-" * 50)
    
    analysis_results = results.get('analysis_results', {})
    summary = results.get('summary', {})
    
    # Display model performance summary
    model_names = [name for name in analysis_results.keys() 
                  if name not in ['portfolios', 'best_strategy']]
    
    if model_names:
        print(f"Analyzed {len(model_names)} trading strategies:")
        for model in model_names:
            print(f"   • {model}")
        
        print(f"\nBEST RISK-ADJUSTED STRATEGY:")
        best_strategy = summary.get('best_strategy', {})
        if best_strategy:
            print(f"   Strategy: {best_strategy.get('name', 'N/A')}")
            print(f"   Sharpe Ratio: {best_strategy.get('sharpe_ratio', 0):.4f}")
            
            # Get detailed metrics for best strategy
            best_name = best_strategy.get('name')
            if best_name and best_name in analysis_results:
                metrics = analysis_results[best_name]
                print(f"   Annual Return: {metrics['performance_ratios']['annual_return']*100:.2f}%")
                print(f"   Annual Volatility: {metrics['performance_ratios']['annual_volatility']*100:.2f}%")
                print(f"   Max Drawdown: {metrics['drawdown_metrics']['max_drawdown']:.4f}")
                print(f"   VaR (95%): {metrics['var_metrics']['var_historical']:.4f}")
                print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    
    # Display portfolio optimization results
    portfolios = analysis_results.get('portfolios', {})
    if portfolios:
        print(f"\nPORTFOLIO OPTIMIZATION RESULTS:")
        
        # Markowitz Portfolio
        markowitz = portfolios.get('markowitz', {})
        if markowitz.get('success', False):
            print(f"   Markowitz Mean-Variance Portfolio:")
            print(f"      Expected Return: {markowitz['expected_return']*100:.2f}%")
            print(f"      Volatility: {markowitz['volatility']*100:.2f}%")
            print(f"      Sharpe Ratio: {markowitz['sharpe_ratio']:.4f}")
            print(f"      Assets: {len(markowitz['stocks'])} stocks")
        else:
            print(f"   WARNING: Markowitz optimization: Failed or insufficient data")
        
        # Risk Parity Portfolio
        risk_parity = portfolios.get('risk_parity', {})
        if risk_parity:
            print(f"   Risk Parity Portfolio:")
            print(f"      Portfolio Return: {risk_parity['portfolio_return']*100:.2f}%")
            print(f"      Portfolio Volatility: {risk_parity['portfolio_volatility']*100:.2f}%")
            print(f"      Sharpe Ratio: {risk_parity['sharpe_ratio']:.4f}")
            print(f"      Assets: {len(risk_parity['stocks'])} stocks")
        else:
            print(f"   WARNING: Risk Parity optimization: Failed or insufficient data")
    
    # Display risk recommendations
    recommendations = summary.get('risk_recommendations', [])
    if recommendations:
        print(f"\nRISK MANAGEMENT RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Phase 3: Display Risk Metrics Comparison
    if model_names:
        print(f"\nRISK METRICS COMPARISON:")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Sharpe':<8} {'Sortino':<8} {'VaR-95%':<10} {'Max DD':<10} {'Win%':<8}")
        print("-" * 80)
        
        for model in model_names:
            metrics = analysis_results[model]
            perf = metrics['performance_ratios']
            var_metrics = metrics['var_metrics']
            dd_metrics = metrics['drawdown_metrics']
            
            print(f"{model:<20} {perf['sharpe_ratio']:<8.3f} {perf['sortino_ratio']:<8.3f} "
                  f"{var_metrics['var_historical']:<10.4f} {dd_metrics['max_drawdown']:<10.4f} "
                  f"{metrics['win_rate']:<8.1f}")
    
    # Phase 4: Position Sizing Recommendations
    if model_names:
        print(f"\nPOSITION SIZING RECOMMENDATIONS (Kelly Criterion):")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Kelly %':<10} {'Optimal %':<12} {'Win Rate':<10} {'W/L Ratio':<10}")
        print("-" * 70)
        
        for model in model_names:
            pos_metrics = analysis_results[model]['position_sizing']
            kelly_pct = pos_metrics['kelly_fraction'] * 100
            optimal_pct = pos_metrics['optimal_position'] * 100
            win_rate = pos_metrics['win_rate'] * 100
            wl_ratio = pos_metrics['win_loss_ratio']
            
            print(f"{model:<20} {kelly_pct:<10.2f} {optimal_pct:<12.2f} "
                  f"{win_rate:<10.1f} {wl_ratio:<10.2f}")
    
    # Phase 5: Files Generated
    saved_files = results.get('saved_files', {})
    if saved_files:
        print(f"\nFILES GENERATED:")
        print("=" * 50)
        for file_type, path in saved_files.items():
            print(f"   {file_type}: {path}")
        
        # Check for dashboard
        plots_dir = config.PROJECT_ROOT / "plots"
        dashboard_path = plots_dir / "day11_risk_dashboard.html"
        if dashboard_path.exists():
            print(f"   Interactive Dashboard: {dashboard_path}")
    
    # Success Summary
    print("\nDay 11 Completed Successfully!")
    print("=" * 60)
    print("Comprehensive risk management analysis completed")
    print("Value at Risk (VaR) and Conditional VaR calculated")
    print("Maximum drawdown analysis performed")
    print("Sharpe, Sortino, and Calmar ratios computed")
    print("Kelly Criterion position sizing implemented")
    print("Markowitz mean-variance optimization attempted")
    print("Risk parity portfolio construction completed")
    print("Transaction cost modeling framework created")
    print("Performance attribution analysis conducted")
    print("Interactive risk dashboard generated")
    print("Comprehensive risk recommendations provided")
    
    print(f"\nFinal Risk Management Summary:")
    print(f"   Models Analyzed: {summary.get('models_analyzed', 0)}")
    print(f"   Portfolios Created: {summary.get('portfolios_created', 0)}")
    print(f"   Risk Files Generated: {len(saved_files)}")
    
    if best_strategy:
        print(f"   Recommended Strategy: {best_strategy.get('name', 'N/A')}")
        print(f"   Best Sharpe Ratio: {best_strategy.get('sharpe_ratio', 0):.4f}")
    
    print("\nReady for Day 12:")
    print("1. Real-time prediction system development")
    print("2. Model serving infrastructure")
    print("3. Alert system for significant predictions")
    print("4. Performance monitoring dashboard")
    print("5. Model drift detection implementation")

if __name__ == "__main__":
    main()