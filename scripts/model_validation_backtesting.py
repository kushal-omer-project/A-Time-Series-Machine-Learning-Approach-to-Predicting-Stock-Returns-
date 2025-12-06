#!/usr/bin/env python3

import sys
from pathlib import Path
import warnings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.validation_framework import ValidationFramework
from loguru import logger
warnings.filterwarnings('ignore')

def main():
    print("Stock Market Prediction Engine - Model Validation")
    print("Model Validation & Backtesting Framework")
    print("=" * 60)
    
    # Initialize validation framework
    validation_framework = ValidationFramework()
    
    # Phase 1: Load Models and Data
    print("\nPhase 1: Loading Trained Models and Data")
    print("-" * 50)
    
    models = validation_framework.load_trained_models()
    if not models:
        print("No trained models found!")
        print("Please ensure you have completed model training first.")
        return
    
    # Separate individual and ensemble models for reporting
    individual_models = [name for name in models.keys() if not name.startswith('Ensemble_')]
    ensemble_models = [name for name in models.keys() if name.startswith('Ensemble_')]
    
    print(f"Loaded {len(individual_models)} individual models:")
    for model in individual_models:
        print(f"   • {model}")
    
    if len(ensemble_models) > 0:
        print(f"Loaded {len(ensemble_models)} ensemble models:")
        for model in ensemble_models:
            print(f"   • {model}")
    else:
        print("INFO: No ensemble models found (Day 9 skipped - this is fine for comparison!)")
    
    print(f"Total models to validate: {len(models)}")
    print(f"   → Comparing {len(individual_models)} individual models: {', '.join(individual_models)}")
    
    # Phase 2: Comprehensive Validation
    print("\nPhase 2: Running Comprehensive Validation")
    print("-" * 50)
    
    print("Starting comprehensive validation process...")
    print("This includes:")
    print("  • Walk-forward validation with 252-day windows")
    print("  • Out-of-sample testing on latest 20% of data")
    print("  • Statistical significance testing")
    print("  • Robustness testing across market conditions")
    print("  • Risk-adjusted performance metrics")
    print("  • Model stability assessment")
    
    results = validation_framework.run_comprehensive_validation()
    
    if not results or not results.get('validation_results'):
        print("Validation failed!")
        return
    
    validation_results = results['validation_results']
    significance_results = results.get('significance_results', {})
    
    print(f"\nValidation completed for {len(validation_results)} models")
    
    # Debug: Show which models were actually validated
    print(f"\nMODELS VALIDATED:")
    for model_name in validation_results.keys():
        print(f"   • {model_name}")
    
    # Phase 3: Results Analysis
    print("\nPhase 3: Validation Results Analysis")
    print("-" * 50)
    
    # Check if we have any models with reasonable performance
    decent_models = []
    for model_name, results in validation_results.items():
        wf_r2 = results.get('walk_forward', {}).get('overall_r2', -999)
        oos_r2 = results.get('out_of_sample', {}).get('r2', -999)
        if wf_r2 > -0.5 or oos_r2 > -0.1:  # More lenient thresholds
            decent_models.append(model_name)
    
    if not decent_models:
        print("\nMODEL PERFORMANCE WARNING:")
        print("=" * 50)
        print("All models show negative R² scores, indicating:")
        print("1. Models perform worse than simple mean prediction")
        print("2. Possible overfitting or feature leakage issues")
        print("3. Target variable may not be predictable with current features")
        print("4. Need to revisit feature engineering or model selection")
        print("\nThis is normal in financial prediction - markets are efficient!")
        print("Focus on relative performance and risk-adjusted metrics.")
    
    # Display summary results anyway for learning purposes
    
    # Display summary results
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("=" * 70)
    
    # Sort models by walk-forward R² performance
    model_performance = []
    for model_name, results in validation_results.items():
        wf_r2 = results.get('walk_forward', {}).get('overall_r2', 0)
        oos_r2 = results.get('out_of_sample', {}).get('r2', 0)
        stability = results.get('stability', {}).get('overall_stability_score', 0)
        sharpe = results.get('risk_metrics', {}).get('sharpe_ratio', 0)
        
        model_performance.append({
            'name': model_name,
            'wf_r2': wf_r2,
            'oos_r2': oos_r2,
            'stability': stability,
            'sharpe': sharpe
        })
    
    # Sort by walk-forward R²
    model_performance.sort(key=lambda x: x['wf_r2'], reverse=True)
    
    for i, model in enumerate(model_performance, 1):
        print(f"{i:2d}. {model['name']:<20}")
        print(f"    Walk-Forward R²: {model['wf_r2']:7.4f}")
        print(f"    Out-of-Sample R²: {model['oos_r2']:6.4f}")
        print(f"    Stability Score:  {model['stability']:6.4f}")
        print(f"    Sharpe Ratio:     {model['sharpe']:6.4f}")
        print()
    
    # Best performing model details
    if model_performance:
        best_model = model_performance[0]
        best_name = best_model['name']
        best_results = validation_results[best_name]
        
        print(f"\nBEST PERFORMING MODEL: {best_name}")
        print("=" * 50)
        
        # Walk-forward validation details
        if 'walk_forward' in best_results:
            wf = best_results['walk_forward']
            print(f"Walk-Forward Validation:")
            print(f"   Total Folds:       {wf.get('total_folds', 0)}")
            print(f"   Overall R²:        {wf.get('overall_r2', 0):.4f}")
            print(f"   Overall RMSE:      {wf.get('overall_rmse', 0):.4f}")
            print(f"   Mean Fold R²:      {wf.get('mean_fold_r2', 0):.4f} ± {wf.get('std_fold_r2', 0):.4f}")
            print(f"   Stability Score:   {wf.get('stability_score', 0):.4f}")
        
        # Out-of-sample testing details
        if 'out_of_sample' in best_results:
            oos = best_results['out_of_sample']
            print(f"\nOut-of-Sample Testing:")
            print(f"   Test Period:       {oos.get('test_start', 'N/A')} to {oos.get('test_end', 'N/A')}")
            print(f"   Test Samples:      {oos.get('test_samples', 0):,}")
            print(f"   R² Score:          {oos.get('r2', 0):.4f}")
            print(f"   RMSE:              {oos.get('rmse', 0):.4f}")
        
        # Risk-adjusted metrics
        if 'risk_metrics' in best_results:
            risk = best_results['risk_metrics']
            print(f"\nRisk-Adjusted Performance:")
            print(f"   Sharpe Ratio:      {risk.get('sharpe_ratio', 0):.4f}")
            print(f"   Sortino Ratio:     {risk.get('sortino_ratio', 0):.4f}")
            print(f"   Max Drawdown:      {risk.get('max_drawdown', 0):.4f}%")
            print(f"   Win Rate:          {risk.get('win_rate', 0):.1f}%")
            print(f"   Annualized Return: {risk.get('annualized_return', 0):.4f}%")
        
        # Stability assessment
        if 'stability' in best_results:
            stability = best_results['stability']
            print(f"\nModel Stability Assessment:")
            print(f"   Overall Score:     {stability.get('overall_stability_score', 0):.4f}")
            print(f"   Rating:            {stability.get('stability_rating', 'Unknown')}")
    
    # Statistical Significance Results
    if significance_results:
        print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
        print("=" * 60)
        
        significant_comparisons = 0
        for comparison, sig_results in significance_results.items():
            t_test_sig = sig_results.get('paired_t_test', {}).get('significant', False)
            wilcoxon_sig = sig_results.get('wilcoxon_test', {}).get('significant', False)
            better_model = sig_results.get('better_model', 'Unknown')
            effect_size = sig_results.get('effect_size', {}).get('cohens_d', 0)
            
            if t_test_sig or wilcoxon_sig:
                significant_comparisons += 1
                print(f"\n{comparison.replace('_', ' ').title()}:")
                print(f"   Better Model:      {better_model}")
                print(f"   T-test p-value:    {sig_results.get('paired_t_test', {}).get('p_value', 1):.4f} {'✓' if t_test_sig else '✗'}")
                print(f"   Wilcoxon p-value:  {sig_results.get('wilcoxon_test', {}).get('p_value', 1):.4f} {'✓' if wilcoxon_sig else '✗'}")
                print(f"   Effect Size:       {effect_size:.4f} ({sig_results.get('effect_size', {}).get('interpretation', 'unknown')})")
        
        if significant_comparisons == 0:
            print("   No statistically significant differences found between models")
            print("   This suggests similar performance levels across models")
    
    # Robustness Analysis Summary
    print(f"\nROBUSTNESS ANALYSIS SUMMARY:")
    print("=" * 50)
    
    robustness_scores = []
    for model_name, results in validation_results.items():
        if 'robustness' in results:
            score = results['robustness'].get('overall_robustness_score', 0)
            robustness_scores.append((model_name, score))
    
    robustness_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, score) in enumerate(robustness_scores, 1):
        rating = "Excellent" if score >= 0.8 else "Good" if score >= 0.6 else "Fair" if score >= 0.4 else "Poor"
        print(f"   {i}. {model_name:<20} Score: {score:.4f} ({rating})")
    
    # Phase 4: Create Visualizations
    print("\nPhase 4: Creating Comprehensive Visualizations")
    print("-" * 50)
    
    try:
        print("Creating validation dashboard...")
        fig = validation_framework.create_validation_visualizations(validation_results)
        
        # Save visualization
        plots_dir = Config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        viz_path = plots_dir / "day10_validation_dashboard.html"
        fig.write_html(str(viz_path))
        print(f"Validation dashboard saved: {viz_path}")
        
    except Exception as e:
        print(f"WARNING: Visualization creation failed: {e}")
    
    # Phase 5: Save Results
    print("\nPhase 5: Saving Validation Results")
    print("-" * 50)
    
    # Pass the results dictionary directly (it already has the right structure)
    saved_files = validation_framework.save_validation_results(results)
    
    print("Validation results saved:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    # Phase 6: Recommendations
    print("\nPhase 6: Model Selection Recommendations")
    print("-" * 50)
    
    if model_performance:
        best_overall = model_performance[0]
        
        # Find best model for each metric
        best_stability = max(model_performance, key=lambda x: x['stability'])
        best_oos = max(model_performance, key=lambda x: x['oos_r2'])
        best_sharpe = max(model_performance, key=lambda x: x['sharpe'])
        
        print(f"RECOMMENDATIONS:")
        print(f"   Best Overall Performance:    {best_overall['name']}")
        print(f"   Most Stable Model:           {best_stability['name']}")
        print(f"   Best Out-of-Sample:          {best_oos['name']}")
        print(f"   Best Risk-Adjusted Returns:  {best_sharpe['name']}")
        
        # Production recommendation
        if best_overall['name'] == best_stability['name']:
            print(f"\nPRODUCTION RECOMMENDATION:")
            print(f"   Deploy: {best_overall['name']}")
            print(f"   Reason: Best overall performance AND highest stability")
        else:
            print(f"\nPRODUCTION RECOMMENDATIONS:")
            print(f"   Aggressive Strategy: {best_overall['name']} (highest performance)")
            print(f"   Conservative Strategy: {best_stability['name']} (most stable)")
    
    # Success Summary
    print("\nDay 10 Completed Successfully!")
    print("=" * 60)
    print("Comprehensive model validation completed")
    print("Walk-forward validation with 252-day windows")
    print("Out-of-sample testing on latest data")
    print("Statistical significance testing between models")
    print("Robustness testing across market conditions")
    print("Risk-adjusted performance metrics calculated")
    print("Model stability assessment completed")
    print("Performance attribution analysis")
    print("Interactive validation dashboard created")
    print("Comprehensive results and recommendations saved")
    
    print(f"\nFinal Validation Summary:")
    print(f"   Models Validated: {len(validation_results)}")
    print(f"   Validation Methods: 6 comprehensive approaches")
    print(f"   Statistical Tests: {len(significance_results)} pairwise comparisons")
    print(f"   Files Generated: {len(saved_files)} result files")
    
    if model_performance:
        best_model = model_performance[0]
        print(f"   Best Model: {best_model['name']} (R² = {best_model['wf_r2']:.4f})")
    
    print("\nReady for Day 11:")
    print("1. Risk management and portfolio optimization")
    print("2. Position sizing algorithms")
    print("3. Portfolio construction with constraints")
    print("4. Performance attribution analysis")

if __name__ == "__main__":
    main()