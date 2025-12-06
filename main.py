#!/usr/bin/env python3
"""
Stock Market Prediction Engine
System Integration & Testing
"""

import asyncio
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config import Config
from src.integration_testing import SystemIntegrationTester, SystemOptimizer

def display_banner():
    print("Stock Market Prediction Engine")
    print("System Integration & Testing")
    print("=" * 50)

def display_test_summary(report: dict):
    summary = report['test_summary']
    print(f"\nTest Execution Summary:")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Execution Time: {summary['total_execution_time']:.2f}s")
    print(f"System Status: {report['system_status']}")

def display_detailed_results(report: dict):
    print("\nDetailed Test Results:")
    print("-" * 30)
    for test_name, result in report['test_results'].items():
        status_icon = "[PASS]" if result['status'] == "PASS" else "[FAIL]" if result['status'] == "FAIL" else "[WARN]"
        print(f"{status_icon} {test_name}")
        if result['details']:
            print(f"   {result['details']}")
        if result['execution_time'] > 0:
            print(f"   Time: {result['execution_time']:.2f}s")

def display_performance_metrics(report: dict):
    if report['performance_metrics']:
        print("\nPerformance Metrics:")
        print("-" * 20)
        for metric, value in report['performance_metrics'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}s")
            else:
                print(f"{metric}: {value}")

def display_recommendations(report: dict):
    if report['recommendations']:
        print("\nSystem Recommendations:")
        print("-" * 25)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

def save_test_report(report: dict) -> str:
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = config.PROCESSED_DATA_PATH / f"integration_test_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    return str(report_path)

async def run_integration_tests():
    print("Initializing System Integration Tester...")
    tester = SystemIntegrationTester()
    print("Running comprehensive integration tests...")
    report = await tester.run_comprehensive_tests()
    
    display_test_summary(report)
    display_detailed_results(report)
    display_performance_metrics(report)
    display_recommendations(report)
    
    report_path = save_test_report(report)
    print(f"\nTest report saved: {report_path}")
    return report

def run_system_optimization():
    print("\nInitializing System Optimizer...")
    optimizer = SystemOptimizer()
    
    print("Optimizing prediction pipeline...")
    pipeline_results = optimizer.optimize_prediction_pipeline()
    
    if pipeline_results.get('optimizations_applied'):
        print("Pipeline Optimizations:")
        for opt in pipeline_results['optimizations_applied']:
            print(f"{opt}")
    
    if pipeline_results.get('performance_improvements'):
        print("Performance Improvements:")
        for improvement, value in pipeline_results['performance_improvements'].items():
            print(f"{improvement}: {value}")
    
    print("\nCleaning up system files...")
    cleanup_results = optimizer.cleanup_system_files()
    
    if cleanup_results.get('files_cleaned', 0) > 0:
        print(f"Cleaned {cleanup_results['files_cleaned']} files")
        print(f"Freed {cleanup_results['space_freed_mb']:.2f} MB")
    
    return {
        'pipeline_optimization': pipeline_results,
        'cleanup_results': cleanup_results
    }

def validate_system_health():
    print("\nValidating system health...")
    config = Config()
    health_checks = {
        'configuration': False,
        'data_files': False,
        'model_files': False,
        'directory_structure': False
    }
    
    try:
        config.create_directories()
        health_checks['configuration'] = True
        print("Configuration system: OK")
    except:
        print("Configuration system: FAILED")
    
    essential_files = [
        config.FEATURES_DATA_PATH / "selected_features.csv",
        config.PROCESSED_DATA_PATH / "target_stocks.txt"
    ]
    
    files_exist = sum(1 for f in essential_files if f.exists())
    if files_exist >= len(essential_files) * 0.5:
        health_checks['data_files'] = True
        print(f"Data files: OK ({files_exist}/{len(essential_files)})")
    else:
        print(f"Data files: WARNING ({files_exist}/{len(essential_files)})")
    
    models_dir = config.PROJECT_ROOT / "models"
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.joblib"))
        if len(model_files) >= 3:
            health_checks['model_files'] = True
            print(f"Model files: OK ({len(model_files)} found)")
        else:
            print(f"Model files: WARNING ({len(model_files)} found)")
    else:
        print("Model directory: MISSING")
    
    required_dirs = [config.DATA_PATH, config.LOGS_PATH, config.PROJECT_ROOT / "src"]
    dirs_exist = sum(1 for d in required_dirs if d.exists())
    if dirs_exist == len(required_dirs):
        health_checks['directory_structure'] = True
        print("Directory structure: OK")
    else:
        print(f"Directory structure: WARNING ({dirs_exist}/{len(required_dirs)})")
    
    health_score = sum(health_checks.values()) / len(health_checks) * 100
    status = "HEALTHY" if health_score >= 75 else "DEGRADED" if health_score >= 50 else "CRITICAL"
    print(f"\nSystem Health Score: {health_score:.0f}% ({status})")
    return health_checks, health_score

def generate_final_report(test_report: dict, optimization_results: dict, health_checks: dict):
    config = Config()
    final_report = {
        'summary': {
            'completion_date': datetime.now().isoformat(),
            'test_success_rate': test_report['test_summary']['success_rate'],
            'system_status': test_report['system_status'],
            'health_score': sum(health_checks.values()) / len(health_checks) * 100
        },
        'integration_testing': test_report,
        'system_optimization': optimization_results,
        'health_validation': health_checks
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = config.PROCESSED_DATA_PATH / f"final_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    return final_report, str(report_path)

async def main():
    display_banner()
    
    try:
        print("\nPhase 1: System Health Validation")
        print("-" * 40)
        health_checks, health_score = validate_system_health()
        
        print("\nPhase 2: Comprehensive Integration Testing")
        print("-" * 45)
        test_report = await run_integration_tests()
        
        print("\nPhase 3: System Optimization")
        print("-" * 35)
        optimization_results = run_system_optimization()
        
        print("\nPhase 4: Final Report Generation")
        print("-" * 35)
        final_report, report_path = generate_final_report(
            test_report, optimization_results, health_checks
        )
        
        print("\nIntegration & Testing Complete!")
        print("=" * 45)
        
        summary = final_report['summary']
        print(f"Test Success Rate: {summary['test_success_rate']:.1f}%")
        print(f"System Health: {summary['health_score']:.0f}%")
        print(f"Overall Status: {summary['system_status']}")
        print(f"\nFinal Report: {report_path}")
        
        if test_report.get('recommendations'):
            print("\nRecommendations:")
            for i, rec in enumerate(test_report['recommendations'][:3], 1):
                print(f"{i}. {rec}")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Execution error: {e}")

if __name__ == "__main__":
    asyncio.run(main())