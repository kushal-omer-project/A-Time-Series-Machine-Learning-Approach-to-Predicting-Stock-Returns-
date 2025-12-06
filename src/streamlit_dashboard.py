#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.realtime_prediction import RealTimePredictionEngine
from src.risk_management import RiskManagementFramework

# Page config
st.set_page_config(
    page_title="Stock Market AI Prediction Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.config = Config()
        self.prediction_engine = None
        self.risk_framework = None
        # Don't initialize components in __init__ - do it lazily when needed
    
    def _initialize_components(self):
        """Initialize ML components lazily"""
        try:
            if 'prediction_engine' not in st.session_state:
                print("Initializing RealTimePredictionEngine...", file=sys.stderr)
                try:
                    self.prediction_engine = RealTimePredictionEngine()
                    print("Loading production models...", file=sys.stderr)
                    success = self.prediction_engine.load_production_models()
                    print(f"Models loaded: {success}", file=sys.stderr)
                    st.session_state.prediction_engine = self.prediction_engine
                    st.session_state.models_loaded = success
                    if not success:
                        print("WARNING: Models failed to load, but continuing...", file=sys.stderr)
                except Exception as e:
                    print(f"ERROR loading models: {e}", file=sys.stderr)
                    self.prediction_engine = None
                    st.session_state.models_loaded = False
            else:
                self.prediction_engine = st.session_state.prediction_engine
            
            if 'risk_framework' not in st.session_state:
                print("Initializing RiskManagementFramework...", file=sys.stderr)
                try:
                    self.risk_framework = RiskManagementFramework()
                    st.session_state.risk_framework = self.risk_framework
                except Exception as e:
                    print(f"WARNING: Risk framework failed to initialize: {e}", file=sys.stderr)
                    self.risk_framework = None
            else:
                self.risk_framework = st.session_state.risk_framework
        except Exception as e:
            print(f"ERROR in _initialize_components: {e}", file=sys.stderr)
            self.prediction_engine = None
            self.risk_framework = None
    
    def load_performance_data(self):
        """Load model performance data"""
        try:
            risk_summary_path = self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
            if risk_summary_path.exists():
                return pd.read_csv(risk_summary_path)
            else:
                # Create dummy data for demo
                return pd.DataFrame({
                    'Model': ['XGBoost', 'LightGBM'],
                    'Sharpe_Ratio': [4.25, 3.82, 3.76, 3.45],
                    'Annual_Return': [0.15, 0.12, 0.11, 0.10],
                    'Max_Drawdown': [-0.08, -0.12, -0.10, -0.15],
                    'Win_Rate': [65.2, 62.1, 61.8, 59.5]
                })
        except Exception as e:
            st.error(f"Error loading performance data: {e}")
            return pd.DataFrame()
    
    def get_target_stocks(self):
        """Get target stocks"""
        try:
            stocks_path = self.config.PROCESSED_DATA_PATH / "target_stocks.txt"
            if stocks_path.exists():
                with open(stocks_path, 'r') as f:
                    return [line.strip() for line in f.readlines()][:10]
            else:
                return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD', 'GOOGL', 'TSLA', 'META', 'NFLX', 'CRM']
        except:
            return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD']
    
    def render_header(self):
        """Render main header"""
        st.markdown('<h1 class="main-header">Stock Market AI Prediction Engine</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            models_loaded = st.session_state.get('models_loaded', False)
            status = "Online" if models_loaded else "Offline"
            st.markdown(f"**System Status:** {status}")
            
        with col2:
            try:
                model_count = len(self.prediction_engine.models) if (self.prediction_engine and hasattr(self.prediction_engine, 'models')) else 0
            except:
                model_count = 0
            st.markdown(f"**Models Loaded:** {model_count}")
            
        with col3:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
            
        with col4:
            if st.button("Refresh Data"):
                st.rerun()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Performance Analytics", "Model Insights"]
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("### Quick Stats")
        performance_data = self.load_performance_data()
        if not performance_data.empty:
            xgboost_data = performance_data[performance_data['Model'] == 'XGBoost']
            lightgbm_data = performance_data[performance_data['Model'] == 'LightGBM']
            
            if not xgboost_data.empty:
                st.sidebar.markdown("**XGBoost**")
                st.sidebar.metric("Sharpe Ratio", f"{xgboost_data.iloc[0]['Sharpe_Ratio']:.3f}")
                st.sidebar.metric("Annual Return", f"{xgboost_data.iloc[0]['Annual_Return']/10:.1f}%")
            
            if not lightgbm_data.empty:
                st.sidebar.markdown("**LightGBM**")
                st.sidebar.metric("Sharpe Ratio", f"{lightgbm_data.iloc[0]['Sharpe_Ratio']:.3f}")
                st.sidebar.metric("Annual Return", f"{lightgbm_data.iloc[0]['Annual_Return']/10:.1f}%")

        
        return page
    
    def render_overview_page(self):
        """Render overview page"""
        st.header("System Overview")
        
        performance_data = self.load_performance_data()
        
        # Model descriptions
        st.subheader("Trading Models")
        col1, col2 = st.columns(2)
        
        if not performance_data.empty:
            xgboost_data = performance_data[performance_data['Model'] == 'XGBoost']
            lightgbm_data = performance_data[performance_data['Model'] == 'LightGBM']
            
            with col1:
                st.markdown("### XGBoost")
                st.markdown("**Purpose:** Used to trade off with risk")
                st.markdown("*Risk-averse strategy for stable returns*")
                if not xgboost_data.empty:
                    xgb = xgboost_data.iloc[0]
                    st.metric("Sharpe Ratio", f"{xgb['Sharpe_Ratio']:.3f}")
                    st.metric("Annual Return", f"{xgb['Annual_Return']/10:.1f}%")
                    st.metric("Max Drawdown", f"{xgb['Max_Drawdown_Percent']:.1f}%")
                    st.metric("Win Rate", f"{xgb['Win_Rate']:.1f}%")
            
            with col2:
                st.markdown("### LightGBM")
                st.markdown("**Purpose:** Used to trade with risk")
                st.markdown("*Risk-taking strategy for higher returns*")
                if not lightgbm_data.empty:
                    lgb = lightgbm_data.iloc[0]
                    st.metric("Sharpe Ratio", f"{lgb['Sharpe_Ratio']:.3f}")
                    st.metric("Annual Return", f"{lgb['Annual_Return']/10:.1f}%")
                    st.metric("Max Drawdown", f"{lgb['Max_Drawdown_Percent']:.1f}%")
                    st.metric("Win Rate", f"{lgb['Win_Rate']:.1f}%")
        
        # Model performance chart
        st.subheader("Model Performance Metrics")
        if not performance_data.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=performance_data['Model'],
                y=performance_data['Sharpe_Ratio'],
                name='Sharpe Ratio',
                marker_color='rgb(55, 126, 184)',
                text=[f"{val:.3f}" for val in performance_data['Sharpe_Ratio']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Sharpe Ratio Comparison",
                xaxis_title="Model",
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Status")
            now = datetime.now()
            market_open = 9 <= now.hour <= 16 and now.weekday() < 5
            status = "Open" if market_open else "Closed"
            st.markdown(f"**Market Status:** {status}")
            st.markdown(f"**Current Time:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.subheader("System Status")
            st.markdown("**Models:** XGBoost & LightGBM")
            st.markdown("**Features:** 73 engineered features")
            st.markdown("**Status:** Production Ready")
    
    async def run_predictions(self, selected_stocks):
        """Run predictions for selected stocks"""
        if not self.prediction_engine or not st.session_state.get('models_loaded', False):
            st.error("Prediction engine not available")
            return {}
        
        try:
            # Override target stocks temporarily
            original_method = self.prediction_engine.get_target_stocks
            self.prediction_engine.get_target_stocks = lambda: selected_stocks
            
            # Run prediction cycle
            results = await self.prediction_engine.run_realtime_cycle()
            
            # Restore original method
            self.prediction_engine.get_target_stocks = original_method
            
            return results
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return {}
    
    def render_predictions_page(self, selected_stocks):
        """Render live predictions page"""
        st.header("Live Predictions")
        
        # Control panel
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Analyzing:** {', '.join(selected_stocks)}")
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)")
        
        with col3:
            if st.button("Generate Predictions"):
                st.session_state.trigger_prediction = True
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Generate predictions
        if st.session_state.get('trigger_prediction', False) or auto_refresh:
            with st.spinner("Generating AI predictions..."):
                # Simulate predictions for demo (replace with real predictions)
                predictions = {}
                for symbol in selected_stocks:
                    pred_value = np.random.normal(0, 0.01)  # Random prediction for demo
                    confidence = "high" if abs(pred_value) > 0.005 else "medium"
                    direction = "BUY" if pred_value > 0.001 else "SELL" if pred_value < -0.001 else "HOLD"
                    
                    predictions[symbol] = {
                        'primary': {
                            'prediction': pred_value,
                            'confidence': confidence
                        },
                        'direction': direction,
                        'timestamp': datetime.now().isoformat()
                    }
                
                st.session_state.predictions = predictions
                st.session_state.trigger_prediction = False
        
        # Display predictions
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            
            # Prediction cards
            cols = st.columns(min(len(predictions), 3))
            for i, (symbol, pred_data) in enumerate(predictions.items()):
                with cols[i % 3]:
                    pred_value = pred_data['primary']['prediction']
                    direction = pred_data['direction']
                    confidence = pred_data['primary']['confidence']
                    
                    card_class = "prediction-positive" if pred_value > 0 else "prediction-negative"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>{symbol}</h3>
                        <h2>{direction}</h2>
                        <p>Prediction: {pred_value:+.4f}</p>
                        <p>Confidence: {confidence.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed prediction table
            st.subheader("Detailed Predictions")
            pred_df = pd.DataFrame([
                {
                    'Symbol': symbol,
                    'Prediction': data['primary']['prediction'],
                    'Direction': data['direction'],
                    'Confidence': data['primary']['confidence'],
                    'Timestamp': data['timestamp']
                }
                for symbol, data in predictions.items()
            ])
            
            st.dataframe(pred_df, use_container_width=True)
            
            # Prediction visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(predictions.keys()),
                y=[data['primary']['prediction'] for data in predictions.values()],
                marker_color=['green' if p > 0 else 'red' for p in [data['primary']['prediction'] for data in predictions.values()]],
                name='Predictions'
            ))
            
            fig.update_layout(
                title="Stock Predictions Comparison",
                xaxis_title="Stock Symbol",
                yaxis_title="Prediction Value",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_page(self):
        """Render performance analytics page"""
        st.header("Performance Analytics")
        
        performance_data = self.load_performance_data()
        if performance_data.empty:
            st.warning("No performance data available")
            return
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Rankings")
            
            # Sort by Sharpe ratio
            sorted_data = performance_data.sort_values('Sharpe_Ratio', ascending=False)
            
            for i, (_, row) in enumerate(sorted_data.iterrows(), 1):
                st.markdown(f"""
                **{i}. {row['Model']}**
                - Sharpe Ratio: {row['Sharpe_Ratio']:.3f}
                - Annual Return: {row['Annual_Return']/10:.1f}%
                - Win Rate: {row['Win_Rate']:.1f}%
                """)
        
        with col2:
            st.subheader("Risk-Return Analysis")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                #x=performance_data['Max_Drawdown'].abs() * 100,
                x=performance_data['Max_Drawdown_Percent'].abs(),
                #y=performance_data['Annual_Return'] * 100,
                y=performance_data['Annual_Return'] / 10,

                mode='markers+text',
                text=performance_data['Model'],
                textposition="top center",
                marker=dict(
                    size=performance_data['Sharpe_Ratio'] * 10,
                    color=performance_data['Sharpe_Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Models'
            ))
            
            fig.update_layout(
                title="Risk vs Return (Bubble size = Sharpe Ratio)",
                xaxis_title="Max Drawdown (%)",
                yaxis_title="Annual Return (%)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_insights_page(self):
        """Render model insights page"""
        st.header("Model Insights")
        
        # Model explanation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("How Our AI Works")
            
            st.markdown("""
            **Our system uses two powerful gradient boosting models:**
            
            1. **XGBoost**: Gradient boosting for complex patterns
            2. **LightGBM**: Fast, efficient tree-based learning
            
            **The system uses 73 engineered features including:**
            - Technical indicators (RSI, MACD, Bollinger Bands)
            - Price momentum and volatility measures
            - Rolling statistics and lag features
            - Time-based patterns
            """)
        
        with col2:
            st.subheader("Feature Importance")
            
            # Simulate feature importance
            features = ['bb_lower', 'ema_200', 'close_to_sma20', 'rsi', 'momentum_5d', 
                       'atr_ratio', 'volume_ratio', 'volatility_20d', 'sma_50', 'stoch_k']
            importance = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="Importance Score",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model architecture
        st.subheader("Architecture Overview")
        
        st.markdown("""
        ```
        Data Input (OHLCV + Features)
              ↓
        Feature Engineering (73 features)
              ↓
        ML Models
              ├── XGBoost (Optimized)
              └── LightGBM (Optimized)
              ↓
        Prediction Aggregation
              ↓
        Final Prediction + Confidence
        ```
        """)
        
        # Performance metrics explanation
        st.subheader("Understanding the Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Sharpe Ratio**: Risk-adjusted returns
            - Higher is better
            - XGBoost: 1.609 (Good)
            - LightGBM: 1.361 (Good)
            - Market average: ~1.0
            
            **Annual Return**: Yearly profit percentage
            - XGBoost: 14.4%
            - LightGBM: 12.2%
            - S&P 500 average: ~10%
            
            **Max Drawdown**: Largest loss from peak
            - Lower is better
            - XGBoost: -4.3% (Low risk)
            - LightGBM: -25.9% (Higher risk)
            - Acceptable: <20%
            """)
        
        with col2:
            st.markdown("""
            **Win Rate**: Percentage of profitable trades
            - XGBoost: 54.8%
            - LightGBM: 54.5%
            - Random chance: 50%
            - Good performance: >55%
            
            **Prediction Speed**: Time to generate forecast
            - Real-time: <3 seconds
            - Batch processing: <30 seconds
            - Updated every 15 minutes
            """)
        
        # Model comparison
        st.subheader("Model Comparison")
        
        performance_data = self.load_performance_data()
        if not performance_data.empty:
            # Radar chart for model comparison
            categories = ['Sharpe Ratio', 'Annual Return', 'Win Rate', 'Stability']
            
            fig = go.Figure()
            
            for _, model in performance_data.head(3).iterrows():
                # Normalize metrics for radar chart
                sharpe_norm = min(model['Sharpe_Ratio'] / 5, 1)  # Normalize to 0-1
                #return_norm = model['Annual_Return'] * 5  # Scale up return
                return_norm = (model['Annual_Return'] / 10) / 100  # Convert to 0-1 range for radar

                win_norm = model['Win_Rate'] / 100  # Convert percentage
                stability_norm = 0.8 + np.random.normal(0, 0.1)  # Simulate stability
                
                values = [sharpe_norm, return_norm, win_norm, stability_norm]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main dashboard runner"""
        # Initialize components only when actually running
        try:
            if self.prediction_engine is None:
                self._initialize_components()
        except Exception as e:
            print(f"Warning: Component initialization failed: {e}", file=sys.stderr)
        
        self.render_header()
        
        # Sidebar navigation
        page = self.render_sidebar()
        
        # Main content based on selected page
        if page == "Overview":
            self.render_overview_page()
        elif page == "Performance Analytics":
            self.render_performance_page()
        elif page == "Model Insights":
            self.render_model_insights_page()
        
        # Footer
        st.markdown("---")
        st.markdown("### Stock Market AI Prediction Engine")
        st.markdown("*Built with Streamlit, Plotly, and Advanced ML Models*")

def main():
    """Main entry point"""
    try:
        print("Starting Streamlit dashboard...", file=sys.stderr)
        dashboard = DashboardApp()
        print("Dashboard created, running...", file=sys.stderr)
        dashboard.run()
    except KeyboardInterrupt:
        print("Dashboard stopped by user", file=sys.stderr)
    except Exception as e:
        import traceback
        error_msg = f"Dashboard Error: {e}\n\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        try:
            st.error(f"Dashboard Error: {e}")
            st.markdown("**Troubleshooting:**")
            st.markdown("1. Ensure all model files are in place")
            st.markdown("2. Check that risk analysis completed")
            st.markdown("3. Verify data files exist in data/processed/")
            st.code(traceback.format_exc())
        except:
            print(f"Fatal error: {error_msg}", file=sys.stderr)
            # Show a simple error page
            st.title("Application Error")
            st.error("The dashboard encountered an error. Please check the logs.")

if __name__ == "__main__":
    main()