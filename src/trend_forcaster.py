import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ForecastFutureMarketTrends:
    def __init__(self, historical_data, forecasted_data, column='Adj Close'):
        self.historical_data = historical_data
        self.forecasted_data = forecasted_data
        self.column = column
        self.predictions = {}
        self.load_data()

    def load_data(self):
        try:
            logging.info('Loading forecasted data')
            if isinstance(self.forecasted_data, str):
                if not self.forecasted_data:
                    raise ValueError("Forecasted data must be provided")
                
                self.forecasted_data = pd.read_csv(self.forecasted_data, index_col=0, parse_dates=True)
                self.forecasted_data.index.name = 'Date'
                logging.info('Forecasted data loaded successfully')
            elif isinstance(self.forecasted_data, pd.DataFrame):
                logging.info('Forecasted data provided as DataFrame')
            else:
                raise ValueError("Forecasted data must be a file path or a DataFrame")
            
            if self.column not in self.historical_data.columns:
                raise ValueError(f"Column '{self.column}' not found in historical data")

            if 'Forecast' not in self.forecasted_data.columns:
                raise ValueError("Forecast CSV must have a 'Forecast' column.")
            
            self.predictions['Forecast'] = self.forecasted_data['Forecast'].values
            self.forecasted_dates = self.forecasted_data.index
            logging.info('Forecasted values extracted successfully')

        except Exception as e:
            logging.error(f'Error loading forecasted data: {e}')

    def plot_forecast(self):
        try: 
            forecasted_dates = self.forecasted_dates
            plt.figure(figsize=(15, 8))
            plt.plot(self.historical_data.index, self.historical_data, label='Actual', color='blue', linewidth=2)

            Forecast = self.predictions['Forecast']
            plt.plot(forecasted_dates, Forecast, label='Forecast', linestyle='--', color='red')
            if 'conf_lower' in self.forecasted_data.columns and 'conf_upper' in self.forecasted_data.columns:
                plt.fill_between(forecasted_dates, self.forecasted_data['conf_lower'], self.forecasted_data['conf_upper'], color='red', alpha=0.25, label='95% Confidence Interval')

            plt.xticks(rotation=45)
            plt.title("Historical vs. Forecast Data with Confidence Intervals", fontsize=16)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel(self.column, fontsize=14)
            plt.legend(loc='best')
            sns.set(style="whitegrid")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error in plotting forecasts: {e}")

    def analyze_forecast(self, threshold=0.05):
        analysis_results = {}
        logging.info("Starting forecast analysis.")
        
        for model_name, forecast in self.predictions.items():
            trend = "upward" if np.mean(np.diff(forecast)) > 0 else "downward"
            trend_magnitude = np.max(np.diff(forecast))
            logging.info(f"{model_name} forecast shows a {trend} trend.")

            volatility = np.std(forecast)
            volatility_level = "High" if volatility > threshold else "Low"
            max_price = np.max(forecast)
            min_price = np.min(forecast)
            price_range = max_price - min_price
            volatility_analysis = self._volatility_risk_analysis(forecast, threshold)
            opportunities_risks = self._market_opportunities_risks(trend, volatility_level)
            
            analysis_results[model_name] = {
                'Trend': trend,
                'Trend_Magnitude': trend_magnitude,
                'Volatility': volatility,
                'Volatility_Level': volatility_level,
                'Max_Price': max_price,
                'Min_Price': min_price,
                'Price_Range': price_range
            }
            print(f"  Volatility and Risk: {volatility_analysis}")
            print(f"  Market Opportunities/Risks: {opportunities_risks}")
            
            logging.info(f"{model_name} Analysis Results:")
            logging.info(f"  Trend: {trend}")
            logging.info(f"  Trend Magnitude: {trend_magnitude:.2f}")
            logging.info(f"  Volatility: {volatility:.2f}")
            logging.info(f"  Volatility Level: {volatility_level}")
            logging.info(f"  Max Price: {max_price:.2f}")
            logging.info(f"  Min Price: {min_price:.2f}")
            logging.info(f"  Price Range: {price_range:.2f}")
            logging.info(f"  Volatility and Risk: {volatility_analysis}")
            logging.info(f"  Market Opportunities/Risks: {opportunities_risks}")
        
        analysis_df = pd.DataFrame(analysis_results).T
        return analysis_df

    def _volatility_risk_analysis(self, forecast, threshold):
        volatility = np.std(forecast)
        volatility_level = "High" if volatility > threshold else "Low"
        increasing_volatility = any(np.diff(forecast) > np.mean(np.diff(forecast)))
        
        if increasing_volatility:
            return "Potential increase in volatility, which could lead to market risk."
        else:
            return "Stable volatility, lower risk."

    def _market_opportunities_risks(self, trend, volatility_level):
        if trend == "upward":
            if volatility_level == "High":
                return "Opportunity with high risk due to increased volatility."
            else:
                return "Opportunity with moderate risk due to stable volatility."
        elif trend == "downward":
            if volatility_level == "High":
                return "Risk of decline with high uncertainty."
            else:
                return "Moderate risk of decline with low volatility."
        else:
            return "Stable market, with minimal risks."