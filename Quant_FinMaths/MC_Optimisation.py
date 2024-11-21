# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Alpha Vantage API endpoint
url = 'https://www.alphavantage.co/query'

# Asset symbols for the multi-asset portfolio
symbols = ['MSFT', 'AAPL', 'GOOGL']
api_key = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key

# Function to fetch historical data for a given symbol
def fetch_data(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    time_series = data.get('Time Series (Daily)', {})
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert strings to numbers
    df = df[['Close']]  # Only keep the 'Close' column
    df.columns = [symbol]  # Rename the column to the symbol
    return df

# Fetch historical data for all symbols
data_frames = [fetch_data(symbol) for symbol in symbols]
data = pd.concat(data_frames, axis=1, join='inner')  # Merge data on common dates

# Calculate daily returns and the correlation matrix
returns = data.pct_change().dropna()
correlation_matrix = returns.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Portfolio weights (assuming equal weight for simplicity)
weights = np.array([1/len(symbols)] * len(symbols))

# Monte Carlo Simulation Parameters
N = 1000  # Number of simulations
N_days = 252  # Number of trading days in a year
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Cholesky decomposition for correlation handling
L = np.linalg.cholesky(cov_matrix)

# Define scenarios and their modifications (for each asset)
scenarios = {
    "20% Crash at Midpoint": lambda prices, day, next_prices: next_prices * 0.8 if day == N_days // 2 else next_prices,
    "30% Crash at Start": lambda prices, day, next_prices: next_prices * 0.7 if day == 0 else next_prices,
    "Increased Volatility (Second Half)": lambda prices, day, next_prices: next_prices * (1 + np.random.multivariate_normal(mean_returns, cov_matrix * (2 if day > N_days // 2 else 1))),
    "Multiple Shocks (10% at Day 50, 15% at Day 200)": lambda prices, day, next_prices: next_prices * 0.9 if day == 50 else (next_prices * 0.85 if day == 200 else next_prices)
}

# Directory to save the plots
save_dir = '/Users/rossmcphee/Documents/Quant_FinMaths/'

# Function to calculate drawdown
def calculate_drawdown(prices):
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return drawdown.min()

# Loop through each scenario, run the simulation, plot, and calculate portfolio metrics
for scenario_name, stress_function in scenarios.items():
    # Initialize simulations
    simulations = np.zeros((N, N_days))
    portfolio_initial_price = np.dot(data.iloc[-1].values, weights)  # Initial portfolio price based on weights

    # Run the simulation
    for i in range(N):
        prices = [portfolio_initial_price]
        asset_prices = data.iloc[-1].values  # Starting prices for each asset
        for day in range(N_days):
            # Generate correlated random returns
            correlated_randoms = np.dot(L, np.random.normal(size=len(symbols)))
            daily_returns = mean_returns.values + correlated_randoms
            next_prices = asset_prices * (1 + daily_returns)
            
            # Apply stress function
            next_prices = stress_function(prices, day, next_prices)
            
            # Calculate the portfolio price
            portfolio_price = np.dot(next_prices, weights)
            prices.append(portfolio_price)
            asset_prices = next_prices
        
        simulations[i, :] = prices[1:]  # Only store the first N_days prices, ignoring the initial

    # Calculate portfolio metrics
    ending_prices = simulations[:, -1]
    portfolio_returns = (ending_prices - portfolio_initial_price) / portfolio_initial_price
    confidence_level = 95
    VaR = np.percentile(portfolio_returns, 100 - confidence_level)
    CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
    max_drawdown = calculate_drawdown(ending_prices)

    # Print metrics for the scenario
    print(f"{scenario_name}:")
    print(f"  Value at Risk (VaR) at {confidence_level}% confidence level: {VaR:.2%}")
    print(f"  Conditional Value at Risk (CVaR): {CVaR:.2%}")
    print(f"  Maximum Drawdown: {max_drawdown:.2%}\n")

    # Plot the simulation
    plt.figure(figsize=(12, 6))
    plt.plot(simulations.T, color='grey', alpha=0.1)  # Adjust alpha for more discernible lines
    plt.title(f'Monte Carlo Simulations with {scenario_name}')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Price')
    plt.grid(True)  # Add gridlines for better readability

    # Annotate VaR, CVaR, and Drawdown on the plot
    text_x = N_days * 0.05  # Position text at 5% of the x-axis
    text_y = np.max(simulations) * 0.9  # Position text at 90% of the y-axis
    plt.text(text_x, text_y, f'VaR ({confidence_level}%): {VaR:.2%}\nCVaR: {CVaR:.2%}\nMax Drawdown: {max_drawdown:.2%}', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # Save the plot to a PDF file
    plot_file_path = os.path.join(save_dir, f'{scenario_name.replace(" ", "_")}_MultiAsset.pdf')
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to avoid displaying it multiple times

    print(f"Saved: {plot_file_path}")

