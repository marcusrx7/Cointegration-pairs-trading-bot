import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
import time
from datetime import datetime
from tqdm import tqdm
import sys

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

sector_names = [
    'technology', 'healthcare', 'financial-services', 'consumer-cyclical', 
    'industrials', 'communication-services', 'consumer-defensive', 
    'energy', 'utilities', 'real-estate', 'basic-materials'
]

def parse_args():
    print(sys.argv)
    return sys.argv[1:]

class Sectors:
    def __init__(self, name):

        self.names = []
        self.symbols = {}

        if type(name) == list:

            if len(name):
                for sector in name:
                    if sector in sector_names:
                        self.names.append(sector)
                        print("accessing", sector + "...")
                    else:
                        print(sector, "is an invalid type")
            else:
                for n in sector_names:
                    self.names.append(n)

        elif type(name) == str and name in sector_names:
            self.names = [name]
            print(self.names)
        else:
            print(name, "is an invalid type")

    def show(self, filter=None):

        names = self.names

        if filter != None:
            names = []
            if len(filter):
                for f in filter:
                    print(f)
                    if f in self.names:
                        names.append(f)
            elif filter in sector_names:
                names.append(filter)

        for sector_name in names:
            print(f"\n{'='*50}")
            print(f"SECTOR: {sector_name.upper().replace('-', ' ')}")
            print(f"{'='*50}")

            try:
                sector = yf.Sector(sector_name)
                industries = sector.industries
                # 
                print(f"Number of industries: {len(industries)}")
                print("\nIndustries:")
                # 
                # Display each industry
                for index, (key, row) in enumerate(industries.iterrows(), 1):
                    print(f"{index:2d}. {row['name']:<40} | Symbol: {row['symbol']:<15} | Weight: {row['market weight']:.4f}")
                    # 
            except Exception as e:
                print(f"Error accessing {sector_name}: {e}")

            print("-" * 50)
    
    def get_tickers(self, industries=None):

        if industries == None:
            for n in self.names:
                print(f"Processing sector: {n}")

                try:
                    # Add delay to avoid rate limiting
                    time.sleep(2)  # Wait 2 seconds between requests

                    sector = yf.Sector(n)
                    sector_industries = sector.industries

                    # Check if we got valid data (rate limiting returns None)
                    if sector_industries is None:
                        print(f"No data returned for sector {n} (possibly rate limited)")
                        continue

                    print(f"Found {len(sector_industries)} industries in {n}")

                    for i, (k, r) in enumerate(sector_industries.iterrows(), 1):
                        try:
                            # Add delay between industry requests
                            time.sleep(1)

                            print(f"  Processing industry: {r['name']}")
                            industry = yf.Industry(k)
                            top_companies = industry.top_companies

                            # Check if DataFrame is not empty and not None
                            if top_companies is not None and not top_companies.empty:
                                # The symbols are in the index
                                symbols = top_companies.index.tolist()
                                self.symbols[k] = symbols
                                print(f"Found {len(symbols)} companies")
                            else:
                                print(f"No companies found for industry {r['name']}")

                        except Exception as e:
                            print(f"Error getting companies for industry {r['name']}: {e}")
                            continue

                except Exception as e:
                    print(f"Error accessing sector {n}: {e}")
                    continue

        else:
            for i in industries:
                for s in self.names:
                    if i in s.industries:
                        pass
        return self.symbols

class build_results:
    
    def __init__(self, prices):
        self.prices = prices
        
        # Clean the data first
        self.prices = self.clean_data(prices)
        
        if self.prices.empty or len(self.prices.columns) < 2:
            print("Not enough clean data for analysis")
            return
            
        # ---------- Iterate through all pairs ----------
        records = []
        pairs = combinations(self.prices.columns, 2)
        total_pairs = len(self.prices.columns) * (len(self.prices.columns) - 1) // 2

        print(f"Analyzing {total_pairs} pairs from {len(self.prices.columns)} stocks...")

        for s1, s2 in tqdm(pairs, total=total_pairs):
            try:
                corr, adf_stat, p_val, beta, alpha = self.analyse_pair(s1, s2)
                records.append((s1, s2, corr, adf_stat, p_val, beta, alpha))
            except Exception as e:
                # Don't print every error, just count them
                continue

        if not records:
            print("No valid pairs found for analysis")
            return

        # ---------- Build results DataFrame ----------
        cols = ["Stock1", "Stock2", "corr", "ADF_stat", "p_value", "beta", "alpha"]
        results = (
            pd.DataFrame(records, columns=cols)
              .sort_values("p_value")              # sort by stationarity strength
              .reset_index(drop=True)
        )

        # ---------- Display ----------
        print(f"\nTotal pairs analysed: {len(results):,}")
        #print("\nTop 10 by lowest p-value:")
        #print(results.head(10))

        self.results = results

    def clean_data(self, prices):
        """Clean price data by removing NaN, inf, and stocks with insufficient data"""
        print(f"Original data shape: {prices.shape}")
        
        # Step 1: Remove columns with too many missing values
        missing_threshold = 0.1  # Allow up to 10% missing values
        min_valid_rows = int(len(prices) * (1 - missing_threshold))
        prices_clean = prices.dropna(thresh=min_valid_rows, axis=1)
        
        # Step 2: Forward fill and backward fill
        prices_clean = prices_clean.ffill().bfill()
        
        # Step 3: Drop any remaining rows with NaN
        prices_clean = prices_clean.dropna()
        
        # Step 4: Remove infinite values
        prices_clean = prices_clean.replace([np.inf, -np.inf], np.nan)
        prices_clean = prices_clean.dropna()
        
        # Step 5: Remove stocks with zero or negative prices
        prices_clean = prices_clean.loc[:, (prices_clean > 0).all()]
        
        # Step 6: Remove stocks with very low price variation (likely data errors)
        price_std = prices_clean.std()
        min_std = 0.01  # Minimum standard deviation
        prices_clean = prices_clean.loc[:, price_std > min_std]
        
        # Step 7: Remove stocks with extreme price changes (likely splits not adjusted)
        daily_returns = prices_clean.pct_change().dropna()
        extreme_return_threshold = 0.5  # 50% daily change threshold
        extreme_stocks = []
        
        for col in daily_returns.columns:
            if (abs(daily_returns[col]) > extreme_return_threshold).any():
                extreme_stocks.append(col)
        
        if extreme_stocks:
            print(f"Removing {len(extreme_stocks)} stocks with extreme price movements: {extreme_stocks[:5]}{'...' if len(extreme_stocks) > 5 else ''}")
            prices_clean = prices_clean.drop(columns=extreme_stocks)
        
        print(f"Cleaned data shape: {prices_clean.shape}")
        print(f"Removed {prices.shape[1] - prices_clean.shape[1]} stocks due to data quality issues")
        
        return prices_clean

    def analyse_pair(self, y_sym, x_sym):
        """Analyze a pair of stocks for cointegration"""
        try:
            # Get the price series
            y = self.prices[y_sym].values
            x = self.prices[x_sym].values
            
            # Ensure we have the same length and enough data
            min_length = min(len(y), len(x))
            if min_length < 100:  # Increased minimum data requirement
                raise ValueError(f"Insufficient data: only {min_length} points")
            
            # Trim to same length
            y = y[:min_length]
            x = x[:min_length]
            
            # Additional validation
            if np.any(np.isnan(y)) or np.any(np.isnan(x)):
                raise ValueError("Contains NaN values")
            
            if np.any(np.isinf(y)) or np.any(np.isinf(x)):
                raise ValueError("Contains infinite values")
            
            if np.std(y) == 0 or np.std(x) == 0:
                raise ValueError("Zero variance in price series")
            
            # Build regression matrix with constant
            X = add_constant(x)
            
            # Final check on regression matrix
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Invalid regression matrix")
            
            # Check for multicollinearity (constant column should not equal x column)
            if np.allclose(X[:, 0], X[:, 1]):
                raise ValueError("Multicollinearity detected")
            
            # Fit regression
            model = OLS(y, X).fit()
            beta0, beta1 = model.params
            
            # Calculate residuals
            resid = y - (beta0 + beta1 * x)
            
            # Check residuals
            if np.std(resid) == 0:
                raise ValueError("Zero variance in residuals")
            
            # Test for stationarity
            adf_stat, p_val, *_ = adfuller(resid, autolag='AIC')
            
            # Calculate correlation
            corr_val = np.corrcoef(y, x)[0, 1]
            
            return corr_val, adf_stat, p_val, beta1, beta0
            
        except Exception as e:
            raise ValueError(f"Analysis failed for {y_sym}-{x_sym}: {str(e)}")

try:
    a = Sectors(parse_args())
    tickers = a.get_tickers()
    
    if not tickers:
        print("No tickers found. Exiting.")
        exit()

    prices = {}
    all_results = {}

    for i in tickers:
        if not tickers[i]:  # Skip empty ticker lists
            continue
            
        print(f"Downloading data for {len(tickers[i])} companies in industry: {i}")
        
        try:
            price_data = yf.download(
                tickers[i],
                start="2021-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True
            )["Close"]
            
            if price_data.empty:
                print(f"No price data found for industry {i}")
                continue
                
            # Basic cleaning before adding to list
            price_data = price_data.ffill().dropna(axis=1, how="all")
            
            if not price_data.empty and len(price_data.columns) >= 2:
                prices[i] = price_data
                print(f"Added {len(price_data.columns)} stocks from industry {i}")
            else:
                print(f"Insufficient data for industry {i} (need at least 2 stocks)")
                
        except Exception as e:
            print(f"Error downloading data for industry {i}: {e}")
            continue

    # Process results only if we have data
    if prices:
        for p in prices:
            if not prices[p].empty and len(prices[p].columns) >= 2:
                print(f"\nProcessing {p} industry with {len(prices[p].columns)} stocks...")
                result_obj = build_results(prices[p])
                if hasattr(result_obj, "results") and result_obj.results is not None:
                    all_results[p] = result_obj.results
                else:
                    print(f"No valid pairs found for industry {p}")
            else:
                print("Skipping industry - insufficient data")
    else:
        print("No price data available for analysis")
    
    for r in all_results:
        print(f"\n--- {r} ---")
        if all_results[r] is not None and not all_results[r].empty:
            print(all_results[r].head(10))
        else:
            print("No results to display for this industry.")
        
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
