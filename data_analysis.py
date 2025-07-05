import os
import glob
import re
import hashlib
from typing import Tuple, Sequence
from datetime import datetime

from concurrent.futures import as_completed, ThreadPoolExecutor
import multiprocessing

import yfinance as yf
import requests_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from statsmodels.tsa.stattools import adfuller

from tqdm import tqdm

from numba_utils import ols_residuals, corr_coef

# cache every request to disk (SQLite), expire items after 1 day
requests_cache.install_cache(
    "yf_cache", 
    backend="sqlite", 
    expire_after=24*3600,
    allowable_methods=("GET", "POST")
)

COLUMNS = ["Stock1", "Stock2", "corr", "ADF_stat", "p_value", "beta", "alpha", "half_life"]
SECTORS = ['technology', 'healthcare', 'financial-services', 'consumer-cyclical', 
    'industrials', 'communication-services', 'consumer-defensive', 
    'energy', 'utilities', 'real-estate', 'basic-materials']
START = "2024-01-01"

@dataclass                               
class CointegrationData:
    data: pd.DataFrame = field(repr=False)

    def __post_init__(self):

        # compute for any missing/extra columns
        missing = set(COLUMNS) - set(self.data.columns)
        extra   = set(self.data.columns) - set(COLUMNS)
        if missing:
            raise ValueError(f"CointegrationData missing cols: {missing}")
        if extra:
            raise ValueError(f"CointegrationData extra cols: {extra}")

        # if there are more rows than 10, trim to that many and force a copy
        if len(self.data) > 10:
            self.data = self.data.iloc[:10].copy()

        nums = COLUMNS[2:]
        self.data.loc[:, nums] = (
            self.data.loc[:, nums]
                .apply(pd.to_numeric, errors="raise")
        )

        self.pairs = self.data[["Stock1", "Stock2"]].values
        self.pairs = [ticker for pair in self.pairs for ticker in pair]
        self.pairs = list(dict.fromkeys(self.pairs))

        self.pair_close_prices = yf.download(self.pairs, start=START)["Close"]

    def show(self):
        print(self.data.to_string(float_format='%.10f'))

    def get(self):
        return self.data
    
    def plot_spread(self, pair: int = 0, save_path: str = None):

        s1, s2 = self.data.loc[pair, ["Stock1","Stock2"]].tolist()

        beta  = float(self.data.at[pair, "beta"])
        alpha = float(self.data.at[pair, "alpha"])

        prices = self.pair_close_prices[[s1, s2]]

        spread = prices[s1] - (alpha + beta * prices[s2])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(spread, label="Spread")
        ax.axhline(spread.mean(), color="red", ls="--", label="Mean")
        ax.set_title(f"Cointegrated Spread: {s1} − (α + β·{s2})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Spread ($)")
        ax.legend(); ax.grid(True); fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
    
    def cache(self, path=None):
        start = START
        end = datetime.today().strftime("%Y-%m-%d")

        if path == None:
            dir = os.path.join("data", "pairs")
            path = os.path.join(dir, f"cointegration_top10_{start}_{end}.pkl")
            
            os.makedirs(dir, exist_ok=True)

        print(f"saving cointegration data to: {path}")
        self.data.to_pickle(path)

        fig_dir = os.path.join("data", "pairs", "spread")
        os.makedirs(fig_dir, exist_ok=True)
        for idx, row in self.data.iterrows():
            s1, s2 = row.Stock1, row.Stock2
            fname  = f"{s1}_{s2}.png"
            fp     = os.path.join(fig_dir, fname)
            if os.path.exists(fp):
                continue    
            self.plot_spread(
                pair=idx,
                save_path=fp
            )

@dataclass
class PriceData:
    prices: pd.DataFrame = field(repr=False)

    def __post_init__(self):

        self.tickers = self.prices.columns.to_list()

        self.parse_data()

        if self.prices.empty:
            raise ValueError("No data found")
        if len(self.prices.columns) < 2:
            raise ValueError("Insufficient data, at least 2 stocks required")
        
    def parse_data(self) -> Tuple[pd.DataFrame, list[str]]:
        """Return clean price data by removing NaN, inf, and invalid stocks or ones with insufficient data."""
        print(f"Original data shape: {self.prices.shape}")
        
        old = self.prices

        #remove columns with too many missing values
        min_valid_rows = int(len(self.prices) * 0.9)
        self.prices = self.prices.dropna(thresh=min_valid_rows, axis=1)
        
        # forward fill and backward fill
        self.prices = self.prices.ffill().bfill()
        
        # Sdrop any remaining rows with NaN
        self.prices = self.prices.dropna()
        
        # remove infinite values
        self.prices = self.prices.replace([np.inf, -np.inf], np.nan)
        self.prices = self.prices.dropna()
        
        # remove stocks with zero/negative prices
        self.prices = self.prices.loc[:, (self.prices > 0).all()]
        
        # remove stocks with very low price variation
        price_std = self.prices.std()
        min_std = 0.01
        self.prices = self.prices.loc[:, price_std > min_std]
        
        # rmove stocks with extreme price changes
        daily_returns = self.prices.pct_change().dropna()
        extreme_return_threshold = 0.5
        mask = (daily_returns.abs() > extreme_return_threshold).any(axis=0)
        extreme_stocks = list(daily_returns.columns[mask])
        
        if extreme_stocks:
            print(f"Removing {len(extreme_stocks)} stocks with extreme price movements: {extreme_stocks[:5]}{'...' if len(extreme_stocks) > 5 else ''}")
            self.prices = self.prices.drop(columns=extreme_stocks)
        
        print(f"Cleaned data shape: {self.prices.shape}")
        print(f"Removed {old.shape[1] - self.prices.shape[1]} stocks due to data quality issues")
            
    def show(self):
        print(self.prices)
    
    def analyse_data(self, corr_thresh: float = 0.5, n_jobs: int = None) -> CointegrationData:
        
        cols = self.prices.columns.to_list()
        X = self.prices.values                # shape (T, N)

        # full corr‐matrix in np
        C = np.corrcoef(X, rowvar=False)             # shape (N, N)
        C = np.abs(C)

        # mask upper‐triangle, threshold ≥ pre_thresh
        pre_thresh = max(corr_thresh, 0.8)
        N = C.shape[0]
        iu = np.triu_indices(N, k=1)
        mask = C[iu] >= pre_thresh                  # 1D boolean array, len = N*(N-1)/2

        # extract i,j and corr values
        i_all = iu[0][mask]
        j_all = iu[1][mask]
        corr_all = C[iu][mask]

        # pick topK by corr with a single partial sort
        K = 50_000
        if len(corr_all) > K:
            idx = np.argpartition(-corr_all, K)[:K]
            i_top = i_all[idx]
            j_top = j_all[idx]
            corr_top = corr_all[idx]
            # now sort K descending
            order = np.argsort(-corr_top)
            i_top = i_top[order]
            j_top = j_top[order]
            corr_top = corr_top[order]
        else:
            i_top, j_top, corr_top = i_all, j_all, corr_all

        total = len(corr_top)
        print(f"Running ADF on top {total:,} pairs by corr≥{pre_thresh:.2f}")

        jobs = [
            (X[:, [ii, jj]], cols[ii], cols[jj])
            for ii, jj in zip(i_top, j_top)
        ]

        # expensive ADF on reduced set with multiprocessing
        records = []
        with multiprocessing.Pool(processes=n_jobs, initializer=_init_worker) as pool:
            for s1, s2, corr, adf, p, b1, b0, hl in tqdm(
                pool.imap_unordered(analyse_pair_raw, jobs, chunksize=5_000),
                total=total,
                desc="Analyzing pairs",
                unit="pair"
            ):
                records.append((s1, s2, corr, adf, p, b1, b0, hl))

        if not records:
            raise RuntimeError("No valid pairs found after filtering")

        df = (
            pd.DataFrame(records, columns=COLUMNS)
              .sort_values(["p_value","half_life"])
              .reset_index(drop=True)
        )
        return CointegrationData(df)
        

@dataclass
class Stocks:
    stocks: Sequence[str] = field(init=False, repr=False)

    def __init__(self, *symbols: str):

        if len(symbols) == 1 and isinstance(symbols[0], (list, tuple)):
            self.stocks = list(symbols[0])
        else:
            self.stocks = list(symbols)

        self.__post_init__()

    def __post_init__(self):

        if not isinstance(self.stocks, (list, tuple)):
            raise TypeError(f"Expected a list or tuple of strings, got {type(self.stocks).__name__}")
        
        bad = [s for s in self.stocks if not isinstance(s, str)]

        if bad:
            self.stocks = [s for s in self.stocks if s not in bad]
            print(f"All items must be str, removed invalid entries: {bad}")
        
        if len(self.stocks) < 2:
            raise ValueError(f"Expected at least 2 symbols, got {len(self.stocks)}: {self.stocks}")

        self.stocks = list(self.stocks)
    
    def show(self):
        print(self.stocks)
    
    def get_close_prices(self, start: str, end: str):
        cache_dir = os.path.join("data", "prices", "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # look for cache file matching prices_YYYY-MM-DD_YYYY-MM-DD_*.pkl
        pattern = os.path.join(cache_dir, "prices_*.pkl")
        for fn in glob.glob(pattern):
            fname = os.path.basename(fn)
            m = re.match(r"prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_[0-9a-f]{8}\.pkl", fname)
            if not m:
                continue

            cached_start, cached_end = m.groups()
            if cached_start <= start and cached_end >= end:
                df = pd.read_pickle(fn)
                if set(self.stocks).issubset(df.columns):
                    print(f"Loaded cached data from {fn}")
                    sub = df.loc[start:end, self.stocks]
                    return PriceData(sub)

        print("Downloading close prices:")
        df = yf.download(
            self.stocks,
            start=start,
            end=end,
            auto_adjust=True,
            progress=True,
            threads=True
        )["Close"]

        key    = ",".join(sorted(self.stocks))
        h      = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        fname  = os.path.join(cache_dir, f"prices_{start}_{end}_{h}.pkl")
        df.to_pickle(fname)

        return PriceData(df)

@dataclass
class Sector:
    sectors: Sequence[str] = field(init=False, repr=False)

    def __init__(self, *sectors : str):

        if len(sectors) == 1 and isinstance(sectors[0], (list, tuple)):
            self.sectors = list(sectors[0])
        else:
            self.sectors = list(sectors)

        self.__post_init__()

    def __post_init__(self):

        if not isinstance(self.sectors, (list, tuple)):
            raise TypeError(f"Expected a list or tuple of strings, got {type(self.sectors).__name__}")

        if all(yf.Sector(s) is None for s in self.sectors):
            invalid = [s for s in self.sectors if yf.Sector(s).name is None]
            self.sectors = [s for s in self.sectors if yf.Sector(s).name is not None]
            print(f"Removed invalid sectors: {invalid}")
    
    def show(self):
        print(self.sectors)
    
    def get_industries(self):

        industries = []
        
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(yf.Sector, s) : s for s in self.sectors
            }

            for fut in as_completed(futures):
                sector = futures[fut]

                try:
                    print(f"Processing sector: {sector}")
                    sector_industries = fut.result().industries

                    if sector_industries is None:
                        print(f"No data returned for sector {sector} (possibly rate limited)")
                        continue

                    print(f"Found {len(sector_industries)} industries in {sector}")

                    for i in sector_industries.index.to_list():
                        industries.append(i)

                except Exception as e:
                    print(f"Error accessing sector {sector}: {e}")
                    continue              

        industries = Industry(industries)
        return industries
    
@dataclass
class Industry:
    industries: Sequence[str] = field(init=False, repr=False)

    def __init__(self, *industries : str):

        if len(industries) == 1 and isinstance(industries[0], (list, tuple)):
            self.industries = list(industries[0])
        else:
            self.industries = list(industries)

        self.__post_init__()

    def __post_init__(self):
                
        if not isinstance(self.industries, (list, tuple)):
            raise TypeError(f"Expected a list or tuple of strings, got {type(self.industries).__name__}")

        if all(yf.Industry(s) is None for s in self.industries):
            invalid = [s for s in self.industries if yf.Industry(s).name is None]
            self.industries = [s for s in self.industries if yf.Industry(s).name is not None]
            print(f"Removed invalid industries: {invalid}")
    
    def show(self):
        print(self.industries)

    def get_stocks(self):
        
        stocks = []

        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(yf.Industry, i): i for i in self.industries
            }

            for fut in as_completed(futures):

                industry = futures[fut]

                try:
                    print(f"Processing industry: {industry}")
                    top_companies = fut.result().top_companies
                    if top_companies is not None and not top_companies.empty:

                        symbols = top_companies.index.tolist()
                        print(f"Found {len(symbols)} companies")
                        for s in symbols:
                            stocks.append(s)

                    else:
                        print(f"No companies found for industry {industry}")  
                except Exception as e:
                    print(f"Error getting companies for industry {industry}: {e}")

        stocks = Stocks(stocks)
        return stocks

def analyse_pairs(prices: pd.DataFrame, y_sym: str, x_sym: str):
    """Analyze a pair of stocks for cointegration"""
    try:
        y = prices[y_sym].values
        x = prices[x_sym].values
        
        min_length = min(len(y), len(x))
        if min_length < 100:
            raise ValueError(f"Insufficient data: only {min_length} points")
        
        y = y[:min_length]
        x = x[:min_length]
        
        # Additional validation
        if np.any(np.isnan(y)) or np.any(np.isnan(x)):
            raise ValueError("Contains NaN values")
        
        if np.any(np.isinf(y)) or np.any(np.isinf(x)):
            raise ValueError("Contains infinite values")
        
        if np.std(y) == 0 or np.std(x) == 0:
            raise ValueError("Zero variance in price series")

        # check for constant x
        if np.allclose(x, x[0]):
            raise ValueError("Multicollinearity / constant regressor detected")

        # jit’d math
        beta0, beta1, resid = ols_residuals(y, x)
        corr_val           = corr_coef(y, x)

        # stationarity test
        adf_stat, p_val, *_ = adfuller(resid, autolag="AIC")

        return corr_val, adf_stat, p_val, beta1, beta0

    except Exception as e:
        raise ValueError(f"Analysis failed for {y_sym}-{x_sym}: {str(e)}")



def _init_worker():
    # warm up Numba functions once per process
    import numpy as np
    dummy_x = np.arange(200, dtype=np.float64)
    dummy_y = dummy_x * 2.0
    ols_residuals(dummy_y, dummy_x)
    corr_coef(dummy_y, dummy_x)

def half_life(resid):
    """
    Estimate the speed of mean-reversion via regression
    Δr_t = β·r_{t-1} + ε ⇒ half-life = –ln(2)/β.
    Return NaN if the linear fit fails.
    """
    try:
        delta  = np.diff(resid)
        lagged = resid[:-1]
        # linear fit... occasional SVD-fail
        beta = np.polyfit(lagged, delta, 1)[0]
        return -np.log(2) / beta
    except np.linalg.LinAlgError:
        return np.nan
    except Exception:
        return np.nan

def process_sector_pairs(sector_name: str):
    """Return list of (s1,s2,corr,adf,p,b1,b0) for one sector."""
    s  = Sector(sector_name)
    i  = s.get_industries()
    st = i.get_stocks()
    pr = st.get_close_prices(
        start=START,
        end=datetime.today().strftime("%Y-%m-%d")
    )
    
    cols    = pr.prices.columns
    corr_m  = pr.prices.corr().abs()
    cand    = [
      (sector_name, cols[i], cols[j])
      for i in range(len(cols))
      for j in range(i+1, len(cols))
      if corr_m.iat[i,j] >= 0.5
    ]
    
    jobs = [
      (np.array(pr.prices[[s1,s2]].values), s1, s2)
      for (_,s1,s2) in cand
    ]

    return list(zip(cand, jobs))

def analyse_one_pair(args):
    """Unpack and call your old analyse_pairs"""
    (_sector,s1,s2), (arr, y_sym, x_sym) = args
    corr, adf, p, b1, b0 = analyse_pairs(pd.DataFrame(arr, columns=[y_sym,x_sym]), y_sym, x_sym)
    return (_sector, s1, s2, corr, adf, p, b1, b0)

def analyse_pair_raw(job):
    """
    job is (arr, s1, s2) where arr.shape == (T,2)
    """
    arr, s1, s2 = job
    y = arr[:,0]; x = arr[:,1]
    n = min(y.size, x.size)
    if n < 100:
        raise ValueError(f"Insufficient data: {n} points")
    y, x = y[:n], x[:n]

    if np.any(np.isnan(y)) or np.any(np.isnan(x)):
        raise ValueError("NaN present")
    if np.any(np.isinf(y)) or np.any(np.isinf(x)):
        raise ValueError("Inf present")
    if np.std(y)==0 or np.std(x)==0 or np.allclose(x, x[0]):
        raise ValueError("Bad variance / constant regressor")

    b0, b1, resid = ols_residuals(y, x)
    try:
        hl = half_life(resid)
    except Exception:
        hl = np.nan
    corr_val = corr_coef(y, x)
    try:
        adf_stat, p, *_ = adfuller(resid, autolag="AIC")
    except ValueError as e:
        adf_stat, p = np.nan, np.nan
    return s1, s2, corr_val, adf_stat, p, b1, b0, hl

def test():
    multiprocessing.freeze_support()

    all_stocks = []
    for sector_name in SECTORS:
        secs = Sector(sector_name).get_industries()
        stk_obj = secs.get_stocks()
        all_stocks.extend(stk_obj.stocks)

    unique_stocks = sorted(set(all_stocks))

    prices_df = Stocks(unique_stocks) \
                   .get_close_prices(start=START,
                                     end=datetime.today().strftime("%Y-%m-%d")) \
                   .prices

    # full screening of all stocks of every sector
    pd_obj = PriceData(prices_df)
    ci = pd_obj.analyse_data(corr_thresh=0.5,
                             n_jobs=multiprocessing.cpu_count())
    ci.show()
    ci.cache()
    ci.plot_spread(0)
    

if __name__=="__main__":
    test()