import pandas as pd
import yfinance as yf
import sys

# List of available sectors
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
            self.names = name
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
                for i, (k, r) in enumerate(yf.Sector(n).industries.iterrows(), 1):
                    
                    #print(f"\nIndustry: {k}")

                    try:
                        industry = yf.Industry(k)
                        top_companies = industry.top_companies
    
                        # Check if DataFrame is not empty
                        if not top_companies.empty:
                            # The symbols are in the index, not a column!
                            symbols = top_companies.index.tolist()
                            self.symbols[k] = symbols
                            #print(f"Company symbols: {symbols}")
                            
                        else:
                            print("No companies found for this industry")

                    except Exception as e:
                        print(f"Error getting companies for industry {r['name']}: {e}")
        else:
            for i in industries:
                for s in self.names:
                    if i in s.industries:
                        pass
        return self.symbols


a = Sectors(parse_args())
tickers = a.get_tickers()

for i in tickers:
    for t in tickers[i]:
        print(t)
