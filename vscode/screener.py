import pandas as pd
import yfinance as yf
import sys

# List of available sectors
sector_names = [
    'technology', 'healthcare', 'financial-services', 'consumer-cyclical', 
    'industrials', 'communication-services', 'consumer-defensive', 
    'energy', 'utilities', 'real-estate', 'basic-materials'
]

class Sectors:
    def __init__(self, name):
        if name in sector_names:
            self.name = name
            print(self.name)
        else:
            print("invalid type")

tech = Sectors("technology")
test = Sectors("test")
# # Iterate through each sector and display its industries
# for sector_name in sector_names:
#     print(f"\n{'='*50}")
#     print(f"SECTOR: {sector_name.upper().replace('-', ' ')}")
#     print(f"{'='*50}")
    
#     try:
#         sector = yf.Sector(sector_name)
#         industries = sector.industries
        
#         print(f"Number of industries: {len(industries)}")
#         print("\nIndustries:")
        
#         # Display each industry
#         for index, (key, row) in enumerate(industries.iterrows(), 1):
#             print(f"{index:2d}. {row['name']:<40} | Symbol: {row['symbol']:<15} | Weight: {row['market weight']:.4f}")
            
#     except Exception as e:
#         print(f"Error accessing {sector_name}: {e}")
    
#     print("-" * 50)