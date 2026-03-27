"""Quick data exploration script"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv(r'c:\Users\shivt\Downloads\FT Data - data.csv')
    print(f"✓ Data loaded successfully\n")
    print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print(f"Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nFirst 2 rows:")
    print(df.head(2).to_string())
    print(f"\nBasic Info:")
    print(f"  Total entries: {len(df)}")
    df.info()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
