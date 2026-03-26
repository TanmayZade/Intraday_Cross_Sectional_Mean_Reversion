import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Read the clean data
table = pq.read_table('polygon_pipeline/data/clean/year=2024/data.parquet')
df = table.to_pandas()

print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\nDate range:')
if 'timestamp' in df.columns:
    print('  Min:', df['timestamp'].min())
    print('  Max:', df['timestamp'].max())
    print('  Unique dates:', df['timestamp'].dt.date.nunique())

print('\nUnique tickers:', df['ticker'].nunique() if 'ticker' in df.columns else 'N/A')
print('\nFirst few rows:')
print(df.head(10))
print('\nLast few rows:')
print(df.tail(10))
print('\nData types:')
print(df.dtypes)
print('\nNulls:')
print(df.isnull().sum())
