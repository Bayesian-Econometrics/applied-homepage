# Data snapshots

Refreshed 2026-09-10 by `tools/refresh_data.py`.

Sources: Federal Reserve Bank of St. Louis (FRED) and the Kenneth R.
French data library, Tuck School of Business at Dartmouth. Factor and
portfolio returns are stored as decimals, not percent.

| File | Rows | Columns | First | Last |
|---|--:|--:|---|---|
| fred_macro_monthly.parquet | 800 | 6 | 1960-01-01 | 2026-08-01 |
| fred_macro_quarterly.parquet | 266 | 2 | 1960-01-01 | 2026-04-01 |
| ff_factors_monthly.parquet | 1201 | 4 | 1926-07-01 | 2026-07-01 |
| ff_factors_daily.parquet | 26296 | 4 | 1926-07-01 | 2026-07-31 |
| ff_momentum_monthly.parquet | 1195 | 1 | 1927-01-01 | 2026-07-01 |
| ff_industry49_monthly.parquet | 1201 | 49 | 1926-07-01 | 2026-07-01 |
