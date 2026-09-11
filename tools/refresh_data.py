"""Refresh the course data snapshots.

Run this on a machine with internet access:

    uv run python tools/refresh_data.py

It downloads the two sources the lecture notes use, FRED for macroeconomic
series and Kenneth French's data library for asset-pricing factors, and writes
Parquet snapshots into data/. The lecture notes read only those snapshots, so
the site renders offline and every printed number is reproducible until this
script is run again.
"""

import io
import zipfile
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
START = "1960-01-01"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
              "ftp/{name}_CSV.zip")

FRED_MONTHLY = {
    "CPIAUCSL": "cpi",              # consumer price index, seasonally adjusted
    "PCEPILFE": "core_pce",         # core PCE price index
    "UNRATE": "unemployment",       # civilian unemployment rate
    "FEDFUNDS": "fed_funds",        # effective federal funds rate
    "GS10": "yield_10y",            # 10-year Treasury constant maturity yield
    "INDPRO": "industrial_prod",    # industrial production index
}
FRED_QUARTERLY = {
    "GDPC1": "real_gdp",            # real gross domestic product
    "PCECC96": "real_consumption",  # real personal consumption expenditures
}
FRENCH_FILES = {
    "F-F_Research_Data_Factors": ("ff_factors_monthly", "monthly"),
    "F-F_Research_Data_Factors_daily": ("ff_factors_daily", "daily"),
    "F-F_Momentum_Factor": ("ff_momentum_monthly", "monthly"),
    "49_Industry_Portfolios": ("ff_industry49_monthly", "monthly"),
}


def fetch(url):
    """Return the raw bytes at url, with a user agent the hosts accept."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (course data refresh)"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fred_series(series):
    """One FRED series as a Series indexed by observation date."""
    raw = fetch(FRED_URL.format(series=series))
    frame = pd.read_csv(io.BytesIO(raw))
    frame.columns = [c.lower() for c in frame.columns]
    date_col = "observation_date" if "observation_date" in frame else "date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[series.lower()], errors="coerce")
    return pd.Series(values.values, index=frame[date_col], name=series)


def fred_panel(mapping):
    """Several FRED series joined on their dates and renamed."""
    panel = pd.concat([fred_series(s) for s in mapping], axis=1, sort=True)
    panel.columns = list(mapping.values())
    panel.index.name = "date"
    return panel.loc[START:].astype("float64")


def french_table(name, frequency):
    """The first data block of a Kenneth French zip file."""
    raw = fetch(FRENCH_URL.format(name=name))
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        text = archive.read(archive.namelist()[0]).decode("latin-1")

    width = 8 if frequency == "daily" else 6
    rows, header = [], None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None and len(parts) > 1 and parts[0] == "":
            header = [p.lower().replace("-", "_").replace(" ", "_") for p in parts[1:]]
            continue
        if header and parts[0][:width].isdigit() and len(parts[0]) == width:
            rows.append([parts[0]] + parts[1:])
        elif header and rows:
            break                       # the annual block starts after a blank line

    frame = pd.DataFrame(rows, columns=["date"] + header)
    fmt = "%Y%m%d" if frequency == "daily" else "%Y%m"
    frame["date"] = pd.to_datetime(frame["date"], format=fmt)
    frame = frame.set_index("date").apply(pd.to_numeric, errors="coerce")
    missing = frame.isin([-99.99, -999.0])          # French missing-value codes
    frame = frame.mask(missing).dropna(how="all")
    return (frame / 100.0).astype("float64")   # percent to decimal returns


def main():
    DATA_DIR.mkdir(exist_ok=True)
    written = []

    for mapping, name in ((FRED_MONTHLY, "fred_macro_monthly"),
                          (FRED_QUARTERLY, "fred_macro_quarterly")):
        panel = fred_panel(mapping)
        panel.to_parquet(DATA_DIR / f"{name}.parquet")
        written.append((f"{name}.parquet", panel.shape, panel.index.min(), panel.index.max()))

    for source, (name, frequency) in FRENCH_FILES.items():
        table = french_table(source, frequency)
        table.to_parquet(DATA_DIR / f"{name}.parquet")
        written.append((f"{name}.parquet", table.shape, table.index.min(), table.index.max()))

    lines = [
        "# Data snapshots",
        "",
        f"Refreshed {date.today().isoformat()} by `tools/refresh_data.py`.",
        "",
        "Sources: Federal Reserve Bank of St. Louis (FRED) and the Kenneth R.",
        "French data library, Tuck School of Business at Dartmouth. Factor and",
        "portfolio returns are stored as decimals, not percent.",
        "",
        "| File | Rows | Columns | First | Last |",
        "|---|--:|--:|---|---|",
    ]
    for name, (rows, cols), first, last in written:
        lines.append(f"| {name} | {rows} | {cols} | {first:%Y-%m-%d} | {last:%Y-%m-%d} |")
    (DATA_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(written)} snapshots to {DATA_DIR}")
    for name, shape, first, last in written:
        print(f"  {name:34s} {shape[0]:6d} x {shape[1]:3d}   {first:%Y-%m} to {last:%Y-%m}")


if __name__ == "__main__":
    main()
