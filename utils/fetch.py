import os, time, random, requests, pandas as pd
from datetime import datetime
from typing import Optional
from joblib import Parallel, delayed
from tqdm import tqdm

BASE = (
    "https://www.twse.com.tw/exchangeReport/MI_INDEX"
    "?response=json&date={}&type=ALLBUT0999"
)
CACHE_DIR = "data/raw"; os.makedirs(CACHE_DIR, exist_ok=True)

def _parse_one(yyyymmdd: str) -> Optional[pd.DataFrame]:
    time.sleep(random.uniform(3, 8))

    url = BASE.format(yyyymmdd)
    try:
        res = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"[WARN] {yyyymmdd} request/JSON error: {e}")
        return None

    tables = res.get("tables")
    if not tables:
        msg = res.get("stat", "no tables")
        print(f"[INFO] {yyyymmdd} skip – {msg}")
        return None

    target = next((t for t in tables if "每日收盤行情" in t["title"]), None)
    if target is None:
        print(f"[INFO] {yyyymmdd} no 收盤 table，跳過")
        return None

    df = pd.DataFrame(target["data"], columns=target["fields"])
    df["date"] = pd.to_datetime(yyyymmdd)
    return df


def _fetch_one(yyyymmdd: str, overwrite: bool = False) -> Optional[pd.DataFrame]:
    fp = f"{CACHE_DIR}/{yyyymmdd}.parquet"
    if os.path.exists(fp) and not overwrite:
        return pd.read_parquet(fp)

    df = _parse_one(yyyymmdd)
    if df is not None:
        df.to_parquet(fp, index=False)
    return df

def fetch_range(start="2020-01-01", end="2024-12-31", n_jobs=1) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    dfs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fetch_one)(d.strftime("%Y%m%d")) for d in tqdm(dates)
    )
    return pd.concat([d for d in dfs if d is not None], ignore_index=True)

if __name__ == "__main__":
    full = fetch_range()
    full.to_parquet("data/twse_daily_2020_2024.parquet", index=False)
    print(f"Done! rows = {len(full):,}")
