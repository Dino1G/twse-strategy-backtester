import pandas as pd
import numpy as np


def load_and_clean(path):
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "證券代號": "symbol", "開盤價": "open", "最高價": "high",
        "最低價": "low", "收盤價": "close", "成交股數": "volume", "date": "date"
    })
    vars_ = ["open", "high", "low", "close", "volume"]
    for v in vars_:
        df[v] = (df[v].astype(str)
                      .str.replace(r"<.*?>", "", regex=True)
                      .str.replace(",", "")
                      .replace("--", np.nan))
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.dropna(subset=["symbol", "date"] + vars_)
    return df


def make_wide(df):
    vars_ = ["open", "high", "low", "close", "volume"]
    wide = df.pivot_table(index="date", columns="symbol",
                          values=vars_, aggfunc="first")
    wide.columns = [f"{sym}_{var}" for var, sym in wide.columns]
    return wide


def add_label(wide, df, target="2330"):
    tsmc = (df[df.symbol == target]
            .set_index("date").sort_index()[["close"]]
            .assign(fut=lambda d: d["close"].shift(-1))
            .assign(y=lambda d: (d["fut"] > d["close"]).astype(int))[["y"]])
    data = wide.join(tsmc, how="inner").dropna(subset=["y"])
    return data
