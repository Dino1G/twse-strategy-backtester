import os
import pandas as pd
from tqdm import tqdm

from features import load_and_clean, make_wide, add_label
from algos import GASelector
from models import RandomForestModel


# 參數設定
DATA_PATH = "data/twse_daily_2020_2024.parquet"
START_DATE = "2022-01-01"
END_DATE = "2022-01-05"
GA_POP_SIZE = 5
GA_NGEN = 1
DAYS_ROLL = 10

# 確保輸出資料夾存在
os.makedirs("signals", exist_ok=True)

# 載入 & 特徵構造
df = load_and_clean(DATA_PATH)
wide = make_wide(df)
data = add_label(wide, df, target="2330")

# 篩出日期區間
all_dates = data.index.sort_values()
dates = all_dates[(all_dates >= pd.to_datetime(START_DATE)) &
                  (all_dates <= pd.to_datetime(END_DATE))]

# 可用特徵
FEATURES = [c for c in data.columns if c != "y" and data[c].notna().all()]

# 準備 selector
selector = GASelector(pop_size=GA_POP_SIZE, ngen=GA_NGEN, days=DAYS_ROLL)

signals = []
# 逐日回測並顯示進度條
for predict_date in tqdm(dates, desc="GA backtest"):
    future = all_dates[all_dates > predict_date]
    if future.empty:
        continue
    next_day = future[0]

    # GA 選特徵 + 投票
    selected_sets = selector.fit(data, FEATURES, label="y")
    votes = []
    for feats in selected_sets:
        Xtr = data.loc[data.index < predict_date, feats]
        ytr = data.loc[data.index < predict_date, "y"]
        clf = RandomForestModel(n_estimators=50, random_state=42)
        clf.fit(Xtr, ytr)
        votes.append(clf.predict(data.loc[[next_day], feats])[0])

    signal = 1 if votes.count(1) > votes.count(0) else 0
    signals.append({
        "predict_date": predict_date,
        "signal_date":  next_day,
        "ga_signal":    signal
    })

# 存檔
df_out = pd.DataFrame(signals)
df_out.to_csv("signals/ga_signals.csv", index=False)
print(f"GA signals saved: {len(signals)} records → signals/ga_signals.csv")
