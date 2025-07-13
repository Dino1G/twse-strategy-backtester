import os
import pandas as pd
from backtesting import Backtest

from features import load_and_clean
from strats import ATRStrategy, PctStrategy

SIGNALS_PATH = "signals/fake_signals.csv"
PRICE_PATH   = "data/twse_daily_2020_2024.parquet"
OUTPUT_DIR   = "results"
TRAIN_YEARS  = 2
START_YEAR   = 2020
END_YEAR     = 2024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 載入 & 清洗價格資料
raw = load_and_clean(PRICE_PATH)
price = (
    raw[raw['symbol']=='2330']
       .rename(columns={
           'open':'Open','high':'High',
           'low':'Low','close':'Close',
           'volume':'Volume','date':'Date'
       })
       .assign(Date=lambda df: pd.to_datetime(df['Date']))
       .set_index('Date')[['Open','High','Low','Close','Volume']]
       .sort_index()
)

# 2. 載入並合併訊號
signals = pd.read_csv(SIGNALS_PATH, parse_dates=['signal_date'])
signal_col = [c for c in signals.columns if c.endswith('_signal')][0]
sig = (
    signals.set_index('signal_date')
           .rename_axis('Date')
           [[signal_col]]
           .rename(columns={signal_col: 'Signal'})
)
# 合併，空訊號=0
data = price.join(sig, how='left').fillna(0)

# 3. Walk-Forward 回測，按年生成權益曲線
for year in range(START_YEAR + TRAIN_YEARS, END_YEAR + 1):
    train_start = f"{year-TRAIN_YEARS}-01-01"
    train_end   = f"{year-1}-12-31"
    test_start  = f"{year}-01-01"
    test_end    = f"{year}-12-31"

    train_data = data.loc[train_start:train_end]
    test_data  = data.loc[test_start:test_end]

    print(f"=== WalkForward Year {year} ===")
    print(f"Train: {train_start} to {train_end}, Test: {test_start} to {test_end}")

    # In-sample 比較
    bt_atr = Backtest(train_data, ATRStrategy, cash=100000, commission=0.001, trade_on_close=False)
    res_atr = bt_atr.run()
    bt_pct = Backtest(train_data, PctStrategy, cash=100000, commission=0.001, trade_on_close=False)
    res_pct = bt_pct.run()
    print(f"  ATR  IS Return%={res_atr['Return [%]']:.2f}, WinRate={res_atr['Win Rate [%]']:.2f}")
    print(f"  PCT  IS Return%={res_pct['Return [%]']:.2f}, WinRate={res_pct['Win Rate [%]']:.2f}")

    # 策略選擇
    if res_atr['Return [%]'] >= res_pct['Return [%]']:
        chosen, strat = 'ATR', ATRStrategy
    else:
        chosen, strat = 'PCT', PctStrategy
    print(f"Chosen for {year}: {chosen}")

    # OOS 回測並生成 HTML
    bt_oos = Backtest(test_data, strat, cash=100000, commission=0.001, trade_on_close=False)
    res_oos = bt_oos.run()
    print(f"  {year} OOS Return%={res_oos['Return [%]']:.2f}, WinRate={res_oos['Win Rate [%]']:.2f}")
    out_file = os.path.join(OUTPUT_DIR, f"walkforward_{year}_{chosen}.html")
    bt_oos.plot(filename=out_file)
    print(f"  Equity curve: {out_file}")
