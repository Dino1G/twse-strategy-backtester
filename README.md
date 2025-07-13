**TWSE Strategy Backtester**

這是一個用於台灣股票市場（日頻資料）策略開發與回測的 Python 專案，涵蓋資料爬取、特徵工程、因子選擇、模型訓練與走勢回測。

---

## 專案結構

```
.
├── algos/                  # 因子選擇演算法（GA、SA）
│   ├── base_selector.py
│   ├── ga_selector.py
│   └── sa_selector.py
├── data/
│   ├── raw/               # 原始 JSON → Parquet 檔案
│   └── twse_daily_2020_2024.parquet  # 完整儲存後的歷史行情
├── features/
│   └── builder.py         # 資料讀取、清洗、寬表/標籤建立
├── models/                # 預測訊號模型
│   ├── base_model.py
│   ├── random_forest.py
│   └── hgb_model.py
├── strats/
│   ├── atr_strategy.py    # ATR 停利停損策略
│   └── pct_strategy.py    # 固定百分比停利停損策略
├── utils/
│   ├── fetch.py           # TWSE 資料爬取
│   └── io.py              # 讀寫輔助
├── signals/
│   ├── fake_signals.csv   # 範例信號
│   ├── ga_signals.csv     # GA 信號結果
│   └── sa_signals.csv     # SA 信號結果
├── results/
│   └── walkforward_*.html # 回測報告
├── ga_signal_generator.py # 產生 GA 信號的主程式
├── sa_signal_generator.py # 產生 SA 信號的主程式
├── walkforward_backtester.py # 回測主程式
├── requirements.txt       # 相依套件清單
└── README.md              # 專案說明（此檔）
```

---

## 環境依賴

* Python 3.8
* 安裝相依：

  ```bash
  pip install -r requirements.txt
  ```

---

## 技術亮點

* **AutoML 特徵與模型搜尋**
  結合多種特徵與模型，以遺傳演算法 (GA) / 模擬退火 (SA) 自動尋找最佳特徵子集與模型組合，並生成交易訊號。
* **週期性滾動測試 (Walk-Forward Rolling)**
  以單週為測試集，其餘資料作為訓練；滾動 10 週並設置測試準確率門檻 (ACC > 0.5) 篩選模型，最後採多模型投票決策當週訊號。
* **高內聚低耦合模組化**
  `features`、`algos`、`models`、`strats` 各自獨立，易於擴充新的特徵工程、機器學習或深度學習模型及交易策略。
* **持續迭代策略**
  信號產出後結合 Walk-Forward 回測 (2 年 IS + 1 年 OOS) 並滾動前進，確保策略可隨市場變化動態更新。
* **強化學習可擴充**
  系統設計允許未來在 `algos` 中整合強化學習演算法，用於策略自動優化。

---

## 功能模組說明

1. **資料爬取** (`utils/fetch.py`)

   * 使用 TWSE API (`MI_INDEX?type=ALLBUT0999`) 逐日抓取「每日收盤行情(不含權證、牛熊證)」。
   * 輸出 parquet 檔於 `data/raw/`，並整合為 `data/twse_daily_2020_2024.parquet`。
2. **特徵工程** (`features/builder.py`)

   * **`load_and_clean(path)`**：讀取 Parquet，清洗（去 HTML 標籤、去逗號、轉數值）、剔除缺漏。
   * **`make_wide(df)`**：將長表轉寬表（各股票開高低收量欄位）。
   * **`add_label(wide, df, target)`**：以明日漲跌作為目標標籤 `y`，串接至寬表。
3. **因子選擇** (`algos/`)

   * **GASelector**：遺傳演算法選擇特徵子集，依滾動測試準確率評分。
   * **SASelector**：模擬退火搜尋最優特徵子集。
4. **模型封裝** (`models/`)

   * **RandomForestModel**, **HistGBModel**：Scikit-Learn 分類器包裝，統一 `fit` / `predict` / `predict_proba` 介面。
5. **信號產生**

   * **GA**：`python ga_signal_generator.py` → `signals/ga_signals.csv`
   * **SA**：`python sa_signal_generator.py` → `signals/sa_signals.csv`
6. **走勢回測** (`walkforward_backtester.py`)

   * 以過去兩年資料訓練，逐年 (2022–2024) 進行走勢回測。
   * 策略：ATR 策略 (`ATRStrategy`) vs 固定百分比策略 (`PctStrategy`)，依 In-sample 表現選擇最佳策略於 Out-of-sample 測試。
   * 結果報告輸出至 `results/walkforward_{year}_{strategy}.html`。

---

## 快速上手範例

```bash
# 1. 抓取並整合行情
python utils/fetch.py

# 2. 產生 GA/SA 信號（可同時執行）
python ga_signal_generator.py
python sa_signal_generator.py

# 3. 走勢回測並產生 HTML 報告
python walkforward_backtester.py
```

---

## 結果檔案

* `data/twse_daily_2020_2024.parquet`：完整歷史行情資料
* `signals/ga_signals.csv`, `signals/sa_signals.csv`：每日買賣信號
* `results/*.html`：各年度走勢回測報表
