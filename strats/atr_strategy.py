from backtesting import Strategy
import pandas as pd


class ATRStrategy(Strategy):
    atr_period = 14
    sl_multiplier = 1.5
    tp_multiplier = 3.0

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        self.atr = self.I(self._ATR, high, low, close, self.atr_period)
        self.signal = self.data.Signal

    @staticmethod
    def _ATR(high, low, close, n):
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = df[['high', 'prev_close']].max(
            axis=1) - df[['low', 'prev_close']].min(axis=1)
        return df['tr'].rolling(n).mean().to_numpy()

    def next(self):
        s = int(self.signal[-1])
        price = self.data.Close[-1]
        atr = self.atr[-1]
        if s == 1 and not self.position:
            sl = price - self.sl_multiplier * atr
            tp = price + self.tp_multiplier * atr
            self.buy(sl=sl, tp=tp)
        elif s == 0 and not self.position:
            sl = price + self.sl_multiplier * atr
            tp = price - self.tp_multiplier * atr
            self.sell(sl=sl, tp=tp)
