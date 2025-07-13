from backtesting import Strategy


class PctStrategy(Strategy):
    sl_pct = 0.02
    tp_pct = 0.04

    def init(self):
        self.signal = self.data.Signal

    def next(self):
        s = int(self.signal[-1])
        price = self.data.Close[-1]
        if s == 1 and not self.position:
            sl = price * (1 - self.sl_pct)
            tp = price * (1 + self.tp_pct)
            self.buy(sl=sl, tp=tp)
        elif s == 0 and not self.position:
            sl = price * (1 + self.sl_pct)
            tp = price * (1 - self.tp_pct)
            self.sell(sl=sl, tp=tp)
