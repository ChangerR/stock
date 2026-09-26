"""合成行情与脚本化策略，用于精确检验引擎记账。"""
from __future__ import annotations

import pandas as pd

from tlab.data.store import prepare
from tlab.engine import Order, Strategy

TIMES = [int(f"{h:02d}{m:02d}") for h, m in
         [(9, 35), (9, 40), (9, 45), (9, 50), (9, 55)] + [(h, m) for h in (10,) for m in range(0, 60, 5)]
         + [(11, m) for m in range(0, 35, 5)] + [(13, m) for m in range(5, 60, 5)]
         + [(14, m) for m in range(0, 60, 5)] + [(15, 0)]]
assert len(TIMES) == 48


def make_md(code: str, days: list[dict]) -> "MarketData":
    """days: [{date, preclose, bars: [(o,h,l,c), ...] 或 flat 价格, isST}]，不足 48 根用最后价补齐。"""
    mrows, drows = [], []
    for d in days:
        bars = d["bars"]
        if not isinstance(bars, list):
            bars = [(bars, bars, bars, bars)]
        bars = list(bars) + [(bars[-1][3],) * 4] * (48 - len(bars))
        for t, (o, h, l, c) in zip(TIMES, bars):
            mrows.append(dict(date=d["date"], time=f"{d['date'].replace('-', '')}{t:04d}00000", open=o, high=h,
                              low=l, close=c, volume=1000, amount=1000 * c))
        close = d.get("close", bars[-1][3])
        drows.append(dict(date=d["date"], open=bars[0][0], high=max(b[1] for b in bars),
                          low=min(b[2] for b in bars), close=close, preclose=d["preclose"], volume=48000,
                          amount=0, pctChg=(close / d["preclose"] - 1) * 100, tradestatus=1,
                          isST=d.get("isST", 0)))
    return prepare(code, pd.DataFrame(mrows), pd.DataFrame(drows))


class Scripted(Strategy):
    """按 {(date, hhmm): [Order, ...]} 在指定 K 线收盘时下单；记录回调，便于断言。"""

    name = "scripted"
    version = "test"
    defaults = {"script": {}, "preopen": {}, "max_hold_days": 0}

    def __init__(self, **p):
        super().__init__(**p)
        self.fills, self.rejects, self.seen_len = [], [], []

    def on_day_start(self, ctx):
        return [Order(*o) for o in self.p["preopen"].get(ctx.date, [])]

    def on_bar(self, ctx):
        self.seen_len.append((ctx.i, len(ctx.close)))
        return [Order(*o) for o in self.p["script"].get((ctx.date, ctx.hhmm), [])]

    def on_fill(self, fill, ctx):
        self.fills.append(fill)

    def on_reject(self, order, reason, ctx):
        self.rejects.append((order.tag, reason))
