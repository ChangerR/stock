"""信号事件研究：偏离带被触发之后，价格究竟是回归 VWAP 还是继续延续？

对每个交易日，取开仓窗口内第一次「收盘 >= VWAP + k×ATR5」（上轨事件）和第一次「收盘 <= VWAP - k×ATR5」（下轨事件），
以下一根开盘价为入场价，统计到尾盘（force_flat 下一根开盘）的前瞻收益，以及先回到 VWAP 还是先触及 s×ATR5 止损。
不含费用与滑点，用来判断「信号本身」有没有方向性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.store import MarketData
from .strategies.vwap_band import atr_intraday


def band_events(md: MarketData, dates: list[str], k: float = 4.0, s: float = 1.5, atr_days: int = 5,
                entry=(945, 1430), eod: int = 1450) -> pd.DataFrame:
    atr = atr_intraday(md.bars, atr_days)
    rows = []
    for date, g in md.bars[md.bars.date.isin(set(dates))].groupby("date", sort=True):
        a = atr.get(date, np.nan)
        if not np.isfinite(a):
            continue
        o, c, t = g.open.to_numpy(), g.close.to_numpy(), g.hhmm.to_numpy()
        vw = np.cumsum(g.amount.to_numpy()) / np.cumsum(g.volume.to_numpy())
        eod_i = int(np.searchsorted(t, eod)) + 1
        if eod_i >= len(t):
            continue
        for side, cond in (("上轨", c >= vw + k * a), ("下轨", c <= vw - k * a)):
            win = np.flatnonzero(cond & (t >= entry[0]) & (t <= entry[1]))
            if not len(win) or win[0] + 1 >= eod_i:
                continue
            i = int(win[0])
            px = o[i + 1]
            sign = 1 if side == "上轨" else -1          # 正数 = 顺着触发方向继续走
            path = c[i + 1:eod_i]
            vpath = vw[i + 1:eod_i]
            revert = np.flatnonzero(sign * (path - vpath) <= 0)
            stop = np.flatnonzero(sign * (path - px) >= s * a)
            first_rev = revert[0] if len(revert) else 10 ** 6
            first_stop = stop[0] if len(stop) else 10 ** 6
            rows.append(dict(date=date, side=side, sig_t=int(t[i]), entry=px, atr=a,
                             fwd_eod_bps=sign * (o[eod_i] / px - 1) * 1e4,
                             fwd_eod_atr=sign * (o[eod_i] - px) / a,
                             first=("回到VWAP" if first_rev < first_stop else
                                    "先触止损" if first_stop < first_rev else "都未发生"),
                             day_ret=float(md.days.loc[date, "ret"])))
    return pd.DataFrame(rows)


def summarize_events(ev: pd.DataFrame) -> pd.DataFrame:
    out = []
    for side, g in ev.groupby("side"):
        x = g.fwd_eod_bps
        out.append({"事件": side, "次数": len(g),
                    "至尾盘顺势收益均值(bp)": round(x.mean(), 1),
                    "t值": round(x.mean() / x.std(ddof=1) * np.sqrt(len(x)), 2) if len(x) > 2 else np.nan,
                    "顺势比例": f"{(x > 0).mean():.0%}",
                    "先回到VWAP": f"{(g['first'] == '回到VWAP').mean():.0%}",
                    "先触止损": f"{(g['first'] == '先触止损').mean():.0%}",
                    "都未发生": f"{(g['first'] == '都未发生').mean():.0%}"})
    return pd.DataFrame(out)
