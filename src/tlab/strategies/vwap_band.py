"""基线策略：自适应 VWAP 偏离带 双向做T（原型 backtest.py 的移植）。

  - 锚：当日累计 VWAP；带宽 = k × ATR5，ATR5 = 前 5 个交易日全部 5 分钟 K 线 TR 的均值（盘前已知）
  - 倒T：收盘 >= VWAP + 带宽 → 卖出 lot 股底仓；收盘 <= VWAP → 买回
  - 正T：收盘 <= VWAP - 带宽 → 买入 lot 股；收盘 >= VWAP → 卖出底仓
  - 止损：收盘相对成交价逆向 >= s × ATR5 → 平仓，且当日停止（halt_on_stop）
  - 每日最多 max_trips 个闭环；仅 entry_start..entry_end 的 K 线可开仓；距涨跌停 0.5% 内不开仓
  - 费用门槛：k×ATR5/开盘价 < gate × 往返保本价差 → 当日不做
  - 14:50 强平由引擎负责
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import DayContext, Fill, Order, Strategy
from ..fees import FeeSchedule, Slippage
from . import register


def atr_intraday(bars: pd.DataFrame, days: int) -> pd.Series:
    """前 `days` 个交易日全部 K 线真实波幅均值，按日期索引，已 shift(1)：第 t 行只用 t 之前的数据。
    TR 跨日连续计算（首根 K 线的 TR 含隔夜跳空），与原型一致。"""
    pc = bars.close.shift(1)
    tr = np.maximum(bars.high, pc.fillna(bars.high)) - np.minimum(bars.low, pc.fillna(bars.low))
    g = tr.groupby(bars.date).agg(["sum", "count"])
    return (g["sum"].rolling(days).sum() / g["count"].rolling(days).sum()).shift(1)


@register
class VwapBand(Strategy):
    name = "vwap_band"
    version = "1"
    defaults = dict(k=4.0, s=1.5, lot=200, max_trips=2, gate=3.0, atr_days=5,
                    entry_start=945, entry_end=1430, near_limit=0.005, sides="both", halt_on_stop=True)

    def __init__(self, **params):
        super().__init__(**params)
        self.bind_costs(FeeSchedule(), Slippage())

    def bind_costs(self, fees: FeeSchedule, slippage: Slippage):
        """费用门槛需要费率与滑点；runner 在回测前注入与引擎相同的配置。"""
        self._fees, self._slip = fees, slippage

    def daily_features(self, md):
        return pd.DataFrame({"atr": atr_intraday(md.bars, self.p["atr_days"])})

    def breakeven_pct(self, px: float, date: str) -> float:
        lot, sl = self.p["lot"], self._slip
        slip = 0.0 if sl.pct <= 0 else max(sl.pct, sl.min_ticks * 0.01 / px)
        return self._fees.round_trip_cost(px, lot, date) / (px * lot) + 2 * slip

    # ---- 生命周期 ----
    def on_day_start(self, day: DayContext):
        self.leg = None
        self.trips = 0
        self.halted = False
        self.gate_ok = None
        self.atr = day.features.get("atr", np.nan)

    def gate(self, ctx: DayContext) -> bool:
        if self.gate_ok is None:
            ref = ctx.open[0]
            self.gate_ok = bool(np.isfinite(self.atr)) and \
                (self.p["k"] * self.atr / ref) >= self.p["gate"] * self.breakeven_pct(ref, ctx.date)
        return self.gate_ok

    def on_bar(self, ctx: DayContext):
        if not np.isfinite(self.atr) or ctx.pending:
            return None
        p, atr = self.p, self.atr
        c, vw, t = ctx.close[-1], ctx.vwap[-1], ctx.hhmm
        leg = self.leg
        if leg is not None:
            if leg["dir"] == "倒T":
                adverse = c - leg["px"]
                if adverse >= p["s"] * atr:
                    return [Order("buy", leg["qty"], "STOP")]
                if c <= vw:
                    return [Order("buy", leg["qty"], "TP")]
            else:
                adverse = leg["px"] - c
                if adverse >= p["s"] * atr:
                    return [Order("sell", leg["qty"], "STOP")]
                if leg.get("exit_below_vwap"):
                    if c < vw:
                        return [Order("sell", leg["qty"], "EXIT")]
                elif c >= vw:
                    return [Order("sell", leg["qty"], "TP")]
            return None
        if ctx.pos != 0 or self.halted or self.trips >= p["max_trips"]:
            return None
        if not (p["entry_start"] <= t <= p["entry_end"]) or not self.gate(ctx):
            return None
        lot = p["lot"]
        near_up = c >= ctx.lim_up * (1 - p["near_limit"])
        near_dn = c <= ctx.lim_dn * (1 + p["near_limit"])
        up_sig = c >= vw + p["k"] * atr
        dn_sig = c <= vw - p["k"] * atr
        if ctx.sellable < lot:
            return None
        return self.entry(ctx, up_sig and not near_up, dn_sig and not near_dn)

    def entry(self, ctx: DayContext, up_sig: bool, dn_sig: bool):
        sides = self.p["sides"]
        lot = self.p["lot"]
        if sides in ("both", "dao") and up_sig:
            return [Order("sell", lot, "open_dao")]
        if sides in ("both", "zheng") and dn_sig:
            return [Order("buy", lot, "open_zheng")]
        return None

    def on_fill(self, fill: Fill, ctx: DayContext):
        if fill.tag == "open_dao":
            self.leg = dict(dir="倒T", px=fill.price, qty=fill.qty)
        elif fill.tag == "open_zheng":
            self.leg = dict(dir="正T", px=fill.price, qty=fill.qty)
        elif fill.tag == "open_flip":
            self.leg = dict(dir="正T", px=fill.price, qty=fill.qty, exit_below_vwap=True)
        elif self.leg is not None and ctx.pos == 0:
            self.leg = None
            self.trips += 1
            if fill.tag == "STOP" and self.p["halt_on_stop"]:
                self.halted = True
