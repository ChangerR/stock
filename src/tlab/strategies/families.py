"""策略族普查（reports/survey/preregistration.md）用到的策略。所有策略都只交易 lot 股、围绕底仓操作，
T+1、申报数量、涨跌停、强平/持有期由引擎负责。

  grid           网格：日内（锚 = 开盘价 / 实时 VWAP）或多日（锚 = 10 日均线），间距按百分比 / ATR5 / 日线 ATR
  orb            开盘区间突破（顺势）：突破上沿做正T，跌破下沿做倒T，回到区间中线止损，尾盘平
  gap            跳空：fade（反向）/ follow（顺势），阈值按日线 ATR% 标准化
  xday_mr        跨日均值回归：14:45 偏离均线 / 昨日 VWAP 超过 θ×日线 ATR 时反向开仓，回到锚点止盈、2×日线 ATR 止损、N 日时间止损
  tod            固定时刻：定时开仓/平仓；或「首半小时收益 → 尾盘半小时」日内动量
  vwap_band_eod  VWAP 偏离带出场改造：不以回到 VWAP 止盈，持有到尾盘，只设宽止损
  overnight      隔夜/日内收益分解：14:50 开仓，次日开盘（或 10:00）平仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import DayContext, Fill, Order, Strategy
from ..features import daily_atr_pct, ma_anchor_ratio, prev_vwap_ratio
from . import register
from .vwap_band import VwapBand, atr_intraday


class FamilyBase(Strategy):
    """共用：日级特征、单腿状态、涨跌停附近不开仓。"""

    def daily_features(self, md):
        d = md.days
        return pd.DataFrame({
            "atr5": atr_intraday(md.bars, 5),
            "datr": daily_atr_pct(d, 14),
            "ma10": ma_anchor_ratio(d, 10),
            "ma20": ma_anchor_ratio(d, 20),
            "pvwap": prev_vwap_ratio(d),
        })

    def on_start(self):
        self.leg = None

    def on_day_start(self, day: DayContext):
        self.f = day.features
        self.trades_today = 0
        if day.pos == 0:
            self.leg = None
        return None

    def ok(self, *keys) -> bool:
        return all(np.isfinite(self.f.get(k, np.nan)) for k in keys)

    def near_limit(self, ctx: DayContext, c: float, pct: float = 0.005) -> bool:
        return c >= ctx.lim_up * (1 - pct) or c <= ctx.lim_dn * (1 + pct)

    def open_order(self, ctx: DayContext, side: str, tag: str = "open"):
        lot = self.p["lot"]
        if side == "sell" and ctx.sellable < lot:
            return None
        if side == "buy" and ctx.sellable < ctx.pos + lot:  # T+1：必须保证当日还能卖回底仓
            return None
        return [Order(side, lot, tag)]

    def close_order(self, ctx: DayContext, tag: str):
        pos = ctx.pos
        if pos > 0:
            return [Order("sell", min(pos, ctx.sellable), tag)] if ctx.sellable > 0 else None
        if pos < 0:
            return [Order("buy", -pos, tag)]
        return None

    def on_fill(self, fill: Fill, ctx: DayContext):
        if ctx.pos == 0:
            self.leg = None
        elif fill.tag.startswith("open"):
            self.leg = dict(dir=1 if fill.side == "buy" else -1, px=fill.price, date=ctx.date,
                            datr=self.f.get("datr", np.nan), mid=getattr(self, "_mid", np.nan))
            self.trades_today += 1


@register
class GridT(FamilyBase):
    name, version = "grid", "1"
    defaults = dict(lot=200, mode="pct", spacing=0.01, anchor="open", layers=2, max_hold_days=0,
                    entry_start=935, entry_end=1430)

    def on_bar(self, ctx: DayContext):
        p, c = self.p, ctx.close[-1]
        if not self.ok("atr5", "datr", "ma10"):
            return None
        A = {"open": ctx.open[0], "vwap": ctx.vwap[-1], "ma10": self.f["ma10"] * ctx.preclose}[p["anchor"]]
        sp = {"pct": p["spacing"] * A, "atr": p["spacing"] * self.f["atr5"],
              "datr": p["spacing"] * self.f["datr"] * ctx.preclose}[p["mode"]]
        lvl = int((c - A) / sp)  # 向 0 取整：偏离每超过一个间距，目标仓位反向一层
        target = -max(-p["layers"], min(p["layers"], lvl)) * p["lot"]
        pos, lot = ctx.pos, p["lot"]
        in_window = p["entry_start"] <= ctx.hhmm <= p["entry_end"]
        if target < pos:
            if (pos <= 0 and not in_window) or ctx.sellable < lot:
                return None
            if pos <= 0 and self.near_limit(ctx, c):
                return None
            return [Order("sell", lot, "open" if pos <= 0 else "grid")]
        if target > pos:
            if (pos >= 0 and not in_window) or (pos >= 0 and self.near_limit(ctx, c)):
                return None
            if pos >= 0 and p["max_hold_days"] == 0 and ctx.sellable < pos + lot:  # T+1：保证当日能卖回
                return None
            return [Order("buy", lot, "open" if pos >= 0 else "grid")]
        return None


@register
class OpeningRangeBreakout(FamilyBase):
    name, version = "orb", "1"
    defaults = dict(lot=200, n_or=6, buffer=0.0, sides="both", entry_end=1400)

    def on_bar(self, ctx: DayContext):
        p, i, c = self.p, ctx.i, ctx.close[-1]
        n = p["n_or"]
        if i < n or not self.ok("atr5"):
            return None
        hi, lo = ctx.high[:n].max(), ctx.low[:n].min()
        self._mid = (hi + lo) / 2
        if self.leg is not None:
            if (self.leg["dir"] > 0 and c < self.leg["mid"]) or (self.leg["dir"] < 0 and c > self.leg["mid"]):
                return self.close_order(ctx, "STOP")
            return None
        if ctx.pos != 0 or self.trades_today >= 1 or ctx.hhmm > p["entry_end"] or self.near_limit(ctx, c):
            return None
        b = p["buffer"] * self.f["atr5"]
        if c > hi + b:
            return self.open_order(ctx, "buy")
        if c < lo - b and p["sides"] == "both":
            return self.open_order(ctx, "sell")
        return None


@register
class GapTrade(FamilyBase):
    name, version = "gap", "1"
    defaults = dict(lot=200, theta=1.0, mode="fade")

    def on_bar(self, ctx: DayContext):
        p, c = self.p, ctx.close[-1]
        if not self.ok("datr"):
            return None
        o0, pc = ctx.open[0], ctx.preclose
        if self.leg is not None:
            d = self.leg["dir"]
            if p["mode"] == "fade":   # 反向：回补缺口止盈；再走一个缺口幅度止损
                g = o0 - pc
                if (d < 0 and c <= pc) or (d > 0 and c >= pc):
                    return self.close_order(ctx, "TP")
                if (d < 0 and c >= o0 + g) or (d > 0 and c <= o0 + g):
                    return self.close_order(ctx, "STOP")
            else:                     # 顺势：缺口完全回补则止损
                if (d > 0 and c <= pc) or (d < 0 and c >= pc):
                    return self.close_order(ctx, "STOP")
            return None
        if ctx.i != 0 or ctx.pos != 0 or self.near_limit(ctx, o0, 0.01):
            return None
        gap = (o0 / pc - 1) / self.f["datr"]
        if abs(gap) < p["theta"]:
            return None
        up = gap > 0
        buy = (not up) if p["mode"] == "fade" else up
        return self.open_order(ctx, "buy" if buy else "sell")


@register
class CrossDayMeanReversion(FamilyBase):
    name, version = "xday_mr", "1"
    defaults = dict(lot=200, anchor="ma20", theta=1.5, stop=2.0, max_hold_days=5, decide_at=1445, sides="both")

    def on_bar(self, ctx: DayContext):
        p, c = self.p, ctx.close[-1]
        if not self.ok("datr", p["anchor"]):
            return None
        A = self.f[p["anchor"]] * ctx.preclose
        if self.leg is not None:
            if self.leg["date"] == ctx.date:
                return None
            d, e, w = self.leg["dir"], self.leg["px"], self.leg["datr"] * self.leg["px"]
            if (d > 0 and c >= A) or (d < 0 and c <= A):
                return self.close_order(ctx, "TP")
            if (d > 0 and c <= e - p["stop"] * w) or (d < 0 and c >= e + p["stop"] * w):
                return self.close_order(ctx, "STOP")
            return None
        if ctx.hhmm != p["decide_at"] or ctx.pos != 0 or self.near_limit(ctx, c):
            return None
        dev = (c - A) / (self.f["datr"] * ctx.preclose)
        if dev <= -p["theta"]:
            return self.open_order(ctx, "buy")
        if dev >= p["theta"] and p["sides"] == "both":
            return self.open_order(ctx, "sell")
        return None


@register
class TimeOfDay(FamilyBase):
    name, version = "tod", "1"
    defaults = dict(lot=200, mode="fixed", side="dao", entry=1000, exit="EOD", theta=0.0, signal_t=1000,
                    decide_at=1425)

    def on_bar(self, ctx: DayContext):
        p, t, c = self.p, ctx.hhmm, ctx.close[-1]
        if self.leg is not None:
            if p["mode"] == "fixed" and p["exit"] != "EOD" and t == int(p["exit"]):
                return self.close_order(ctx, "EXIT")
            return None
        if ctx.pos != 0 or self.near_limit(ctx, c):
            return None
        if p["mode"] == "fixed":
            if t == p["entry"]:
                return self.open_order(ctx, "sell" if p["side"] == "dao" else "buy")
            return None
        # 日内动量：首半小时收益（前收 → 10:00）决定尾盘半小时方向
        if t != p["decide_at"] or not self.ok("datr"):
            return None
        k = int(np.searchsorted(ctx.times, p["signal_t"]))
        if k >= len(ctx.times) or ctx.times[k] != p["signal_t"]:
            return None
        r = ctx.close[k] / ctx.preclose - 1
        if r == 0 or abs(r) < p["theta"] * self.f["datr"]:
            return None
        return self.open_order(ctx, "buy" if r > 0 else "sell")


@register
class VwapBandEOD(VwapBand):
    """VWAP 偏离带 · 出场改造：带外反向开仓后不在 VWAP 止盈，持有到尾盘；只设 w×ATR5 宽止损（w=0 表示不设）。"""
    name, version = "vwap_band_eod", "1"
    defaults = dict(VwapBand.defaults, s=0.0, max_trips=1)

    def on_bar(self, ctx: DayContext):
        if not np.isfinite(self.atr) or ctx.pending:
            return None
        leg = self.leg
        if leg is not None:
            c = ctx.close[-1]
            adverse = (c - leg["px"]) if leg["dir"] == "倒T" else (leg["px"] - c)
            if self.p["s"] > 0 and adverse >= self.p["s"] * self.atr:
                return [Order("buy" if leg["dir"] == "倒T" else "sell", leg["qty"], "STOP")]
            return None
        return super().on_bar(ctx)


@register
class Overnight(FamilyBase):
    name, version = "overnight", "1"
    defaults = dict(lot=200, side="long", exit="open", entry_sig=1445, max_hold_days=1)

    def on_day_start(self, day: DayContext):
        super().on_day_start(day)
        if self.leg is not None and self.p["exit"] == "open":
            return self.close_order(day, "EXIT")
        return None

    def on_bar(self, ctx: DayContext):
        p = self.p
        if self.leg is not None:
            if p["exit"] != "open" and self.leg["date"] != ctx.date and ctx.hhmm == int(p["exit"]):
                return self.close_order(ctx, "EXIT")
            return None
        if ctx.hhmm == p["entry_sig"] and ctx.pos == 0 and not self.near_limit(ctx, ctx.close[-1]):
            return self.open_order(ctx, "buy" if p["side"] == "long" else "sell")
        return None
