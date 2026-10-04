"""研究策略：基线 VWAP 偏离带 + 趋势/状态过滤（规则与参数网格事先登记，见 reports/preregistration.md）。

过滤器只改变「开仓」：判定为上涨趋势时，倒T 信号被跳过（action=skip）或翻转为顺势正T（action=flip）；
symmetric=True 时，判定为下跌趋势时正T 信号也被跳过。所有判定只用 [0..i] 的 K 线或开盘前已知的日线。

filter 取值（th 为阈值）：
  none         不过滤（等同基线）
  day_ret      当日涨幅 (close_i - 前收)/ATR5 >= th                       （盘中，ATR5 单位）
  vwap_frac    开盘以来收在 VWAP 之上的 K 线占比 >= th（至少 6 根后才生效）
  vwap_slope   VWAP 在最近 6 根 K 线（30 分钟）的变化 / ATR5 >= th
  orb          收盘价突破开盘前 th 根 K 线的最高价（开盘区间未走完前不判定）
  gap          开盘跳空 (open_0 - 前收)/ATR5 >= th                         （开盘即知，全天有效）
  daily_trend  昨收 > 昨日及之前 th 日均线（基于复权收益链，盘前已知，全天有效）
"""
from __future__ import annotations

import numpy as np

from ..engine import DayContext, Order
from . import register
from .vwap_band import VwapBand

FILTERS = ("none", "day_ret", "vwap_frac", "vwap_slope", "orb", "gap", "daily_trend")


@register
class VwapBandRegime(VwapBand):
    name = "vwap_band_regime"
    version = "1"
    defaults = dict(VwapBand.defaults, filter="none", th=0.0, symmetric=False, action="skip")

    def __init__(self, **params):
        super().__init__(**params)
        if self.p["filter"] not in FILTERS:
            raise ValueError(f"未知过滤器 {self.p['filter']}，可选 {FILTERS}")
        if self.p["action"] not in ("skip", "flip"):
            raise ValueError("action 只能是 skip / flip")

    def daily_features(self, md):
        f = super().daily_features(md)
        if self.p["filter"] == "daily_trend":
            L = int(self.p["th"])
            idx = (1 + md.days.ret / 100).cumprod()  # 复权价格指数：除权除息不产生假信号
            f["trend_up"] = (idx > idx.rolling(L).mean()).shift(1)
            f["trend_dn"] = (idx < idx.rolling(L).mean()).shift(1)
        return f

    def trend_flags(self, ctx: DayContext) -> tuple[bool, bool]:
        flt, th, atr, i = self.p["filter"], self.p["th"], self.atr, ctx.i
        if flt == "none":
            return False, False
        if flt == "day_ret":
            x = (ctx.close[-1] - ctx.preclose) / atr
            return x >= th, x <= -th
        if flt == "vwap_frac":
            if i < 5:
                return False, False
            above = float(np.mean(ctx.close > ctx.vwap))
            below = float(np.mean(ctx.close < ctx.vwap))
            return above >= th, below >= th
        if flt == "vwap_slope":
            vw = ctx.vwap
            x = (vw[-1] - vw[max(0, i - 6)]) / atr
            return x >= th, x <= -th
        if flt == "orb":
            m = int(th)
            if i < m:
                return False, False
            c = ctx.close[-1]
            return c > ctx.high[:m].max(), c < ctx.low[:m].min()
        if flt == "gap":
            x = (ctx.open[0] - ctx.preclose) / atr
            return x >= th, x <= -th
        if flt == "daily_trend":
            return _flag(ctx.features.get("trend_up")), _flag(ctx.features.get("trend_dn"))
        raise AssertionError(flt)

    def entry(self, ctx: DayContext, up_sig: bool, dn_sig: bool):
        if not (up_sig or dn_sig):
            return None
        sides, lot = self.p["sides"], self.p["lot"]
        up_tr, dn_tr = self.trend_flags(ctx)
        if up_sig and sides in ("both", "dao") and up_tr:
            return [Order("buy", lot, "open_flip")] if self.p["action"] == "flip" else None
        if dn_sig and sides in ("both", "zheng") and dn_tr and self.p["symmetric"]:
            return None
        return super().entry(ctx, up_sig, dn_sig)


def _flag(v) -> bool:
    return bool(v) if isinstance(v, (bool, np.bool_)) else False


def describe(p: dict) -> str:
    if p.get("filter", "none") == "none":
        return "无过滤(基线)"
    s = f"{p['filter']}≥{p['th']:g}"
    if p.get("symmetric"):
        s += " 对称"
    if p.get("action") == "flip":
        s += " 翻转"
    return s

