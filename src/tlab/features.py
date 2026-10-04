"""常用日级特征。每个函数返回按日期索引的 Series，第 t 行只用 t 日开盘前已知的数据（均已 shift）。

价格锚统一表示成「相对前收盘的比例」，乘以当日前收即得当日价格口径下的锚点。
这样除权除息日不会产生假信号（交易所公布的前收盘价已经除权）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_atr_pct(days: pd.DataFrame, n: int = 14) -> pd.Series:
    """日线 ATR(n) / 收盘价（用前收计算 TR，对除权不敏感）。"""
    tr = np.maximum(days.high, days.preclose) - np.minimum(days.low, days.preclose)
    return (tr / days.preclose).rolling(n).mean().shift(1)


def adj_index(days: pd.DataFrame) -> pd.Series:
    return (1 + days.ret / 100).cumprod()


def ma_anchor_ratio(days: pd.DataFrame, n: int) -> pd.Series:
    """过去 n 日（含昨日）复权收盘均线 / 昨日复权收盘。当日锚点 = 该比例 × 当日前收。"""
    idx = adj_index(days)
    return (idx.rolling(n).mean() / idx).shift(1)


def prev_vwap_ratio(days: pd.DataFrame) -> pd.Series:
    """昨日全天 VWAP / 昨日收盘。当日锚点 = 该比例 × 当日前收。"""
    vw = days.amount / days.volume.replace(0, np.nan)
    return (vw / days.close).shift(1)
