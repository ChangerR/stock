"""交易费用与滑点。

费率支持按日期分段（印花税 2023-08-28 由 0.1% 降为 0.05%；过户费 2022-04-29 由 0.002% 降为 0.001%）。
配置里写一个数字表示全期固定费率，写列表 [[起始日, 费率], ...] 表示分段费率。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Union

from .market import TICK, round_price

Rate = Union[float, list]

HISTORICAL_STAMP_DUTY = [["2008-09-19", 0.001], ["2023-08-28", 0.0005]]
HISTORICAL_TRANSFER = [["2015-08-01", 0.00002], ["2022-04-29", 0.00001]]


class DatedRate:
    def __init__(self, spec: Rate):
        if isinstance(spec, (int, float)):
            self.dates, self.rates = [""], [float(spec)]
        else:
            pairs = sorted((str(d), float(r)) for d, r in spec)
            self.dates, self.rates = [d for d, _ in pairs], [r for _, r in pairs]

    def at(self, date: str) -> float:
        i = bisect.bisect_right(self.dates, date) - 1
        if i < 0:
            raise ValueError(f"{date} 早于费率表起始日 {self.dates[0]}")
        return self.rates[i]


@dataclass
class FeeSchedule:
    commission: float = 0.00025      # 佣金（双向）
    commission_min: float = 5.0      # 单笔最低佣金
    stamp_duty: Rate = 0.0005        # 印花税（仅卖出）
    transfer: Rate = 0.00001         # 过户费（双向）
    _stamp: DatedRate = field(init=False, repr=False)
    _transfer: DatedRate = field(init=False, repr=False)

    def __post_init__(self):
        if self.stamp_duty == "historical":
            self.stamp_duty = HISTORICAL_STAMP_DUTY
        if self.transfer == "historical":
            self.transfer = HISTORICAL_TRANSFER
        self._stamp = DatedRate(self.stamp_duty)
        self._transfer = DatedRate(self.transfer)

    def cost(self, side: str, price: float, qty: int, date: str = "9999-12-31") -> float:
        amt = price * qty
        f = max(self.commission_min, self.commission * amt) + self._transfer.at(date) * amt
        if side == "sell":
            f += self._stamp.at(date) * amt
        return f

    def round_trip_cost(self, price: float, qty: int, date: str = "9999-12-31") -> float:
        return self.cost("buy", price, qty, date) + self.cost("sell", price, qty, date)


@dataclass
class Slippage:
    """对自己不利方向偏移 max(min_ticks 个 tick, 价格×pct)。pct=0 且 min_ticks=0 时无滑点。"""
    pct: float = 0.0002
    min_ticks: int = 1

    def delta(self, price: float) -> float:
        if self.pct <= 0 and self.min_ticks <= 0:
            return 0.0
        if self.pct <= 0:
            return self.min_ticks * TICK
        return max(self.min_ticks * TICK, round_price(price * self.pct))

    def apply(self, price: float, side: str) -> float:
        d = self.delta(price)
        return round_price(price + d if side == "buy" else price - d)

    def as_fraction(self, price: float) -> float:
        return self.delta(price) / price
