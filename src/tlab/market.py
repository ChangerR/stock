"""A 股板块规则：涨跌幅限制、最小申报数量、价格最小变动单位。

只覆盖沪深 A 股（主板 / 创业板 / 科创板）。北交所等未支持，遇到会直接报错。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TICK = 0.01
# 创业板注册制改革：2020-08-24 起涨跌幅由 10% 改为 20%
CHINEXT_20PCT_FROM = "2020-08-24"


@dataclass(frozen=True)
class BoardRules:
    board: str          # "main" / "chinext" / "star"
    limit_pct: float    # 当日涨跌幅限制
    min_buy: int        # 单笔买入最小股数
    buy_step: int       # 超过最小数量后的递增单位
    sell_step: int      # 卖出递增单位（余股可一次性卖出）
    min_sell: int

    def valid_buy_qty(self, qty: int) -> bool:
        return qty >= self.min_buy and (qty - self.min_buy) % self.buy_step == 0

    def valid_sell_qty(self, qty: int, position: int) -> bool:
        if qty <= 0 or qty > position:
            return False
        if qty == position:  # 余股必须一次性卖出，全部卖出总是允许
            return True
        return qty >= self.min_sell and (qty - self.min_sell) % self.sell_step == 0


def board_of(code: str) -> str:
    """code 形如 'sh.688981' / 'sz.300059' / 'sh.600000'。"""
    ex, num = code.split(".")
    if ex == "sh" and num.startswith(("688", "689")):
        return "star"
    if ex == "sz" and num.startswith(("300", "301")):
        return "chinext"
    if (ex == "sh" and num.startswith(("600", "601", "603", "605"))) or (
        ex == "sz" and num.startswith(("000", "001", "002", "003"))
    ):
        return "main"
    raise ValueError(f"不支持的证券代码/板块: {code}")


def rules_for(code: str, date: str, is_st: bool = False) -> BoardRules:
    board = board_of(code)
    if board == "star":
        # 科创板：买入 >=200 股、以 1 股递增；卖出同样 >=200 股（余股一次卖出）。ST 仍为 20%
        return BoardRules("star", 0.20, 200, 1, 1, 200)
    if board == "chinext":
        pct = 0.20 if date >= CHINEXT_20PCT_FROM else (0.05 if is_st else 0.10)
        return BoardRules("chinext", pct, 100, 100, 100, 100)
    return BoardRules("main", 0.05 if is_st else 0.10, 100, 100, 100, 100)


def round_price(x: float) -> float:
    """交易所口径的四舍五入到分（避免浮点 x.xx5 被银行家舍入）。"""
    return math.floor(x * 100 + 0.5 + 1e-9) / 100


def limit_prices(preclose: float, limit_pct: float) -> tuple[float, float]:
    return round_price(preclose * (1 + limit_pct)), round_price(preclose * (1 - limit_pct))
