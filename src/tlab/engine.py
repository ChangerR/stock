"""日内做T 回测引擎。

核心约定（所有策略共享，策略自己不需要、也不应该处理这些）：
  - 无未来函数：策略在第 i 根 K 线收盘时只能看到 [0..i] 的切片；指令默认在第 i+1 根开盘价成交（± 滑点）。
  - T+1：当日可卖额度 = 开盘时持股数，只减不增；当日买入的股份当日不可卖。
  - 板块规则：申报数量（主板/创业板 100 股整数倍；科创板 >=200 股），涨跌幅（主板 10%、ST 5%、创业板/科创板 20%）。
    整根 K 线封死涨停时买单无法成交、封死跌停时卖单无法成交；成交价被限制在涨跌停价之内。
  - 强制回到底仓：在 force_flat_time 这根 K 线收盘时，若持股 != 底仓，引擎自动下单回补/卖出（EOD）；
    成交失败（封板）会逐根重试，当日仍失败则持仓过夜，次日开盘继续回补（CARRY）。
  - 盈亏口径：相对「只持有底仓」的超额收益，逐日盯市
    pnl_t = Δcash_t + (收盘持股 - 底仓) × close_t - (开盘持股 - 底仓) × preclose_t
    （用前收盘而不是昨收，除权除息日不会产生虚假盈亏）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.store import MarketData
from .fees import FeeSchedule, Slippage
from .market import rules_for


@dataclass
class Order:
    side: str          # "buy" / "sell"
    qty: int
    tag: str = ""


@dataclass
class Fill:
    date: str
    sig_time: int      # 信号 K 线结束时刻（HHMM）
    fill_time: int     # 成交 K 线结束时刻（HHMM）
    side: str
    qty: int
    price: float
    fee: float
    tag: str


@dataclass
class EngineConfig:
    base_shares: int = 400
    force_flat_time: int = 1450
    fill: str = "next_open"          # "next_open"（默认）或 "close"（乐观：信号 K 线收盘价成交，仅作对比）
    enforce_limits: bool = True
    slippage: Slippage = field(default_factory=Slippage)
    fees: FeeSchedule = field(default_factory=FeeSchedule)


class DayContext:
    """某一交易日的静态信息 + 当前 K 线的因果切片。策略只应通过这里读取数据。"""

    __slots__ = ("date", "preclose", "lim_up", "lim_dn", "rules", "features", "n", "i", "hhmm",
                 "_o", "_h", "_l", "_c", "_v", "_a", "_vw", "_t", "shares", "base", "sellable",
                 "pending", "force_flat")

    def __init__(self, date, info, rules, features, arrays, base):
        self.date = date
        self.preclose = float(info["preclose"])
        self.lim_up = float(info["lim_up"])
        self.lim_dn = float(info["lim_dn"])
        self.rules = rules
        self.features = features
        self._o, self._h, self._l, self._c, self._v, self._a, self._vw, self._t = arrays
        self.n = len(self._c)
        self.i = -1
        self.hhmm = 0
        self.base = base
        self.shares = base
        self.sellable = base
        self.pending: list[Order] = []
        self.force_flat = False

    # 因果切片：只包含 [0..i]
    open = property(lambda s: s._o[: s.i + 1])
    high = property(lambda s: s._h[: s.i + 1])
    low = property(lambda s: s._l[: s.i + 1])
    close = property(lambda s: s._c[: s.i + 1])
    volume = property(lambda s: s._v[: s.i + 1])
    amount = property(lambda s: s._a[: s.i + 1])
    vwap = property(lambda s: s._vw[: s.i + 1])
    times = property(lambda s: s._t[: s.i + 1])

    @property
    def pos(self) -> int:
        """相对底仓的偏离：>0 表示多持有（正T 已买未卖），<0 表示倒T 已卖未买回。"""
        return self.shares - self.base


class Strategy:
    """策略基类。子类实现 on_bar，按需实现 daily_features / on_day_start / on_fill / on_reject。"""

    name = "base"
    version = "0"
    defaults: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: 未知参数 {sorted(unknown)}")
        self.p = {**self.defaults, **params}

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def daily_features(self, md: MarketData) -> pd.DataFrame:
        """按日期索引的日级特征。第 t 行只能用 t 日开盘前已知的信息（tests 中有截断检验）。"""
        return pd.DataFrame(index=md.days.index)

    def on_day_start(self, day: DayContext) -> None:
        pass

    def on_bar(self, ctx: DayContext) -> list[Order] | None:
        raise NotImplementedError

    def on_fill(self, fill: Fill, ctx: DayContext) -> None:
        pass

    def on_reject(self, order: Order, reason: str, ctx: DayContext) -> None:
        pass


@dataclass
class BacktestResult:
    trips: pd.DataFrame
    daily: pd.DataFrame
    fills: pd.DataFrame
    rejects: pd.DataFrame
    strategy_id: str
    params: dict
    code: str


TRIP_COLS = ["date", "entry_date", "dir", "entry_tag", "entry_sig_t", "entry_fill_t", "exit_sig_t",
             "exit_fill_t", "qty", "buy_px", "sell_px", "reason", "gross", "fees", "net", "overnight"]


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def run(self, md: MarketData, strategy: Strategy, dates: list[str] | None = None,
            features: pd.DataFrame | None = None) -> BacktestResult:
        cfg = self.cfg
        dates = list(md.days.index) if dates is None else list(dates)
        feats = strategy.daily_features(md) if features is None else features
        feat_rows = feats.to_dict("index") if len(feats.columns) else {}
        bars = md.bars
        day_slices = _day_slices(bars)
        O, H, L, C = (bars[c].to_numpy(float) for c in ("open", "high", "low", "close"))
        V, A = bars.volume.to_numpy(float), bars.amount.to_numpy(float)
        T = bars.hhmm.to_numpy(int)
        VW = _intraday_vwap(bars, day_slices)

        shares = cfg.base_shares
        cash = 0.0
        trip = None
        trips, fills, rejects, daily = [], [], [], []

        for date in dates:
            if date not in day_slices:
                continue
            a, b = day_slices[date]
            info = md.days.loc[date]
            rules = rules_for(md.code, date, bool(info.get("isST", 0)))
            ctx = DayContext(date, info, rules, feat_rows.get(date, {}),
                             (O[a:b], H[a:b], L[a:b], C[a:b], V[a:b], A[a:b], VW[a:b], T[a:b]),
                             cfg.base_shares)
            ctx.shares = shares
            ctx.sellable = shares
            sod_shares, sod_cash = shares, cash
            n = ctx.n
            carry = shares != cfg.base_shares  # 昨日未能回到底仓：开盘即回补，回补完成前不调用策略
            if carry:
                ctx.pending = [self._restore_order(ctx, "CARRY")]
            strategy.on_day_start(ctx)
            sig_t = {id(o): 0 for o in ctx.pending}
            day_fills = 0

            def do_fill(order, j, raw, st):
                nonlocal shares, cash, trip, day_fills
                reason = self._check(order, ctx, j, raw)
                if reason:
                    rejects.append(dict(date=date, sig_time=st, bar_time=int(ctx._t[j]), side=order.side,
                                        qty=order.qty, tag=order.tag, reason=reason))
                    strategy.on_reject(order, reason, ctx)
                    return False
                px = cfg.slippage.apply(raw, order.side)
                if cfg.enforce_limits:
                    px = min(max(px, ctx.lim_dn), ctx.lim_up)
                fee = cfg.fees.cost(order.side, px, order.qty, date)
                before = shares
                if order.side == "buy":
                    shares += order.qty
                    cash -= px * order.qty + fee
                else:
                    shares -= order.qty
                    ctx.sellable -= order.qty
                    cash += px * order.qty - fee
                ctx.shares = shares
                f = Fill(date, st, int(ctx._t[j]), order.side, order.qty, px, fee, order.tag)
                fills.append(f)
                day_fills += 1
                if trip is None and before == cfg.base_shares:
                    trip = dict(entry_date=date, entry_tag=order.tag, entry_sig_t=st, entry_fill_t=f.fill_time,
                                dir="倒T" if order.side == "sell" else "正T",
                                buy_amt=0.0, sell_amt=0.0, buy_qty=0, sell_qty=0, fees=0.0)
                if trip is not None:
                    k = "buy" if order.side == "buy" else "sell"
                    trip[f"{k}_amt"] += px * order.qty
                    trip[f"{k}_qty"] += order.qty
                    trip["fees"] += fee
                    if shares == cfg.base_shares:
                        gross = trip["sell_amt"] - trip["buy_amt"]
                        trips.append(dict(
                            date=date, entry_date=trip["entry_date"], dir=trip["dir"], entry_tag=trip["entry_tag"],
                            entry_sig_t=trip["entry_sig_t"], entry_fill_t=trip["entry_fill_t"], exit_sig_t=st,
                            exit_fill_t=f.fill_time, qty=trip["buy_qty"],
                            buy_px=trip["buy_amt"] / trip["buy_qty"], sell_px=trip["sell_amt"] / trip["sell_qty"],
                            reason=order.tag, gross=gross, fees=trip["fees"], net=gross - trip["fees"],
                            overnight=trip["entry_date"] != date))
                        trip = None
                strategy.on_fill(f, ctx)
                return True

            for i in range(n):
                # 1) 上一根收盘产生的指令，以本根开盘价成交
                if ctx.pending and cfg.fill == "next_open":
                    todo, ctx.pending = ctx.pending, []
                    for o in todo:
                        do_fill(o, i, O[a + i], sig_t.get(id(o), 0))
                if carry and shares == cfg.base_shares:
                    carry = False
                if i >= n - 1:
                    break
                ctx.i, ctx.hhmm = i, int(T[a + i])
                # 2) 本根收盘时生成新指令
                if ctx.hhmm >= cfg.force_flat_time:
                    ctx.force_flat = True
                    new = [self._restore_order(ctx, "EOD")] if shares != cfg.base_shares else []
                elif carry:
                    new = [self._restore_order(ctx, "CARRY")]
                else:
                    new = list(strategy.on_bar(ctx) or [])
                new = [o for o in new if o is not None and o.qty > 0]
                for o in new:
                    sig_t[id(o)] = ctx.hhmm
                if cfg.fill == "close":
                    for o in new:
                        do_fill(o, i, C[a + i], ctx.hhmm)
                else:
                    ctx.pending = new

            close_px = float(info["close"])  # 官方收盘价（与最后一根 5 分钟收盘可能因集合竞价口径略有差异）
            pnl = (cash - sod_cash) + (shares - cfg.base_shares) * close_px \
                - (sod_shares - cfg.base_shares) * ctx.preclose
            daily.append(dict(date=date, pnl=pnl, close=close_px, preclose=ctx.preclose, ret=float(info["ret"]),
                              bench=cfg.base_shares * (close_px - ctx.preclose), shares_eod=shares,
                              n_fills=day_fills))

        tr = pd.DataFrame(trips, columns=TRIP_COLS)
        dl = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame(
            columns=["pnl", "close", "preclose", "ret", "bench", "shares_eod", "n_fills"])
        dl["realized"] = tr.groupby("date").net.sum().reindex(dl.index).fillna(0.0) if len(tr) else 0.0
        dl["cum_pnl"] = dl.pnl.cumsum()
        dl["cum_bench"] = dl.bench.cumsum()
        fl = pd.DataFrame([f.__dict__ for f in fills])
        rj = pd.DataFrame(rejects, columns=["date", "sig_time", "bar_time", "side", "qty", "tag", "reason"])
        return BacktestResult(tr, dl, fl, rj, strategy.id, dict(strategy.p), md.code)

    def _restore_order(self, ctx: DayContext, tag: str) -> Order:
        diff = ctx.shares - ctx.base
        if diff > 0:
            return Order("sell", min(diff, ctx.sellable), tag)
        return Order("buy", -diff, tag)

    def _check(self, o: Order, ctx: DayContext, j: int, raw: float) -> str:
        if o.side == "sell":
            if o.qty > ctx.sellable:
                return "T+1: 超过当日可卖额度"
            if not ctx.rules.valid_sell_qty(o.qty, ctx.shares):
                return "申报数量不合规(卖)"
        elif o.side == "buy":
            if not ctx.rules.valid_buy_qty(o.qty):
                return "申报数量不合规(买)"
        else:
            return f"未知方向 {o.side}"
        if ctx.force_flat:
            after = ctx.shares + (o.qty if o.side == "buy" else -o.qty)
            if abs(after - ctx.base) > abs(ctx.shares - ctx.base):
                return "强平时段禁止扩大偏离"
        if self.cfg.enforce_limits:
            if o.side == "buy" and ctx._l[j] >= ctx.lim_up - 1e-9:
                return "涨停封板: 买单无法成交"
            if o.side == "sell" and ctx._h[j] <= ctx.lim_dn + 1e-9:
                return "跌停封板: 卖单无法成交"
        return ""


def _day_slices(bars: pd.DataFrame) -> dict[str, tuple[int, int]]:
    d = bars.date.to_numpy()
    idx = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    return {d[idx[k]]: (int(idx[k]), int(idx[k + 1])) for k in range(len(idx) - 1)}


def _intraday_vwap(bars: pd.DataFrame, slices) -> np.ndarray:
    a = bars.amount.to_numpy(float)
    v = bars.volume.to_numpy(float)
    out = np.empty(len(bars))
    for s, e in slices.values():
        ca, cv = np.cumsum(a[s:e]), np.cumsum(v[s:e])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:e] = np.where(cv > 0, ca / cv, np.nan)
    return out
