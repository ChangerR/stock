"""绩效指标。所有盈亏均为相对「只持有底仓」的超额收益（元），已扣费用和滑点。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult

UP_TH, DN_TH = 2.0, -2.0  # 事后归因用的日类型阈值（日涨跌幅 %），不能用于交易决策


def day_type(r: float) -> str:
    return "上涨趋势日" if r > UP_TH else ("下跌趋势日" if r < DN_TH else "震荡日")


def summary(res: BacktestResult) -> dict:
    tr, dl = res.trips, res.daily
    n = len(tr)
    net = float(dl.pnl.sum())
    wins, losses = tr[tr.net > 0], tr[tr.net <= 0]
    dd = float((dl.cum_pnl - dl.cum_pnl.cummax().clip(lower=0)).min()) if len(dl) else 0.0
    bench = float(dl.bench.sum())
    sd = dl.pnl.std(ddof=1)
    t_day = float(dl.pnl.mean() / sd * np.sqrt(len(dl))) if sd > 0 else float("nan")
    lo, hi = bootstrap_ci(dl.pnl.to_numpy())
    fees, slip = cost_totals(res)
    return {
        "区间": f"{dl.index[0]}~{dl.index[-1]}" if len(dl) else "-",
        "交易日数": len(dl),
        "有做T的天数": int(tr.date.nunique()) if n else 0,
        "闭环次数": n,
        "倒T/正T": f"{(tr.dir == '倒T').sum()}/{(tr.dir == '正T').sum()}",
        "做T净收益(元)": round(net, 1),
        "毛收益(含滑点,未扣费)(元)": round(net + fees, 1),
        "费用合计(元)": round(fees, 1),
        "滑点合计(元)": round(slip, 1),
        "胜率(净)": f"{len(wins) / n:.1%}" if n else "-",
        "平均盈利(元)": round(float(wins.net.mean()), 1) if len(wins) else 0.0,
        "平均亏损(元)": round(float(losses.net.mean()), 1) if len(losses) else 0.0,
        "单笔平均净收益(元)": round(net / n, 2) if n else 0.0,
        "日均净收益t值": round(t_day, 2),
        "净收益95%自助法区间(元)": f"[{lo:,.0f}, {hi:,.0f}]",
        "最差单日(元)": round(float(dl.pnl.min()), 1) if len(dl) else 0.0,
        "累计净收益最大回撤(元)": round(dd, 1),
        "止盈TP/止损STOP/尾盘EOD": "/".join(str(int((tr.reason == r).sum())) for r in ("TP", "STOP", "EOD")),
        "跨夜未回补次数": int(tr.overnight.sum()) if n else 0,
        "被拒指令数": len(res.rejects),
        "基准:只持有底仓盈亏(元)": round(bench, 1),
        "持有+做T 合计盈亏(元)": round(bench + net, 1),
    }


def cost_totals(res: BacktestResult) -> tuple[float, float]:
    """按成交汇总的费用与滑点。毛收益统一定义为「逐日盯市净收益 + 费用」，这样跨除权除息日的持仓也能正确对账
    （闭环价差 gross 在跨除权日时会把除权造成的价格下跌误算为收益）。"""
    fl = res.fills
    if fl is None or fl.empty:
        return 0.0, 0.0
    return float(fl.fee.sum()), float(fl.slip.sum()) if "slip" in fl else 0.0


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """按交易日独立重抽样的「总净收益」置信区间（日内策略日间相关性弱，iid 自助法足够粗看）。"""
    if len(x) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    sums = x[rng.integers(0, len(x), size=(n_boot, len(x)))].sum(axis=1)
    return float(np.quantile(sums, alpha / 2)), float(np.quantile(sums, 1 - alpha / 2))


def bootstrap_p_positive(x: np.ndarray, n_boot: int = 20000, seed: int = 0) -> float:
    """单侧自助法 p 值：重抽样总和 <= 0 的比例（原假设：期望 <= 0）。"""
    if len(x) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    out, left = 0, n_boot
    while left > 0:
        k = min(left, 2000)
        out += int((x[rng.integers(0, len(x), size=(k, len(x)))].sum(axis=1) <= 0).sum())
        left -= k
    return out / n_boot


def deflated_sharpe(x: np.ndarray, sr_trials: np.ndarray, var_sr: float | None = None) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio。
    x：被选中策略的日度收益序列；sr_trials：所有试验（配置）的日夏普率（同一口径），N = len(sr_trials)。
    返回 {sr, sr0, dsr}：sr0 为 N 次独立试验下最大夏普率的期望，dsr = P(真实 SR > sr0)。"""
    from statistics import NormalDist

    nd = NormalDist()
    x = np.asarray(x, float)
    T = len(x)
    sd = x.std(ddof=1)
    sr = x.mean() / sd if sd > 0 else 0.0
    N = len(sr_trials)
    v = var_sr if var_sr is not None else (float(np.var(sr_trials, ddof=1)) if N > 1 else 0.0)
    gamma = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - gamma) * nd.inv_cdf(1 - 1 / N) + gamma * nd.inv_cdf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    z = (x - x.mean()) / sd if sd > 0 else x * 0
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    dsr = nd.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return {"sr": float(sr), "sr0": float(sr0), "dsr": float(dsr), "N": N, "T": T}


def max_drawdown(pnl: pd.Series) -> float:
    cum = pnl.cumsum()
    return float((cum - cum.cummax().clip(lower=0)).min()) if len(cum) else 0.0


def paired_diff_ci(a: pd.Series, b: pd.Series, **kw) -> tuple[float, float, float]:
    """两策略同期逐日盈亏之差 (a-b) 的总和及自助法区间。"""
    d = (a - b.reindex(a.index).fillna(0)).to_numpy()
    lo, hi = bootstrap_ci(d, **kw)
    return float(d.sum()), lo, hi


def by_daytype(res: BacktestResult) -> pd.DataFrame:
    tr, dl = res.trips, res.daily
    dt = dl.ret.apply(day_type)
    rows = []
    for name in ["上涨趋势日", "震荡日", "下跌趋势日"]:
        days = dt[dt == name].index
        t = tr[tr.date.isin(days)]
        rows.append({"日类型": name, "天数": len(days), "闭环": len(t),
                     "净收益(元)": round(float(dl.pnl[dt == name].sum()), 1),
                     "胜率": f"{(t.net > 0).mean():.0%}" if len(t) else "-",
                     "倒T净收益": round(float(t[t.dir == "倒T"].net.sum()), 1),
                     "正T净收益": round(float(t[t.dir == "正T"].net.sum()), 1)})
    return pd.DataFrame(rows)


def by_reason(res: BacktestResult) -> pd.DataFrame:
    tr = res.trips
    if not len(tr):
        return pd.DataFrame(columns=["dir", "reason", "次数", "净收益", "平均"])
    g = tr.groupby(["dir", "entry_tag", "reason"]).agg(次数=("net", "size"), 净收益=("net", "sum"), 平均=("net", "mean"))
    return g.round(1).reset_index()


def by_year(res: BacktestResult) -> pd.DataFrame:
    dl, tr = res.daily.copy(), res.trips.copy()
    dl["year"] = dl.index.str[:4]
    tr["year"] = tr.date.str[:4]
    rows = []
    for y, g in dl.groupby("year"):
        t = tr[tr.year == y]
        rows.append({"年份": y, "交易日": len(g), "闭环": len(t), "做T净收益(元)": round(float(g.pnl.sum()), 1),
                     "费用(元)": round(float(t.fees.sum()), 1), "只持有底仓盈亏(元)": round(float(g.bench.sum()), 1),
                     "年内涨跌幅": f"{(1 + g.ret / 100).prod() - 1:.1%}"})
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw)
