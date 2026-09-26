"""基于真实 688981 数据：日级特征截断检验、盘中未来数据扰动检验、原型复现回归。"""
import numpy as np
import pytest

from tlab.config import load_config
from tlab.data.store import MarketData, REPO_ROOT, load
from tlab.engine import Engine
from tlab.runner import run_one
from tlab.strategies import make


@pytest.fixture(scope="module")
def md():
    return load("sh.688981")


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "configs" / "baseline_688981.yaml")


def _truncate(md, date, perturb_same_day=True):
    bars = md.bars[md.bars.date <= date].copy()
    days = md.days.loc[:date].copy()
    if perturb_same_day:  # 当天数据被改掉：第 date 行特征仍不能变
        sel = bars.date == date
        for c in ["open", "high", "low", "close"]:
            bars.loc[sel, c] *= 1.37
        days.loc[date, ["close", "ret"]] = [days.loc[date, "close"] * 1.37, 37.0]
    return MarketData(md.code, bars.reset_index(drop=True), days)


@pytest.mark.parametrize("params", [dict(), dict(filter="daily_trend", th=20)])
def test_daily_features_use_only_past(md, params):
    s = make("vwap_band_regime", **params)
    full = s.daily_features(md)
    for date in md.days.index[[10, 200, 700, 1200, -1]]:
        part = s.daily_features(_truncate(md, date)).loc[date]
        pd_full = full.loc[date]
        for col in full.columns:
            a, b = part[col], pd_full[col]
            assert (a == b) or (a != a and b != b), (date, col, a, b)


@pytest.mark.parametrize("params", [dict(filter="none"), dict(filter="vwap_frac", th=0.75, symmetric=True),
                                    dict(filter="orb", th=6, action="flip")])
def test_intraday_decisions_ignore_future_bars(md, cfg, params):
    """把第 j 根之后的 K 线改掉：信号时刻早于第 j 根的成交必须完全不变。"""
    t = cfg.tickers[0]
    ecfg = cfg.engine_config(t)
    s = make("vwap_band_regime", **dict(cfg.params, **params))
    s.bind_costs(ecfg.fees, ecfg.slippage)
    feats = s.daily_features(md)
    rng = np.random.default_rng(1)
    days = [d for d in md.days.index[300:1400:37]]
    compared = 0
    for date in days:
        base = Engine(ecfg).run(md, s, [date], features=feats).fills
        j = int(rng.integers(3, 44))
        bars = md.bars.copy()
        idx = bars.index[bars.date == date][j + 1:]
        bars.loc[idx, ["open", "high", "low", "close"]] *= rng.uniform(0.9, 1.1, size=(len(idx), 1))
        cut = int(bars.loc[bars.index[bars.date == date][j], "hhmm"])
        alt = Engine(ecfg).run(MarketData(md.code, bars, md.days), s, [date], features=feats).fills
        # 信号早于第 j 根收盘 → 成交在第 j 根或之前（未被扰动），信号与成交价都必须完全一致
        pick = lambda f: [] if f.empty else [tuple(r) for r in
                                             f[f.sig_time < cut][["sig_time", "side", "qty", "price", "tag"]].values]
        a, b = pick(base), pick(alt)
        assert a == b, date
        compared += len(a)
    assert compared > 0


def test_baseline_reproduces_prototype(cfg):
    r = run_one(cfg, cfg.tickers[0])
    ins, oos = r["样本内"], r["样本外"]
    assert (round(ins.daily.pnl.sum(), 1), len(ins.trips)) == (-5143.5, 287)
    assert (round(oos.daily.pnl.sum(), 1), len(oos.trips)) == (-6526.5, 182)
    assert round(ins.trips.gross.sum(), 1) == -730.0 and round(oos.trips.fees.sum(), 1) == 3950.5
    assert (ins.trips.dir == "倒T").sum() == 168 and (oos.trips.dir == "倒T").sum() == 110
    assert round(ins.daily.bench.sum(), 1) == -12116.0 and round(oos.daily.bench.sum(), 1) == 28280.0


def test_prototype_grid_cell(cfg):
    r = run_one(cfg, cfg.tickers[0], k=3.0, s=1.5)
    assert round(r["样本内"].daily.pnl.sum(), 1) == -11918.5 and len(r["样本外"].trips) == 334


def test_regime_filter_none_equals_baseline(cfg):
    base = run_one(cfg, cfg.tickers[0])
    cfg2 = load_config(REPO_ROOT / "configs" / "baseline_688981.yaml")
    cfg2.strategy = "vwap_band_regime"
    reg = run_one(cfg2, cfg2.tickers[0], filter="none")
    for label in base:
        assert base[label].trips.equals(reg[label].trips)


def test_config_extends_and_ticker_overrides():
    c = load_config(REPO_ROOT / "configs" / "multistock.yaml")
    assert c.fees["stamp_duty"] == "historical" and c.params["k"] == 4.0
    t = {x.code: x for x in c.tickers}["sz.300033"]
    assert c.strategy_params(t)["lot"] == 100 and c.engine_config(t).base_shares == 200
    assert c.engine_config(t).fees.cost("sell", 100, 100, "2023-01-03") == pytest.approx(5 + 0.1 + 10)
