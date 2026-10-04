"""把配置、数据、引擎、策略串起来。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import RunConfig, TickerSpec
from .data.store import REPO_ROOT, MarketData, load
from .engine import BacktestResult, Engine
from .strategies import make


@lru_cache(maxsize=16)
def load_cached(code: str, freq: str, data_dir: str) -> MarketData:
    d = Path(data_dir)
    return load(code, freq, d if d.is_absolute() else REPO_ROOT / d)


def run_one(cfg: RunConfig, ticker: TickerSpec, periods: dict[str, tuple[str, str]] | None = None,
            md: MarketData | None = None, **override) -> dict[str, BacktestResult]:
    md = md or load_cached(ticker.code, cfg.freq, cfg.data_dir)
    ecfg = cfg.engine_config(ticker)
    strat = make(cfg.strategy, **cfg.strategy_params(ticker, **override))
    if hasattr(strat, "bind_costs"):
        strat.bind_costs(ecfg.fees, ecfg.slippage)
    feats = strat.daily_features(md)
    eng = Engine(ecfg)
    out = {}
    for label, (s, e) in (periods or cfg.periods).items():
        out[label] = eng.run(md, strat, md.slice_days(s, e), features=feats)
    return out


def run_config(cfg: RunConfig, **override) -> dict[str, dict[str, BacktestResult]]:
    return {t.code: run_one(cfg, t, **override) for t in cfg.tickers}
