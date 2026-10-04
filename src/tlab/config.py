"""YAML 配置 → 引擎/策略对象。新增股票或策略只需要写一个配置文件。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .engine import EngineConfig
from .fees import FeeSchedule, Slippage


@dataclass
class TickerSpec:
    code: str
    base_shares: int
    lot: int | None = None     # 覆盖策略参数 lot


@dataclass
class RunConfig:
    name: str
    tickers: list[TickerSpec]
    periods: dict[str, tuple[str, str]]
    strategy: str
    params: dict
    engine: dict = field(default_factory=dict)
    fees: dict = field(default_factory=dict)
    data_dir: str = "data"
    freq: str = "5"
    description: str = ""
    grid: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def engine_config(self, ticker: TickerSpec) -> EngineConfig:
        e = copy.deepcopy(self.engine)
        slip = Slippage(**e.pop("slippage", {}))
        return EngineConfig(base_shares=ticker.base_shares, slippage=slip, fees=FeeSchedule(**self.fees), **e)

    def strategy_params(self, ticker: TickerSpec, **override) -> dict:
        p = dict(self.params, **override)
        if ticker.lot is not None:
            p["lot"] = ticker.lot
        return p


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_raw_config(path: str | Path) -> dict:
    """支持 `extends: 其他配置.yaml`（相对路径），子配置按键深度覆盖父配置；列表整体替换。"""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent:
        raw = _deep_merge(load_raw_config(path.parent / parent), raw)
    return raw


def load_config(path: str | Path) -> RunConfig:
    raw = load_raw_config(path)
    tickers = [TickerSpec(**t) if isinstance(t, dict) else TickerSpec(code=t, base_shares=raw["base_shares"])
               for t in raw["tickers"]]
    periods = {k: (str(v[0]), str(v[1])) for k, v in raw["periods"].items()}
    st = raw["strategy"]
    return RunConfig(name=raw["name"], tickers=tickers, periods=periods, strategy=st["name"],
                     params=st.get("params", {}), engine=raw.get("engine", {}), fees=raw.get("fees", {}),
                     data_dir=raw.get("data_dir", "data"), freq=str(raw.get("freq", "5")),
                     description=raw.get("description", ""), grid=raw.get("grid", {}), raw=raw)
