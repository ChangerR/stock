"""策略注册表。新增策略：在本目录新建模块，用 @register 装饰类，并在下方 import。"""
from __future__ import annotations

from ..engine import Strategy

REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in REGISTRY:
        raise ValueError(f"策略名重复: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


def make(name: str, **params) -> Strategy:
    try:
        return REGISTRY[name](**params)
    except KeyError:
        raise ValueError(f"未知策略 {name}，可选: {sorted(REGISTRY)}") from None


from . import families, vwap_band, vwap_band_regime  # noqa: E402,F401
