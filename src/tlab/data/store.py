"""本地数据缓存与预处理。

缓存布局：data/{code}_{freq}.csv.gz（freq = 5 / d），data/MANIFEST.json 记录来源、区间、行数、sha256。
仓库里提交的是固定区间的快照（见 README「数据」一节）；新下载的数据写到同一位置，
是否提交由人决定（git diff 能看到 MANIFEST 的变化）。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..market import board_of, limit_prices, rules_for
from .sources import DAILY_FIELDS, MINUTE_FIELDS, get_source

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _path(data_dir: Path, code: str, freq: str) -> Path:
    return Path(data_dir) / f"{code}_{'5min' if freq == '5' else freq}.csv.gz"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _update_manifest(data_dir: Path, p: Path, df: pd.DataFrame, source: str):
    mf = Path(data_dir) / "MANIFEST.json"
    man = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    man[p.name] = {
        "source": source,
        "adjust": "none (baostock adjustflag=3)",
        "first_date": str(df.date.min()),
        "last_date": str(df.date.max()),
        "rows": int(len(df)),
        "sha256": _sha256(p),
        "fetched_at": dt.date.today().isoformat(),
    }
    mf.write_text(json.dumps(dict(sorted(man.items())), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(code: str, start: str, end: str, freqs=("5", "d"), source: str = "baostock",
          data_dir: Path = DEFAULT_DATA_DIR, verbose: bool = True) -> None:
    """下载并写入缓存（覆盖同名文件）。"""
    src = get_source(source, verbose=verbose)
    try:
        for f in freqs:
            df = src.fetch_daily(code, start, end) if f == "d" else src.fetch_minute(code, start, end, f)
            if df.empty:
                raise RuntimeError(f"{source} 返回空数据: {code} {f} {start}..{end}")
            p = _path(data_dir, code, f)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
            _update_manifest(data_dir, p, df, source)
            if verbose:
                print(f"写入 {p} ({len(df)} 行, {df.date.min()} ~ {df.date.max()})")
    finally:
        if hasattr(src, "close"):
            src.close()


def load_raw(code: str, freq: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = _path(data_dir, code, freq)
    if not p.exists():
        raise FileNotFoundError(f"缺少缓存 {p}，先运行: tlab fetch {code} --start ... --end ...")
    return pd.read_csv(p, dtype={"date": str, "time": str})


@dataclass
class MarketData:
    """单只股票的预处理数据。bars 为分钟线（含 hhmm），days 以 date 为索引。"""
    code: str
    bars: pd.DataFrame
    days: pd.DataFrame

    def slice_days(self, start: str | None, end: str | None) -> list[str]:
        idx = self.days.index
        return [d for d in idx if (start is None or d >= start) and (end is None or d <= end)]


def prepare(code: str, m: pd.DataFrame, d: pd.DataFrame) -> MarketData:
    d = d.copy()
    for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tradestatus"] = pd.to_numeric(d["tradestatus"], errors="coerce").fillna(0).astype(int)
    d["isST"] = pd.to_numeric(d["isST"], errors="coerce").fillna(0).astype(int) if "isST" in d else 0
    d = d[d.tradestatus == 1].copy()  # 去掉停牌日
    m = m.copy()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m[m.date.isin(d.date) & (m.close > 0)].copy()
    m["hhmm"] = m.time.astype(str).str[8:12].astype(int)  # K 线结束时刻，如 935、1450
    m = m.sort_values(["date", "hhmm"]).reset_index(drop=True)
    d = d[d.date.isin(m.date.unique())].set_index("date").sort_index()

    lim = [rules_for(code, date, bool(st)) for date, st in zip(d.index, d.isST)]
    d["limit_pct"] = [r.limit_pct for r in lim]
    ups, dns = zip(*(limit_prices(pc, r.limit_pct) for pc, r in zip(d.preclose, lim))) if len(d) else ((), ())
    d["lim_up"], d["lim_dn"] = np.array(ups, dtype=float), np.array(dns, dtype=float)
    d["ret"] = d.pctChg.astype(float)
    d["n_bars"] = m.groupby("date").size()
    d["board"] = board_of(code)
    return MarketData(code, m, d)


def load(code: str, freq: str = "5", data_dir: Path = DEFAULT_DATA_DIR) -> MarketData:
    return prepare(code, load_raw(code, freq, data_dir), load_raw(code, "d", data_dir))


def quality_report(md: MarketData) -> dict:
    """数据质量摘要：缺根天数、分钟汇总收盘与日线收盘的偏差。"""
    last = md.bars.groupby("date").close.last()
    diff = (last / md.days.close - 1).abs()
    return {
        "code": md.code,
        "days": len(md.days),
        "bars": len(md.bars),
        "first": md.days.index[0],
        "last": md.days.index[-1],
        "days_not_48_bars": int((md.days.n_bars != 48).sum()),
        "close_mismatch_median": float(diff.median()),
        "close_mismatch_max": float(diff.max()),
        "preclose_ne_prev_close_days": int((md.days.preclose.iloc[1:].values
                                            != md.days.close.shift(1).iloc[1:].values).sum()),
    }
