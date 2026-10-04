"""命令行：

  tlab fetch sh.688981 --start 2020-07-16 --end 2026-09-24     # 下载 5 分钟 + 日线到 data/
  tlab check sh.688981                                          # 数据质量摘要
  tlab run configs/baseline_688981.yaml [--out reports/runs/x]  # 按配置回测并生成报告
  tlab grid configs/baseline_688981.yaml                        # 按配置里的 grid 做样本内/外参数网格
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data.store import REPO_ROOT, fetch, load, quality_report
from .report import write_run_report
from .runner import run_config, run_one


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tlab", description="A股底仓做T 研究框架（非投资建议）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="从数据源下载并缓存")
    f.add_argument("codes", nargs="+")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--source", default="baostock")
    f.add_argument("--freq", nargs="+", default=["5", "d"])
    c = sub.add_parser("check", help="数据质量摘要")
    c.add_argument("codes", nargs="+")
    r = sub.add_parser("run", help="按配置回测")
    r.add_argument("config")
    r.add_argument("--out")
    g = sub.add_parser("grid", help="参数网格（样本内/外）")
    g.add_argument("config")
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        for code in a.codes:
            fetch(code, a.start, a.end, tuple(a.freq), a.source)
    elif a.cmd == "check":
        for code in a.codes:
            print(json.dumps(quality_report(load(code)), ensure_ascii=False))
    elif a.cmd == "run":
        cfg = load_config(a.config)
        res = run_config(cfg)
        out = Path(a.out) if a.out else REPO_ROOT / "reports" / "runs" / cfg.name
        p = write_run_report(cfg.name, cfg.description, res, out, meta={"config": a.config})
        print(f"报告: {p}")
    elif a.cmd == "grid":
        cfg = load_config(a.config)
        keys = list(cfg.grid)
        rows = []
        for t in cfg.tickers:
            for vals in itertools.product(*(cfg.grid[k] for k in keys)):
                ov = dict(zip(keys, vals))
                for label, res in run_one(cfg, t, **ov).items():
                    rows.append(dict(code=t.code, period=label, **ov, net=round(res.daily.pnl.sum(), 1),
                                     trips=len(res.trips)))
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
