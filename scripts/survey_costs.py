"""事后探索（未预注册、不参与判定）：各族入选配置在更低成本下的表现，用来区分「信号无优势」与「优势被成本吃掉」。

场景：A 当前口径（佣金 0.025% 最低 5 元，历史印花税/过户费，滑点 0.02%/1tick）
      B 低佣金（佣金 0.01%、无最低，其余同 A）
      C 低佣金 + 零滑点（理论上限；印花税等法定费用无法避免）
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from tlab.config import load_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "survey"
SCEN = {"A 当前口径": {}, "B 低佣金无最低": {"fees": {"commission": 0.0001, "commission_min": 0.0}},
        "C 低佣金+零滑点": {"fees": {"commission": 0.0001, "commission_min": 0.0},
                        "engine": {"slippage": {"pct": 0.0, "min_ticks": 0}}}}


def _run(args):
    fam, strat, params, scen = args
    from tlab.runner import run_one

    cfg = load_config(ROOT / "configs" / "survey.yaml")
    cfg.strategy, cfg.params = strat, {}
    cfg.fees.update(SCEN[scen].get("fees", {}))
    cfg.engine.update(SCEN[scen].get("engine", {}))
    out = {}
    for t in cfg.tickers:
        for label, r in run_one(cfg, t, **params).items():
            out[label] = out.get(label, 0.0) + float(r.daily.pnl.sum())
    return fam, scen, out


def main():
    winners = json.loads((OUT / "winners.json").read_text(encoding="utf-8"))
    import yaml

    grid = yaml.safe_load((ROOT / "configs" / "survey_grid.yaml").read_text(encoding="utf-8"))
    jobs = [(f, grid["families"][f]["strategy"], w["params"], s) for f, w in winners.items() if w for s in SCEN]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_run, jobs))
    rows = {}
    for fam, scen, out in res:
        r = rows.setdefault(fam, {"策略族": winners[fam]["label"], "入选配置": winners[fam]["config"]})
        r[f"{scen} 样本内"] = round(out["样本内"])
        r[f"{scen} 样本外"] = round(out["样本外"])
    df = pd.DataFrame(rows.values())
    df.to_csv(OUT / "cost_sensitivity.csv", index=False)
    (OUT / "cost_sensitivity.md").write_text(
        "# 事后探索：成本敏感性（5 股合计净超额收益，元；未预注册，不参与判定）\n\n"
        "> 非投资建议，历史不代表未来。\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_string())


if __name__ == "__main__":
    main()
