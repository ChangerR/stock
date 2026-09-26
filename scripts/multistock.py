"""多股票检验：基线 / 预注册选出的过滤方案 / 只做正T 对照，参数完全不变（历史费率）。

用法：python scripts/multistock.py configs/multistock.yaml
配置里的 tickers 来自 scripts/screen_universe.py 的筛选结果，底仓/每次股数按预注册的市值折算规则写死在配置中。
缺数据时自动从 baostock 下载（2020-07-01 起，留出 ATR 预热）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from tlab.analysis import band_events, summarize_events
from tlab.config import load_config
from tlab.data.store import DEFAULT_DATA_DIR, _path, fetch
from tlab.metrics import by_daytype, paired_diff_ci, summary, to_md
from tlab.report import DISCLAIMER, concat_daily, plot_equity
from tlab.runner import load_cached, run_one
from tlab.strategies.vwap_band_regime import describe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "multistock"


def main(cfg_path):
    cfg = load_config(cfg_path)
    cfg.strategy = "vwap_band_regime"
    sel = json.loads((ROOT / "reports" / "regime" / "selection.json").read_text(encoding="utf-8"))
    final = sel["final"]
    variants = {"基线": dict(filter="none"), f"过滤: {describe(final)}": final, "只做正T(对照)": dict(sides="zheng")}
    names = cfg.raw.get("names", {})
    start, end = cfg.raw.get("fetch_range", ["2020-07-01", "2026-09-24"])
    OUT.mkdir(parents=True, exist_ok=True)
    rows, L, failed = [], [DISCLAIMER, ""], []
    for t in cfg.tickers:
        if not (_path(DEFAULT_DATA_DIR, t.code, "5").exists() and _path(DEFAULT_DATA_DIR, t.code, "d").exists()):
            try:
                fetch(t.code, start, end)
            except Exception as e:  # 数据源失败如实记录，不编造
                failed.append((t.code, repr(e)))
                print("下载失败", t.code, e)
                continue
        md = load_cached(t.code, cfg.freq, cfg.data_dir)
        res = {name: run_one(cfg, t, md=md, **ov) for name, ov in variants.items()}
        nm = names.get(t.code, "")
        L += [f"## {t.code} {nm}（底仓 {t.base_shares} 股，每次 {t.lot} 股）", ""]
        st = {}
        for name, r in res.items():
            for label in cfg.periods:
                s = summary(r[label])
                st[f"{name} | {label}"] = s
                rows.append(dict(code=t.code, name=nm, variant=name, period=label, net=s["做T净收益(元)"],
                                 trips=s["闭环次数"], bench=s["基准:只持有底仓盈亏(元)"],
                                 fees=s["费用合计(元)"], gross=s["毛收益(含滑点,未扣费)(元)"],
                                 ci=s["净收益95%自助法区间(元)"]))
        keep = ["交易日数", "闭环次数", "倒T/正T", "做T净收益(元)", "毛收益(含滑点,未扣费)(元)", "费用合计(元)",
                "胜率(净)", "净收益95%自助法区间(元)", "被拒指令数", "跨夜未回补次数", "基准:只持有底仓盈亏(元)",
                "持有+做T 合计盈亏(元)"]
        L += [to_md(pd.DataFrame(st).loc[keep].reset_index().rename(columns={"index": "指标"})), ""]
        for label in cfg.periods:
            d, lo, hi = paired_diff_ci(res[f"过滤: {describe(final)}"][label].daily.pnl, res["基线"][label].daily.pnl)
            L.append(f"- {label}：过滤方案 − 基线 = {d:,.0f} 元，95% 区间 [{lo:,.0f}, {hi:,.0f}]")
        L += ["", f"### 基线按日类型（事后归因，全样本）", ""]
        allb = res["基线"]
        for label in cfg.periods:
            L += [f"{label}：", "", to_md(by_daytype(allb[label])), ""]
        ev = pd.concat([band_events(md, md.slice_days(*cfg.periods[p])).assign(period=p) for p in cfg.periods])
        for p in cfg.periods:
            L += [f"### 偏离带事件研究（{p}，不含费用）", "", to_md(summarize_events(ev[ev.period == p])), ""]
        safe = t.code.replace(".", "")
        plot_equity({k: concat_daily(v) for k, v in res.items()}, OUT / f"equity_{safe}.png",
                    f"{t.code} {nm}：基线 vs 过滤 vs 只做正T（历史费率，累计做T超额收益）",
                    list(cfg.periods.values())[1][0])
        L += [f"![{t.code}](equity_{safe}.png)", ""]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    if failed:
        L += ["## 数据下载失败", ""] + [f"- {c}: {e}" for c, e in failed]
    if len(df):
        pv = df.pivot_table(index=["code", "name"], columns=["variant", "period"], values="net", aggfunc="first")
        bench = df[df.variant == "基线"].pivot_table(index=["code", "name"], columns="period", values="bench",
                                                    aggfunc="first")
        bench.columns = pd.MultiIndex.from_tuples([("只持有底仓", c) for c in bench.columns])
        tab = pd.concat([pv, bench], axis=1)
        tab.columns = [f"{a} | {b}" for a, b in tab.columns]
        head = ["# 多股票检验明细", "", "## 汇总：做T 净超额收益（元，历史费率，已扣费用和滑点）", "",
                tab.reset_index().to_markdown(index=False), ""]
        L = head + L
    (OUT / "details.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs" / "multistock.yaml")
