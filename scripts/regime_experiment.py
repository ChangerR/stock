"""倒T 趋势/状态过滤实验（严格按 reports/preregistration.md 执行）。

输出到 reports/regime/：candidates.csv、selection.json、图表，以及供报告引用的 tables.md。
"""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlab.config import load_config
from tlab.metrics import by_daytype, by_reason, paired_diff_ci, summary, to_md
from tlab.report import DISCLAIMER, concat_daily, plot_equity, setup_fonts
from tlab.strategies.vwap_band_regime import describe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "regime"
CFG = ROOT / "configs" / "baseline_688981_histfees.yaml"
CFG_PROTO = ROOT / "configs" / "baseline_688981.yaml"

GRID = {
    "day_ret": [3, 6, 9],
    "vwap_frac": [0.6, 0.75, 0.9],
    "vwap_slope": [0.5, 1, 2],
    "orb": [3, 6, 12],
    "gap": [2, 4, 8],
    "daily_trend": [5, 20, 60],
}


def _run(args):
    cfg_path, override = args
    from tlab.runner import run_one

    cfg = load_config(cfg_path)
    cfg.strategy = "vwap_band_regime"
    return override, run_one(cfg, cfg.tickers[0], **override)


def row(override, res):
    ins, oos = res["样本内"], res["样本外"]
    return dict(label=describe(dict(dict(filter="none", th=0, symmetric=False, action="skip"), **override))
                if "sides" not in override else f"只做{'正T' if override['sides'] == 'zheng' else '倒T'}(对照)",
                **{k: override.get(k) for k in ("filter", "th", "symmetric", "action", "sides")},
                is_net=round(ins.daily.pnl.sum(), 1), is_trips=len(ins.trips),
                is_dao=int((ins.trips.dir == "倒T").sum()),
                oos_net=round(oos.daily.pnl.sum(), 1), oos_trips=len(oos.trips),
                oos_dao=int((oos.trips.dir == "倒T").sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [dict(filter="none"), dict(sides="zheng"), dict(sides="dao")]
    for flt, ths in GRID.items():
        for th, sym in itertools.product(ths, [False, True]):
            jobs.append(dict(filter=flt, th=th, symmetric=sym))
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(_run, [(CFG, j) for j in jobs]))
    res_by = {json.dumps(o, sort_keys=True): r for o, r in results}
    df = pd.DataFrame([row(o, r) for o, r in results])
    base = df.iloc[0]
    cand = df[df["filter"].notna() & (df["filter"] != "none")].copy()

    # ---- 选择（只用样本内）----
    min_trips = 0.5 * base.is_trips
    cand["eligible"] = cand.is_trips >= min_trips
    elig = cand[cand.eligible].sort_values(["is_net", "is_trips"], ascending=[False, False])
    best = elig.iloc[0]
    cand["is_rank"] = cand.is_net.rank(ascending=False, method="min").astype(int)
    cand["oos_rank"] = cand.oos_net.rank(ascending=False, method="min").astype(int)
    skip_ov = dict(filter=best["filter"], th=float(best.th) if best["filter"] not in ("orb", "daily_trend")
                   else int(best.th), symmetric=bool(best.symmetric))
    flip_ov = dict(skip_ov, action="flip")
    _, flip_res = _run((CFG, flip_ov))
    flip_row = row(flip_ov, flip_res)
    use_flip = flip_row["is_net"] > best.is_net
    final_ov = flip_ov if use_flip else skip_ov
    final_res = flip_res if use_flip else _run((CFG, skip_ov))[1]

    base_res = res_by[json.dumps(dict(filter="none"), sort_keys=True)]
    zheng_res = res_by[json.dumps(dict(sides="zheng"), sort_keys=True)]
    _, final_proto = _run((CFG_PROTO, final_ov))
    _, base_proto = _run((CFG_PROTO, dict(filter="none")))

    tests = {}
    for label in ["样本内", "样本外"]:
        d, lo, hi = paired_diff_ci(final_res[label].daily.pnl, base_res[label].daily.pnl)
        d2, lo2, hi2 = paired_diff_ci(final_res[label].daily.pnl, zheng_res[label].daily.pnl)
        tests[label] = dict(vs_baseline=[round(d, 1), round(lo, 1), round(hi, 1)],
                            vs_zheng_only=[round(d2, 1), round(lo2, 1), round(hi2, 1)])

    sel = dict(min_is_trips=min_trips, best_skip=skip_ov, best_skip_is_net=float(best.is_net),
               flip_is_net=flip_row["is_net"], use_flip=bool(use_flip), final=final_ov,
               final_is_net=round(final_res["样本内"].daily.pnl.sum(), 1),
               final_oos_net=round(final_res["样本外"].daily.pnl.sum(), 1),
               baseline_is_net=float(base.is_net), baseline_oos_net=float(base.oos_net),
               proto_fees={"final": {k: round(v.daily.pnl.sum(), 1) for k, v in final_proto.items()},
                           "baseline": {k: round(v.daily.pnl.sum(), 1) for k, v in base_proto.items()}},
               paired_bootstrap=tests)
    (OUT / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["label", "filter", "th", "symmetric", "action", "sides", "is_net", "is_trips", "is_dao",
            "oos_net", "oos_trips", "oos_dao"]
    allrows = pd.concat([df[cols], pd.DataFrame([flip_row])[cols]], ignore_index=True)
    allrows = allrows.merge(cand[["label", "eligible", "is_rank", "oos_rank"]], on="label", how="left")
    allrows.to_csv(OUT / "candidates.csv", index=False)

    # ---- 表格 ----
    L = ["# 倒T 趋势过滤实验明细（688981，历史费率）", "", DISCLAIMER, "", "规则见 ../preregistration.md，结论见 ../README.md §4。", "", f"最终方案：`{json.dumps(final_ov, ensure_ascii=False)}`", ""]
    show = allrows.copy()
    show["symmetric"] = show.symmetric.map({True: "是", False: "否"}).fillna("")
    L += ["## 全部候选（样本内选择；样本外仅展示）", "",
          to_md(show[["label", "is_net", "is_trips", "is_dao", "eligible", "is_rank", "oos_net", "oos_trips",
                      "oos_rank"]].rename(columns={
                          "label": "方案", "is_net": "样本内净收益", "is_trips": "样本内闭环", "is_dao": "样本内倒T",
                          "eligible": "满足闭环数要求", "is_rank": "样本内排名", "oos_net": "样本外净收益",
                          "oos_trips": "样本外闭环", "oos_rank": "样本外排名"})), ""]
    comp = {"基线(历史费率)": base_res, "只做正T(对照)": zheng_res, f"最终: {describe(final_ov)}": final_res}
    st = {}
    for name, r in comp.items():
        for label in ["样本内", "样本外"]:
            st[f"{name} / {label}"] = summary(r[label])
    L += ["## 基线 / 对照 / 最终方案 指标", "", to_md(pd.DataFrame(st).reset_index().rename(columns={"index": "指标"})), ""]
    for name, r in comp.items():
        for label in ["样本内", "样本外"]:
            L += [f"### {name}：按日类型（事后归因，{label}）", "", to_md(by_daytype(r[label])), ""]
    for label in ["样本内", "样本外"]:
        L += [f"### 最终方案：按方向与平仓原因（{label}）", "", to_md(by_reason(final_res[label])), ""]
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- 图 ----
    setup_fonts()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    c = allrows[allrows["filter"].notna() & (allrows["filter"] != "none")]
    for flt, g in c.groupby("filter"):
        ax.scatter(g.is_net, g.oos_net, label=flt, s=40, alpha=0.8,
                   marker="o" if flt != "daily_trend" else "s")
    for lab, r, mk in [("基线", base, "X"), ("只做正T", df.iloc[1], "P"), ("只做倒T", df.iloc[2], "v")]:
        ax.scatter(r.is_net, r.oos_net, marker=mk, s=140, color="black")
        ax.annotate(lab, (r.is_net, r.oos_net), textcoords="offset points", xytext=(6, 6))
    fr = allrows[allrows.label == describe(final_ov)].iloc[0]
    ax.scatter(fr.is_net, fr.oos_net, s=260, facecolors="none", edgecolors="red", lw=2, label="最终方案")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("样本内净收益（元）")
    ax.set_ylabel("样本外净收益（元）")
    ax.set_title("688981 过滤器候选：样本内 vs 样本外（历史费率）")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.text(0.99, 0.005, "非投资建议，历史不代表未来", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "is_vs_oos_scatter.png", dpi=120)
    plt.close(fig)
    plot_equity({k: concat_daily(v) for k, v in comp.items()}, OUT / "equity_688981.png",
                "688981 基线 vs 过滤方案 vs 只做正T（历史费率，累计做T超额收益）", "2024-07-24")
    print(json.dumps(sel, ensure_ascii=False, indent=2))
    print(allrows.to_string())


if __name__ == "__main__":
    main()
