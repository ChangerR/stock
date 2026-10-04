"""做T 策略族普查（严格按 reports/survey/preregistration.md 与 configs/survey_grid.yaml 执行）。

输出到 reports/survey/：all_configs.csv（每组配置 × 股票 × 时段）、pooled_configs.csv（5 股合计）、
winners.json（入选配置与判定）、tables.md（全部表格）、若干图表。
"""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tlab.config import load_config
from tlab.metrics import bootstrap_ci, bootstrap_p_positive, cost_totals, deflated_sharpe, max_drawdown
from tlab.report import DISCLAIMER, setup_fonts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "survey"
CFG = ROOT / "configs" / "survey.yaml"
GRID = ROOT / "configs" / "survey_grid.yaml"
PERIODS = ["样本内", "样本外"]


def expand(grid: dict) -> list[tuple[str, str, str, dict]]:
    jobs = []
    for fam, spec in grid["families"].items():
        k = 0
        for prod in spec["products"]:
            keys = list(prod)
            for vals in itertools.product(*(prod[x] for x in keys)):
                params = dict(zip(keys, vals))
                jobs.append((fam, f"{fam}#{k:02d}", spec["strategy"], params))
                k += 1
    return jobs


def _run(job):
    fam, cid, strat, params, code = job
    from tlab.runner import run_one

    cfg = load_config(CFG)
    cfg.strategy, cfg.params = strat, {}
    t = next(x for x in cfg.tickers if x.code == code)
    res = run_one(cfg, t, **params)
    out = {}
    for label, r in res.items():
        tr, dl = r.trips, r.daily
        fees, slip = cost_totals(r)
        net = float(dl.pnl.sum())
        turnover = float((r.fills.price * r.fills.qty).sum()) if len(r.fills) else 0.0
        out[label] = dict(net=net, gross=net + fees, fees=fees, slip=slip, turnover=turnover,
                          trips=len(tr), wins=int((tr.net > 0).sum()),
                          mdd=max_drawdown(dl.pnl), bench=float(dl.bench.sum()), rejects=len(r.rejects),
                          overnight=int(tr.overnight.sum()) if len(tr) else 0,
                          pnl=dl.pnl.to_dict(), close=dl.close.to_dict())
    return fam, cid, strat, params, code, out


def describe(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main():
    grid = yaml.safe_load(GRID.read_text(encoding="utf-8"))
    sel = grid["selection"]
    cfg = load_config(CFG)
    codes = [t.code for t in cfg.tickers]
    names = cfg.raw["names"]
    base_jobs = expand(grid)
    jobs = [(f, c, s, p, code) for f, c, s, p in base_jobs for code in codes]
    print(f"{len(base_jobs)} 组配置 × {len(codes)} 只股票 = {len(jobs)} 个回测任务（每个含样本内/外）", flush=True)
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(_run, jobs, chunksize=4))

    rows, series = [], {}
    for fam, cid, strat, params, code, out in results:
        for label, m in out.items():
            series[(cid, code, label)] = pd.Series(m.pop("pnl"))
            m.pop("close")
            rows.append(dict(family=fam, config=cid, strategy=strat, params=json.dumps(params, ensure_ascii=False),
                             code=code, name=names[code], period=label, **m))
    per = pd.DataFrame(rows)
    per["pure_gross"] = per.gross + per.slip
    per.to_csv(OUT / "all_configs.csv", index=False, float_format="%.2f")

    # ---- 5 股合计 ----
    pooled_rows, pooled_series = [], {}
    for (fam, cid, label), g in per.groupby(["family", "config", "period"], sort=False):
        s = pd.concat([series[(cid, c, label)] for c in codes], axis=1).fillna(0).sum(axis=1).sort_index()
        pooled_series[(cid, label)] = s
        sd = s.std(ddof=1)
        pooled_rows.append(dict(family=fam, config=cid, params=g.params.iloc[0], period=label,
                                net=g.net.sum(), gross=g.gross.sum(), pure_gross=g.pure_gross.sum(),
                                fees=g.fees.sum(), slip=g.slip.sum(), trips=int(g.trips.sum()),
                                edge_bp=g.pure_gross.sum() / max(1.0, g.turnover.sum() / 2) * 1e4,
                                cost_bp=(g.fees.sum() + g.slip.sum()) / max(1.0, g.turnover.sum() / 2) * 1e4,
                                win=g.wins.sum() / max(1, g.trips.sum()), mdd=max_drawdown(s),
                                bench=g.bench.sum(), sr=float(s.mean() / sd) if sd > 0 else 0.0,
                                pos_stocks=int((g.net > 0).sum())))
    pooled = pd.DataFrame(pooled_rows)
    wide = pooled.pivot_table(index=["family", "config", "params"], columns="period",
                              values=["net", "gross", "pure_gross", "fees", "slip", "trips", "win", "mdd", "sr", "pos_stocks",
                                      "edge_bp", "cost_bp"],
                              aggfunc="first")
    wide.columns = [f"{a}_{'is' if b == '样本内' else 'oos'}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT / "pooled_configs.csv", index=False, float_format="%.4f")

    # ---- 选择（只用样本内）----
    sr_trials = wide.sr_is.to_numpy()
    winners = {}
    for fam in grid["families"]:
        w = wide[(wide.family == fam) & (wide.trips_is >= sel["min_pooled_is_trips"])]
        if w.empty:
            winners[fam] = None
            continue
        best = w.sort_values(["net_is", "trips_is"], ascending=[False, False]).iloc[0]
        cid = best.config
        s_is, s_oos = pooled_series[(cid, "样本内")], pooled_series[(cid, "样本外")]
        p_oos = bootstrap_p_positive(s_oos.to_numpy())
        lo, hi = bootstrap_ci(s_oos.to_numpy())
        lo_is, hi_is = bootstrap_ci(s_is.to_numpy())
        ds = deflated_sharpe(s_is.to_numpy(), sr_trials)
        ds_lenient = deflated_sharpe(s_is.to_numpy(), sr_trials, var_sr=1 / len(s_is))  # 敏感性：试验间方差取纯噪声 1/T
        n_pos = int(best.pos_stocks_oos)
        c1 = best.net_oos > 0 and p_oos < sel["alpha"] / sel["n_families"]
        c2 = n_pos >= sel["min_positive_stocks_oos"]
        c3 = ds["dsr"] >= sel["dsr_threshold"]
        verdict = "稳健有效" if (c1 and c2 and c3) else (
            "待验证" if (best.net_oos > 0 and (p_oos < sel["alpha"]) and c2) else "无效")
        winners[fam] = dict(config=cid, params=json.loads(best.params), label=grid["families"][fam]["label"],
                            net_is=best.net_is, net_oos=best.net_oos, ci_is=[lo_is, hi_is], ci_oos=[lo, hi],
                            p_oos=p_oos, pos_stocks_oos=n_pos, dsr=ds, dsr_lenient=ds_lenient, c1=bool(c1), c2=bool(c2), c3=bool(c3),
                            verdict=verdict, is_rank_in_family=1,
                            oos_rank_in_family=int((wide[wide.family == fam].net_oos > best.net_oos).sum() + 1),
                            n_family=int((wide.family == fam).sum()))
    (OUT / "winners.json").write_text(json.dumps(winners, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")

    # ---- 表格 ----
    L = ["# 策略族普查：全部表格（自动生成）", "", DISCLAIMER, "",
         "规则见 [preregistration.md](preregistration.md)，结论见 [README.md](README.md)。单位：元；"
         "净收益 = 相对只持有底仓的超额收益（已扣费用和滑点）；毛收益 = 含滑点、未扣费；纯毛收益 = 未扣滑点和费用。", ""]
    cross = []
    for fam, w in winners.items():
        if w is None:
            continue
        g = per[per.config == w["config"]]
        for code in codes + ["合计"]:
            for label in PERIODS:
                if code == "合计":
                    r = pooled[(pooled.config == w["config"]) & (pooled.period == label)].iloc[0]
                    d = dict(net=r.net, gross=r.gross, fees=r.fees, trips=r.trips, win=r.win, mdd=r.mdd, bench=r.bench)
                else:
                    r = g[(g.code == code) & (g.period == label)].iloc[0]
                    d = dict(net=r.net, gross=r.gross, fees=r.fees, trips=r.trips, win=r.wins / max(1, r.trips),
                             mdd=r.mdd, bench=r.bench)
                cross.append({"策略族": w["label"], "股票": "5股合计" if code == "合计" else f"{names[code]} {code}",
                              "时段": label, "净收益": round(d["net"]), "毛收益": round(d["gross"]),
                              "费用": round(d["fees"]), "闭环": int(d["trips"]), "胜率": f"{d['win']:.0%}",
                              "最大回撤": round(d["mdd"]), "只持有盈亏": round(d["bench"]),
                              "跑赢持有": "是" if d["net"] > 0 else "否"})
    cross = pd.DataFrame(cross)
    cross.to_csv(OUT / "cross_family.csv", index=False)
    wide_rows = []
    for (fam, stock), g in cross.groupby(["策略族", "股票"], sort=False):
        a, b = g[g.时段 == "样本内"].iloc[0], g[g.时段 == "样本外"].iloc[0]
        wide_rows.append({"策略族": fam, "股票": stock, "净收益 内/外": f"{a.净收益:,} / {b.净收益:,}",
                          "毛收益 内/外": f"{a.毛收益:,} / {b.毛收益:,}", "闭环 内/外": f"{a.闭环} / {b.闭环}",
                          "胜率 内/外": f"{a.胜率} / {b.胜率}", "最大回撤 内/外": f"{a.最大回撤:,} / {b.最大回撤:,}",
                          "只持有 内/外": f"{a.只持有盈亏:,} / {b.只持有盈亏:,}",
                          "跑赢持有 内/外": f"{a.跑赢持有} / {b.跑赢持有}"})
    (OUT / "cross_family_wide.md").write_text(pd.DataFrame(wide_rows).to_markdown(index=False) + "\n", encoding="utf-8")

    summ = []
    for fam, w in winners.items():
        if w is None:
            continue
        r = wide[wide.config == w["config"]].iloc[0]
        summ.append({"策略族": w["label"], "入选配置": f"{w['config']}：{describe(w['params'])}",
                     "样本内净收益": round(r.net_is), "样本外净收益": round(r.net_oos),
                     "样本外95%区间": f"[{w['ci_oos'][0]:,.0f}, {w['ci_oos'][1]:,.0f}]",
                     "样本外单侧p": round(w["p_oos"], 4),
                     "样本内纯毛收益/毛收益/费用": f"{r.pure_gross_is:,.0f} / {r.gross_is:,.0f} / {r.fees_is:,.0f}",
                     "样本外纯毛收益/毛收益/费用": f"{r.pure_gross_oos:,.0f} / {r.gross_oos:,.0f} / {r.fees_oos:,.0f}",
                     "闭环 内/外": f"{int(r.trips_is)} / {int(r.trips_oos)}",
                     "样本外为正股票数": f"{w['pos_stocks_oos']}/5",
                     "每笔纯毛优势/成本(bp) 内": f"{r.edge_bp_is:.1f} / {r.cost_bp_is:.1f}",
                     "每笔纯毛优势/成本(bp) 外": f"{r.edge_bp_oos:.1f} / {r.cost_bp_oos:.1f}",
                     "样本内日夏普": round(w["dsr"]["sr"], 3),
                     "DSR(N=%d) 预注册口径 / 宽松口径" % len(sr_trials):
                         f"{w['dsr']['dsr']:.3f} / {w['dsr_lenient']['dsr']:.3f}",
                     "判定": w["verdict"]})
    summ = pd.DataFrame(summ)
    L += ["## 1. 各族入选配置（5 股合计）", "", summ.to_markdown(index=False), ""]
    L += ["## 2. 跨族对比（入选配置，逐股票与合计）", "", (OUT / "cross_family_wide.md").read_text(encoding="utf-8"), ""]
    show = wide.copy()
    show["入选"] = show.config.isin([w["config"] for w in winners.values() if w])
    show = show[["family", "config", "params", "net_is", "trips_is", "pure_gross_is", "gross_is", "fees_is",
                 "net_oos", "trips_oos", "gross_oos", "fees_oos", "pos_stocks_oos", "入选"]].round(0)
    fam_best = []
    for fam, g in wide.groupby("family", sort=False):
        b = g.sort_values("net_oos", ascending=False).iloc[0]
        w = winners.get(fam)
        fam_best.append({"策略族": grid["families"][fam]["label"], "组数": len(g),
                         "样本内为正的组数": int((g.net_is > 0).sum()), "样本外为正的组数": int((g.net_oos > 0).sum()),
                         "两段都为正的组数": int(((g.net_is > 0) & (g.net_oos > 0)).sum()),
                         "事后样本外最好（仅展示）": f"{b.config} {b.net_oos:,.0f}（样本内 {b.net_is:,.0f}）",
                         "入选配置样本外排名": f"{w['oos_rank_in_family']}/{w['n_family']}" if w else "-"})
    L += ["## 3. 各族参数稳健性（5 股合计；「事后最好」不参与选择）", "", pd.DataFrame(fam_best).to_markdown(index=False), ""]
    L += [f"## 4. 全部 {len(wide)} 组配置（5 股合计）", "", show.to_markdown(index=False), ""]
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    charts(winners, wide, pooled, pooled_series, per, codes, names, grid)
    print(summ.to_string())


def charts(winners, wide, pooled, pooled_series, per, codes, names, grid):
    import matplotlib.pyplot as plt

    setup_fonts()
    foot = lambda fig: fig.text(0.99, 0.005, "非投资建议，历史不代表未来", ha="right", fontsize=8, color="gray")
    W = {f: w for f, w in winners.items() if w}

    # 1) 5 股合计累计净收益
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for fam, w in W.items():
        s = pd.concat([pooled_series[(w["config"], p)] for p in PERIODS]).sort_index()
        ax.plot(pd.to_datetime(s.index), s.cumsum(), lw=1.3, label=f"{w['label']}（{w['config']}）")
    ax.axvspan(pd.to_datetime("2024-07-24"), pd.to_datetime("2026-09-24"), color="tab:blue", alpha=0.07, label="样本外")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("各策略族入选配置：5 只股票合计的累计做T 超额收益（历史费率，已扣费用和滑点）")
    ax.set_ylabel("元（> 0 才跑赢只持有）")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)
    foot(fig)
    fig.tight_layout()
    fig.savefig(OUT / "equity_pooled.png", dpi=120)
    plt.close(fig)

    # 2) 纯毛收益 → 滑点 → 费用 → 净收益
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, label, suf in zip(axes, PERIODS, ["is", "oos"]):
        labs = [W[f]["label"] for f in W]
        rows = [wide[wide.config == W[f]["config"]].iloc[0] for f in W]
        x = np.arange(len(rows))
        ax.bar(x - 0.3, [r[f"pure_gross_{suf}"] for r in rows], 0.2, label="纯毛收益（未扣滑点和费用）", color="tab:green")
        ax.bar(x - 0.1, [-(r[f"pure_gross_{suf}"] - r[f"gross_{suf}"]) for r in rows], 0.2, label="滑点", color="tab:orange")
        ax.bar(x + 0.1, [-r[f"fees_{suf}"] for r in rows], 0.2, label="费用", color="tab:red")
        ax.bar(x + 0.3, [r[f"net_{suf}"] for r in rows], 0.2, label="净收益", color="tab:blue")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, labs, rotation=25, ha="right", fontsize=8.5)
        ax.set_title(f"{label}：5 股合计，收益分解（元）")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8.5)
    foot(fig)
    fig.tight_layout()
    fig.savefig(OUT / "gross_vs_net.png", dpi=120)
    plt.close(fig)

    # 3) 热力图：族 × 股票 样本外净收益
    fams = list(W)
    mat = np.array([[per[(per.config == W[f]["config"]) & (per.code == c) & (per.period == "样本外")].net.iloc[0]
                     for c in codes] for f in fams])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    vmax = np.abs(mat).max()
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(codes)), [f"{names[c]}\n{c}" for c in codes], fontsize=8.5)
    ax.set_yticks(range(len(fams)), [W[f]["label"] for f in fams], fontsize=9)
    for i in range(len(fams)):
        for j in range(len(codes)):
            ax.text(j, i, f"{mat[i, j]:,.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="样本外净超额收益（元）")
    ax.set_title("各族入选配置：样本外净超额收益（绿 = 跑赢只持有）")
    foot(fig)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_oos.png", dpi=120)
    plt.close(fig)

    # 4) 全部配置：样本内 vs 样本外
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for fam, g in wide.groupby("family"):
        ax.scatter(g.net_is, g.net_oos, s=30, alpha=0.75, label=grid["families"][fam]["label"])
    for fam, w in W.items():
        r = wide[wide.config == w["config"]].iloc[0]
        ax.scatter(r.net_is, r.net_oos, s=160, facecolors="none", edgecolors="black", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("样本内净收益（元，5 股合计）")
    ax.set_ylabel("样本外净收益（元，5 股合计）")
    ax.set_title(f"全部 {len(wide)} 组配置：样本内 vs 样本外（黑圈 = 各族入选）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    foot(fig)
    fig.tight_layout()
    fig.savefig(OUT / "is_vs_oos_all.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
