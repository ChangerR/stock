"""报告：Markdown 表格、逐笔明细、图表。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .engine import BacktestResult  # noqa: E402
from .metrics import by_daytype, by_reason, by_year, summary, to_md  # noqa: E402

DISCLAIMER = "> **非投资建议，历史不代表未来。** 本仓库仅做历史数据研究，不含任何下单或券商接口。"

_CJK = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc", "C:/Windows/Fonts/msyh.ttc"]


def setup_fonts() -> bool:
    for fp in _CJK:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            name = font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def concat_daily(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """把各时段的逐日结果按时间拼接成一条连续曲线（各时段独立回测，起点均为空仓）。"""
    dl = pd.concat([r.daily for r in results.values()]).sort_index()
    dl = dl[~dl.index.duplicated()]
    dl["cum_pnl"] = dl.pnl.cumsum()
    dl["cum_bench"] = dl.bench.cumsum()
    return dl


def plot_equity(curves: dict[str, pd.DataFrame], path: Path, title: str, oos_start: str | None = None,
                show_hold: bool = True) -> None:
    """上：各方案累计做T 超额收益 + 股价；下：只持有 vs 持有+做T（取第一条曲线）。"""
    setup_fonts()
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    first = next(iter(curves.values()))
    x0 = pd.to_datetime(first.index)
    for label, dl in curves.items():
        ax1.plot(pd.to_datetime(dl.index), dl.cum_pnl, lw=1.4, label=label)
    ax1.axhline(0, color="gray", lw=0.8)
    if oos_start:
        ax1.axvspan(pd.to_datetime(oos_start), x0[-1], color="tab:blue", alpha=0.07, label="样本外")
    ax1.set_ylabel("累计做T 超额收益（元，已扣费用与滑点）")
    ax2 = ax1.twinx()
    ax2.plot(x0, first.close, color="gray", lw=0.7, alpha=0.5, label="收盘价（右轴）")
    ax2.set_ylabel("股价（元）")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    if show_hold:
        ax3.plot(x0, first.cum_bench, color="gray", lw=1.1, label="只持有底仓 盈亏")
        for label, dl in curves.items():
            ax3.plot(pd.to_datetime(dl.index), dl.cum_bench + dl.cum_pnl, lw=0.8, alpha=0.85,
                     label=f"持有 + {label}")
        ax3.axhline(0, color="gray", lw=0.6)
        ax3.legend(loc="upper left", fontsize=8)
        ax3.set_ylabel("元")
        ax3.grid(alpha=0.3)
    fig.text(0.99, 0.005, "非投资建议，历史不代表未来", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_run_report(name: str, description: str, results: dict[str, dict[str, BacktestResult]],
                     outdir: Path, meta: dict | None = None) -> Path:
    """标准回测报告：每只股票的指标表、分解表、逐笔 CSV、逐日 CSV 和图。"""
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [f"# 回测报告：{name}", "", DISCLAIMER, "", description.strip(), "",
             f"- 代码版本：`{git_rev()}`"]
    if meta:
        lines.append(f"- 配置：`{json.dumps(meta, ensure_ascii=False, default=str)}`")
    lines.append("")
    for code, per in results.items():
        any_r = next(iter(per.values()))
        lines += [f"## {code}  （策略 `{any_r.strategy_id}`）", "",
                  f"参数：`{json.dumps(any_r.params, ensure_ascii=False)}`", ""]
        st = pd.DataFrame({k: summary(v) for k, v in per.items()})
        st.index.name = "指标"
        lines += [to_md(st.reset_index()), ""]
        for label, r in per.items():
            lines += [f"### 按日类型（事后归因，{label}）", "", to_md(by_daytype(r)), "",
                      f"### 按方向与平仓原因（{label}）", "", to_md(by_reason(r)), ""]
        allr = concat_daily(per)
        fake = BacktestResult(pd.concat([r.trips for r in per.values()]), allr, pd.DataFrame(), pd.DataFrame(),
                              any_r.strategy_id, any_r.params, code)
        lines += ["### 分年度", "", to_md(by_year(fake)), ""]
        safe = code.replace(".", "")
        fake.trips.to_csv(outdir / f"trades_{safe}.csv", index=False, float_format="%.4f")
        allr.to_csv(outdir / f"daily_{safe}.csv", float_format="%.4f")
        labels = list(per)
        oos = per[labels[1]].daily.index[0] if len(labels) > 1 else None
        png = outdir / f"equity_{safe}.png"
        plot_equity({any_r.strategy_id: allr}, png, f"{code} {any_r.strategy_id} 累计做T超额收益", oos)
        lines += [f"![{code}](equity_{safe}.png)", ""]
    p = outdir / "README.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
