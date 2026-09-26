"""多股票检验的选股筛选（规则事先固定，只用样本内日线，避免看样本外结果挑股票）。

候选池（尽量减少幸存者偏差）：
  - 沪深300 在 2020-06-29 的成分股（baostock query_hs300_stocks(date=2020-07-01)）
  - 以及 2020-06-30 之前上市的全部科创板股票（含之后退市的）
窗口：样本内 2020-07-23 ~ 2024-07-23 的不复权日线。
入选条件：
  1. 窗口内可交易天数 >= 95%，窗口内从未 ST；窗口首日收盘价 >= 10 元（一个 tick <= 0.1%，与 688981 可比）
  2. 「区间震荡」：窗口首尾收盘价涨跌幅绝对值 <= 50%（|ln(P_end/P_start)| <= ln 1.5，价格用复权因子无关的前收盘链计算）
  3. 打分 = 日内振幅中位数 (high-low)/preclose 的百分位排名（越高越好）
          + 日内效率中位数 |close-open|/(high-low) 的百分位排名（越低越好，代表盘中来回震荡而非单边）
  4. 每个板块取得分最高者：科创板 1 只（排除 688981 本身）、创业板 1 只、主板 2 只（沪、深各 1 只）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tlab.data.sources import BaostockSource
from tlab.market import board_of

IS_START, IS_END = "2020-07-23", "2024-07-23"
OUT = Path(__file__).resolve().parents[1] / "reports" / "universe_screen.csv"

src = BaostockSource(verbose=False)
src._login()
bs = src.bs
rs = bs.query_hs300_stocks(date="2020-07-01")
hs = []
while rs.next():
    hs.append(rs.get_row_data())
rs = bs.query_stock_basic()
basic = []
while rs.next():
    basic.append(rs.get_row_data())
basic = pd.DataFrame(basic, columns=rs.fields)
names = dict(zip(basic.code, basic.code_name))
star = basic[basic.code.str.startswith("sh.688") & (basic.ipoDate < "2020-06-30") & (basic.type == "1")].code
codes = sorted(set([r[1] for r in hs]) | set(star))
codes = [c for c in codes if c != "sh.688981"]
print(f"候选 {len(codes)} 只 (沪深300@2020-06-29: {len(hs)}, 科创板早期上市: {len(star)})", flush=True)

rows = []
for i, c in enumerate(codes):
    try:
        board = board_of(c)
    except ValueError:
        continue
    try:
        d = src.fetch_daily(c, IS_START, IS_END)
    except RuntimeError as e:
        print("失败", c, e, flush=True)
        rows.append(dict(code=c, name=names.get(c, ""), board=board, error=str(e)))
        continue
    for col in ["open", "high", "low", "close", "preclose", "tradestatus", "isST"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    n_all = len(d)
    t = d[(d.tradestatus == 1) & (d.volume.astype(str) != "") & (d.high > d.low)]
    if n_all == 0 or len(t) < 50:
        rows.append(dict(code=c, name=names.get(c, ""), board=board, error="数据不足"))
        continue
    drift = float(np.log(t.close / t.preclose).sum())  # 前收盘链：对除权除息不敏感
    rows.append(dict(
        code=c, name=names.get(c, ""), board=board, days=len(t), trade_frac=len(t) / n_all,
        ever_st=int(d.isST.max()), px_start=float(t.close.iloc[0]),
        is_drift=float(np.expm1(drift)),
        range_med=float(((t.high - t.low) / t.preclose).median()),
        eff_med=float(((t.close - t.open).abs() / (t.high - t.low)).median()),
    ))
    if i % 25 == 0:
        print(f"  {i}/{len(codes)}", flush=True)
src.close()

df = pd.DataFrame(rows)
ok = df.error.isna() if "error" in df else pd.Series(True, index=df.index)
elig = df[ok & (df.trade_frac >= 0.95) & (df.ever_st == 0) & (df.px_start >= 10)
          & (df.is_drift.abs() <= 0.5)].copy()
elig["score"] = elig.range_med.rank(pct=True) + (-elig.eff_med).rank(pct=True)
df = df.merge(elig[["code", "score"]], on="code", how="left").sort_values("score", ascending=False)
df["eligible"] = df.score.notna()

picks = []
e = df[df.eligible]
picks += e[e.board == "star"].head(1).code.tolist()
picks += e[e.board == "chinext"].head(1).code.tolist()
picks += e[(e.board == "main") & e.code.str.startswith("sh.")].head(1).code.tolist()
picks += e[(e.board == "main") & e.code.str.startswith("sz.")].head(1).code.tolist()
df["picked"] = df.code.isin(picks)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, float_format="%.5f")
print(df[df.eligible].head(20).to_string())
print("入选:", picks)
sys.exit(0)
