"""一次性脚本：导入原型附带的 688981 数据，并与 baostock 现拉数据抽样核对。

- 5 分钟线：直接使用原型 CSV（2020-07-16 ~ 2026-09-24），抽取若干月份从 baostock 重新下载逐根比对。
- 日线：原型 CSV 缺 isST 字段，从 baostock 重新下载，并与原型日线逐行比对。
"""
import sys
from pathlib import Path

import pandas as pd

from tlab.data.sources import BaostockSource
from tlab.data.store import DEFAULT_DATA_DIR, _path, _update_manifest

CODE = "sh.688981"
up = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cursor/projects/workspace/uploads"
m = pd.read_csv(up / "688981_5min_5a0a.csv", dtype=str)
d_old = pd.read_csv(up / "688981_daily_585a.csv", dtype=str)

src = BaostockSource(verbose=False)
d_new = src.fetch_daily(CODE, "2020-07-16", "2026-09-24")
cols = ["date", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus"]
num = [c for c in cols if c != "date"]
a = d_old[cols].set_index("date")[num].apply(pd.to_numeric, errors="coerce")
b = d_new[cols].set_index("date")[num].apply(pd.to_numeric, errors="coerce")
print(f"日线: 原型 {len(a)} 行, baostock {len(b)} 行, 日期一致: {a.index.equals(b.index)}")
print("日线最大绝对差(各列):", (a - b.reindex(a.index)).abs().max().to_dict())

bad = 0
for s, e in [("2020-07-01", "2020-07-31"), ("2022-03-01", "2022-03-31"), ("2024-10-01", "2024-10-31"),
             ("2026-09-01", "2026-09-24")]:
    fresh = src._query(CODE, list(m.columns), s, e, "5").set_index("time")
    old = m[(m.date >= s) & (m.date <= e)].set_index("time")
    diff = (old[["open", "high", "low", "close", "volume", "amount"]].astype(float)
            - fresh.reindex(old.index)[["open", "high", "low", "close", "volume", "amount"]].astype(float)).abs()
    n = int((diff.max(axis=1) > 1e-6).sum())
    bad += n
    print(f"5分钟抽查 {s[:7]}: 原型 {len(old)} 根, baostock {len(fresh)} 根, 不一致 {n} 根")
src.close()

for freq, df in [("5", m), ("d", d_new)]:
    p = _path(DEFAULT_DATA_DIR, CODE, freq)
    df.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
    _update_manifest(DEFAULT_DATA_DIR, p, df, "baostock")
    print("写入", p, len(df))
print("抽查不一致总数:", bad)
