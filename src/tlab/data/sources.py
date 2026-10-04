"""可插拔数据源。新增数据源只需实现 DataSource 的两个方法并注册到 SOURCES。

统一输出格式（全部为不复权原始价格，字符串日期）：
  分钟线: date, time(YYYYMMDDHHMMSSsss, K 线结束时刻), open, high, low, close, volume(股), amount(元)
  日线:   date, open, high, low, close, preclose, volume, amount, pctChg, tradestatus, isST
"""
from __future__ import annotations

import time
from typing import Protocol

import pandas as pd

MINUTE_FIELDS = ["date", "time", "open", "high", "low", "close", "volume", "amount"]
DAILY_FIELDS = ["date", "open", "high", "low", "close", "preclose", "volume", "amount",
                "pctChg", "tradestatus", "isST"]


class DataSource(Protocol):
    name: str

    def fetch_minute(self, code: str, start: str, end: str, freq: str) -> pd.DataFrame: ...

    def fetch_daily(self, code: str, start: str, end: str) -> pd.DataFrame: ...


class BaostockSource:
    """baostock 免费公开接口（http://baostock.com）。分钟线按月分段下载，失败自动重试。"""

    name = "baostock"

    def __init__(self, retries: int = 6, verbose: bool = True):
        import baostock as bs

        self.bs = bs
        self.retries = retries
        self.verbose = verbose
        self._logged_in = False

    def _login(self):
        if not self._logged_in:
            lg = self.bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"baostock 登录失败: {lg.error_code} {lg.error_msg}")
            self._logged_in = True

    def close(self):
        if self._logged_in:
            self.bs.logout()
            self._logged_in = False

    def _query(self, code, fields, start, end, freq):
        self._login()
        last = None
        for attempt in range(self.retries):
            rs = self.bs.query_history_k_data_plus(code, ",".join(fields), start_date=start, end_date=end,
                                                   frequency=freq, adjustflag="3")
            rows = []
            ok = rs.error_code == "0"
            while ok and rs.next():
                rows.append(rs.get_row_data())
            if ok and rs.error_code == "0":
                return pd.DataFrame(rows, columns=fields)
            last = f"{rs.error_code} {rs.error_msg}"
            if self.verbose:
                print(f"  重试 {code} {start}..{end}: {last}", flush=True)
            time.sleep(2 * (attempt + 1))
            self.close()
            self._login()
        raise RuntimeError(f"baostock 下载失败 {code} {start}..{end}: {last}")

    def fetch_minute(self, code, start, end, freq="5"):
        parts = []
        for m in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
            s = max(pd.Timestamp(start), m).strftime("%Y-%m-%d")
            e = min(pd.Timestamp(end), m + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            df = self._query(code, MINUTE_FIELDS, s, e, freq)
            if self.verbose:
                print(f"  {code} {s[:7]} {len(df)} 根", flush=True)
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=MINUTE_FIELDS)

    def fetch_daily(self, code, start, end):
        return self._query(code, DAILY_FIELDS, start, end, "d")


SOURCES = {"baostock": BaostockSource}


def get_source(name: str, **kw) -> DataSource:
    try:
        return SOURCES[name](**kw)
    except KeyError:
        raise ValueError(f"未知数据源 {name}，可选: {list(SOURCES)}") from None
