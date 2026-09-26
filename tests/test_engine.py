import pytest

from helpers import Scripted, make_md
from tlab.engine import Engine, EngineConfig

STAR = "sh.688981"
MAIN = "sh.600000"


def flat(px, n=48):
    return [(px, px, px, px)] * n


def run(md, script, base=400, **ecfg):
    s = Scripted(script=script)
    res = Engine(EngineConfig(base_shares=base, **ecfg)).run(md, s)
    return res, s


def test_fill_next_open_with_slippage_and_fees():
    bars = flat(100.0)
    bars[5] = (101.0, 101.0, 101.0, 101.0)
    md = make_md(STAR, [dict(date="2024-01-02", preclose=100.0, bars=bars)])
    res, s = run(md, {("2024-01-02", 955): [("sell", 200, "open_dao")],
                      ("2024-01-02", 1030): [("buy", 200, "TP")]})
    f1, f2 = s.fills
    assert (f1.sig_time, f1.fill_time, f1.price) == (955, 1000, 100.98)   # 下一根开盘 101 - 滑点 0.02
    assert (f2.sig_time, f2.fill_time, f2.price) == (1030, 1035, 100.02)
    sell_fee = max(5, 100.98 * 200 * 0.00025) + 100.98 * 200 * (0.00001 + 0.0005)
    buy_fee = max(5, 100.02 * 200 * 0.00025) + 100.02 * 200 * 0.00001
    tr = res.trips.iloc[0]
    assert tr.dir == "倒T" and tr.reason == "TP"
    assert tr.gross == pytest.approx((100.98 - 100.02) * 200)
    assert tr.fees == pytest.approx(sell_fee + buy_fee)
    assert res.daily.pnl.sum() == pytest.approx(tr.net)
    assert res.daily.shares_eod.iloc[-1] == 400


def test_t_plus_one_sellable_quota():
    md = make_md(STAR, [dict(date="2024-01-02", preclose=100.0, bars=flat(100.0))])
    d = "2024-01-02"
    res, s = run(md, {(d, 955): [("buy", 200, "open_zheng")],
                      (d, 1000): [("sell", 600, "bad")],        # 超过开盘持股 400
                      (d, 1005): [("sell", 200, "TP")],
                      (d, 1010): [("sell", 400, "bad2")]})      # 可卖额度只剩 200
    assert [r[0] for r in s.rejects] == ["bad", "bad2"]
    assert all("T+1" in r[1] for r in s.rejects)
    assert res.daily.shares_eod.iloc[-1] == 400
    assert len(res.trips) == 1 and res.trips.iloc[0].dir == "正T"


def test_forced_close_at_force_flat_time():
    md = make_md(STAR, [dict(date="2024-01-02", preclose=100.0, bars=flat(100.0))])
    res, s = run(md, {("2024-01-02", 955): [("sell", 200, "open_dao")]})
    tr = res.trips.iloc[0]
    assert (tr.reason, tr.exit_sig_t, tr.exit_fill_t) == ("EOD", 1450, 1455)
    assert res.daily.shares_eod.iloc[-1] == 400
    assert not tr.overnight


def test_sealed_limit_up_blocks_buyback_and_carries_overnight():
    # 主板 10%：前收 10 → 涨停 11.00。14:30 之后封死涨停，倒T 无法买回，次日开盘回补
    bars = flat(10.5, 32) + flat(11.0, 16)
    md = make_md(MAIN, [dict(date="2024-01-02", preclose=10.0, bars=bars, close=11.0),
                        dict(date="2024-01-03", preclose=11.0, bars=flat(11.5))])
    res, s = run(md, {("2024-01-02", 1000): [("sell", 500, "open_dao")]}, base=1000)
    assert res.daily.loc["2024-01-02", "shares_eod"] == 500
    assert any("涨停封板" in r.reason for r in res.rejects.itertuples())
    tr = res.trips.iloc[0]
    assert tr.overnight and tr.reason == "CARRY" and tr.date == "2024-01-03"
    assert tr.buy_px == 11.51 and tr.exit_fill_t == 935
    # 逐日盯市之和 == 闭环净收益
    assert res.daily.pnl.sum() == pytest.approx(tr.net)
    assert res.daily.loc["2024-01-02", "pnl"] == pytest.approx(500 * 10.49 - s.fills[0].fee - 500 * 11.0)


def test_fill_price_clamped_to_limit():
    bars = flat(10.5)
    bars[6] = (11.0, 11.0, 10.9, 10.95)     # 开在涨停但盘中打开：可以成交，价格不能超过 11.00
    md = make_md(MAIN, [dict(date="2024-01-02", preclose=10.0, bars=bars)])
    res, s = run(md, {("2024-01-02", 955): [("sell", 500, "open_dao")],
                      ("2024-01-02", 1000): [("buy", 500, "TP")]}, base=1000)
    assert s.fills[1].price == 11.0


def test_lot_size_rules():
    d = "2024-01-02"
    md = make_md(STAR, [dict(date=d, preclose=100.0, bars=flat(100.0))])
    _, s = run(md, {(d, 955): [("buy", 150, "small")]})
    assert s.rejects == [("small", "申报数量不合规(买)")]
    md = make_md(MAIN, [dict(date=d, preclose=10.0, bars=flat(10.0))])
    _, s = run(md, {(d, 955): [("sell", 150, "odd")]}, base=1000)
    assert s.rejects == [("odd", "申报数量不合规(卖)")]


def test_st_limit_is_5pct():
    md = make_md(MAIN, [dict(date="2024-01-02", preclose=10.0, bars=flat(10.2), isST=1)])
    assert md.days.lim_up.iloc[0] == 10.5 and md.days.lim_dn.iloc[0] == 9.5


def test_strategy_only_sees_past_bars():
    md = make_md(STAR, [dict(date="2024-01-02", preclose=100.0, bars=flat(100.0))])
    _, s = run(md, {})
    assert s.seen_len and all(n == i + 1 for i, n in s.seen_len)
    assert max(i for i, _ in s.seen_len) < 46      # 14:50 之后不再调用策略；最后一根不产生信号


def test_close_fill_mode_is_same_bar():
    bars = flat(100.0)
    bars[4] = (100.0, 100.5, 99.5, 100.4)
    md = make_md(STAR, [dict(date="2024-01-02", preclose=100.0, bars=bars)])
    _, s = run(md, {("2024-01-02", 955): [("sell", 200, "open_dao")]}, fill="close")
    assert s.fills[0].fill_time == 955 and s.fills[0].price == pytest.approx(100.38)


def _days(n, px=100.0):
    return [dict(date=f"2024-01-{2 + k:02d}", preclose=px, bars=flat(px)) for k in range(n)]


def test_multi_day_hold_and_time_stop():
    md = make_md(STAR, _days(4))
    s = Scripted(script={("2024-01-02", 1000): [("buy", 200, "open")]}, max_hold_days=2)
    res = Engine(EngineConfig(base_shares=400)).run(md, s)
    assert list(res.daily.shares_eod) == [600, 600, 400, 400]     # 第 2 个交易日 14:50 强制回到底仓
    tr = res.trips.iloc[0]
    assert (tr.reason, tr.entry_date, tr.date, tr.overnight) == ("TIME", "2024-01-02", "2024-01-04", True)
    assert res.daily.pnl.sum() == pytest.approx(tr.net)


def test_preopen_order_fills_at_first_bar_open():
    bars = flat(100.0)
    bars[0] = (99.0, 100.0, 99.0, 100.0)
    md = make_md(STAR, [dict(date="2024-01-02", preclose=100.0, bars=bars)])
    s = Scripted(preopen={"2024-01-02": [("sell", 200, "pre")]})
    Engine(EngineConfig(base_shares=400)).run(md, s)
    assert (s.fills[0].sig_time, s.fills[0].fill_time, s.fills[0].price) == (0, 935, 98.98)


def test_fill_crossing_base_splits_round_trips():
    d = "2024-01-02"
    md = make_md(STAR, [dict(date=d, preclose=100.0, bars=flat(100.0))])
    res, s = run(md, {(d, 955): [("buy", 200, "open_zheng")], (d, 1000): [("sell", 400, "flip")]})
    assert list(res.trips.dir) == ["正T", "倒T"] and list(res.trips.qty) == [200, 200]
    assert res.trips.fees.sum() == pytest.approx(sum(f.fee for f in s.fills))
    assert res.daily.pnl.sum() == pytest.approx(res.trips.net.sum())
