import pytest

from tlab.fees import FeeSchedule, Slippage
from tlab.market import board_of, limit_prices, round_price, rules_for


def test_commission_minimum_and_stamp_only_on_sell():
    f = FeeSchedule(commission=0.00025, commission_min=5.0, stamp_duty=0.0005, transfer=0.00001)
    # 200 股 × 50 元 = 10000：佣金 2.5 → 最低 5；过户费 0.1；印花税 5（仅卖）
    assert f.cost("buy", 50.0, 200) == pytest.approx(5.0 + 0.1)
    assert f.cost("sell", 50.0, 200) == pytest.approx(5.0 + 0.1 + 5.0)
    # 大额：100 万 → 佣金 250
    assert f.cost("buy", 100.0, 10000) == pytest.approx(250 + 10)


def test_dated_rates_historical():
    f = FeeSchedule(stamp_duty="historical", transfer="historical")
    assert f.cost("sell", 100, 1000, "2023-08-25") == pytest.approx(25 + 1 + 100)   # 印花 0.1%，过户已是 0.001%
    assert f.cost("sell", 100, 1000, "2022-04-28") == pytest.approx(25 + 2 + 100)   # 过户 0.002%
    assert f.cost("sell", 100, 1000, "2023-08-28") == pytest.approx(25 + 1 + 50)    # 印花 0.05%，过户 0.001%
    assert f.cost("buy", 100, 1000, "2022-04-28") == pytest.approx(25 + 2)
    assert f.cost("buy", 100, 1000, "2022-04-29") == pytest.approx(25 + 1)


def test_slippage_min_tick_and_disabled():
    s = Slippage(pct=0.0002, min_ticks=1)
    assert s.apply(10.00, "buy") == 10.01            # 0.002 < 1 tick
    assert s.apply(120.00, "sell") == 119.98          # 0.024 → 0.02
    assert Slippage(pct=0.0, min_ticks=0).apply(10.0, "buy") == 10.0


def test_boards_and_limits():
    assert board_of("sh.688981") == "star"
    assert board_of("sz.300059") == "chinext"
    assert board_of("sh.600000") == board_of("sz.002230") == "main"
    with pytest.raises(ValueError):
        board_of("bj.430047")
    assert rules_for("sh.600000", "2024-01-02").limit_pct == 0.10
    assert rules_for("sh.600000", "2024-01-02", is_st=True).limit_pct == 0.05
    assert rules_for("sh.688981", "2024-01-02", is_st=True).limit_pct == 0.20
    assert rules_for("sz.300059", "2020-08-21").limit_pct == 0.10     # 注册制改革前
    assert rules_for("sz.300059", "2020-08-24").limit_pct == 0.20
    assert limit_prices(10.05, 0.10) == (11.06, 9.05)                  # 11.055 → 11.06（四舍五入）
    assert round_price(2.675) == 2.68


def test_lot_rules():
    star, main = rules_for("sh.688981", "2024-01-02"), rules_for("sh.600000", "2024-01-02")
    assert star.valid_buy_qty(200) and star.valid_buy_qty(201) and not star.valid_buy_qty(199)
    assert main.valid_buy_qty(100) and not main.valid_buy_qty(150)
    assert main.valid_sell_qty(150, 150)          # 余股一次性卖出
    assert not main.valid_sell_qty(150, 400)
    assert not star.valid_sell_qty(100, 400) and star.valid_sell_qty(100, 100)
