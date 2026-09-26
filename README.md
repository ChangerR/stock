# tlab：A 股底仓做T 研究框架

> **非投资建议，历史不代表未来。** 本仓库只做历史数据研究：没有任何下单、券商接口或实盘功能。
> 仓库是公开的，**不要提交任何个人持仓、成本价、账户信息**（`private/`、`*.local.yaml` 已加入 `.gitignore`）。

「做T」指在已有底仓的前提下做日内回转：**倒T** 是先卖底仓、再买回；**正T** 是先买入、再卖出底仓。
收盘时持股数回到底仓，赚取的是日内差价。本框架回答的问题只有一个：**扣除真实费用和滑点后，做T 是否跑赢只持有底仓？**

研究结论：
- 第一期（VWAP 偏离带做T、趋势过滤、多股票检验）：[`reports/README.md`](reports/README.md)
- 第二期（7 个做T 策略族普查）：[`reports/survey/README.md`](reports/survey/README.md)

**两期的共同结论：在测试过的股票和规则下，扣费后没有任何做T 方式稳健地跑赢只持有底仓。**

## 目录结构

```
src/tlab/
  data/sources.py      数据源接口（DataSource 协议），默认 baostock，可插拔
  data/store.py        本地缓存（data/*.csv.gz + MANIFEST.json）、预处理、数据质量检查
  market.py            板块规则：涨跌幅（主板 10%、ST 5%、创业板/科创板 20%，创业板 2020-08-24 前为 10%）、申报数量
  fees.py              费用（佣金+最低 5 元、印花税仅卖出、过户费，支持按日期分段）与滑点
  engine.py            日内引擎：无未来函数、T+1、封板拒单、强平、跨夜回补、相对底仓逐日盯市
  strategies/          版本化策略（注册表）：vwap_band@v1 为原型移植，vwap_band_regime@v1 为趋势过滤研究，
                       families.py 为第二期的 7 个策略族（网格、开盘区间突破、跳空、跨日均值回归、时刻效应、VWAP 出场改造、隔夜）
  features.py          日级特征（日线 ATR%、均线锚、昨日 VWAP 锚），均只用前一日及更早的数据
  metrics.py           指标、日类型归因、分年度、自助法置信区间
  analysis.py          信号事件研究（触发后价格是回归还是延续）
  report.py            Markdown 报告与图表
  config.py / runner.py / cli.py
configs/               YAML 配置（支持 extends 继承）
data/                  缓存的行情快照（gzip CSV）与 MANIFEST.json
scripts/               一次性研究脚本：选股筛选、过滤实验、多股票检验、原型数据导入
reports/               研究报告（中文）与图表
tests/                 引擎记账、规则、无未来函数、原型复现的测试
```

## 安装

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                          # 全部测试，约 5 秒
```

版本已在 `requirements.txt` / `pyproject.toml` 中锁定。图表中文字体会自动查找 Noto Sans CJK / 苹方 / 微软雅黑；
Linux 上可以用 `apt install fonts-noto-cjk` 安装。

## 常用命令

```bash
tlab run configs/baseline_688981.yaml            # 复现原型（原型费率口径）→ reports/runs/baseline_688981/
tlab run configs/baseline_688981_histfees.yaml   # 修正为历史真实费率的基线（研究主口径）
tlab grid configs/baseline_688981.yaml           # 样本内/外参数网格（k × s）
tlab fetch sz.300059 --start 2020-07-01 --end 2026-09-24   # 下载新股票
tlab check sh.688981                             # 数据质量摘要

python scripts/regime_experiment.py              # 倒T 趋势过滤实验（按 reports/preregistration.md）
python scripts/screen_universe.py                # 多股票检验的选股筛选（只用样本内日线）
python scripts/multistock.py configs/multistock.yaml
python scripts/survey.py                         # 第二期策略族普查（按 reports/survey/preregistration.md）
python scripts/survey_costs.py                   # 成本敏感性（事后探索）
```

## 新增一只股票或一个策略

**新股票**：先 `tlab fetch <代码> --start ... --end ...`，再写一个配置，只写与父配置不同的部分：

```yaml
extends: baseline_688981_histfees.yaml
name: my_test
tickers:
  - {code: sz.300059, base_shares: 1400, lot: 700}   # 主板/创业板须为 100 股整数倍；科创板 >= 200 股
```

**新策略**：在 `src/tlab/strategies/` 新建模块，继承 `Strategy`（或已有策略），用 `@register` 注册，并在 `strategies/__init__.py` 里 import：

```python
@register
class MyStrategy(Strategy):
    name, version = "my_strategy", "1"
    defaults = dict(lot=200, x=1.0)

    def daily_features(self, md):          # 可选：日级特征，第 t 行只能用 t 日开盘前已知的数据
        ...
    def on_bar(self, ctx):                 # ctx.close / ctx.vwap 等只包含第 0..i 根 K 线
        if ...:
            return [Order("sell", self.p["lot"], "open_dao")]
```

T+1、申报数量、涨跌停封板、强平、跨夜回补、费用和滑点都由引擎处理，策略不需要也不应该自己处理。
策略改动后如果会改变结果，请**提升 `version`**，并在报告里注明版本，让旧结论始终可追溯。

## 引擎约定（重要）

| 项目 | 约定 |
|---|---|
| 信号与成交 | 第 i 根 K 线收盘产生信号，第 i+1 根开盘价 ± 滑点成交（`fill: close` 为乐观对照模式） |
| T+1 | 当日可卖额度 = 开盘持股数，只减不增；当日买入的股份当日不可卖 |
| 涨跌停 | 按板块和 ST 状态计算涨跌停价（四舍五入到分）；整根 K 线封死涨停时买单拒绝，封死跌停时卖单拒绝；成交价被限制在涨跌停价以内 |
| 申报数量 | 主板/创业板买入为 100 股整数倍；科创板买入 >= 200 股；卖出余股须一次卖完 |
| 强平 | 日内策略在 `force_flat_time`（默认 14:50）收盘时自动回到底仓；封板无法成交则逐根重试，仍失败就持仓过夜，次日开盘回补（记为 CARRY） |
| 多日持有 | 策略参数 `max_hold_days = N` 时允许跨日偏离底仓，第 N 个交易日 14:50 强制回到底仓（记为 TIME） |
| 开盘前指令 | `on_day_start` 返回的指令以第一根 K 线开盘价成交（≈ 开盘集合竞价价格） |
| 盈亏 | 相对「只持有底仓」的超额收益，逐日盯市（用前收盘价，除权除息日不产生虚假盈亏）；基准 = 底仓 × Σ(收盘 − 前收)。毛收益 = 净收益 + 费用（跨除权日的闭环价差会失真，不用它） |
| 费用 | 佣金双向（最低 5 元），印花税只在卖出时收，过户费双向；费率可按日期分段（`historical` = 真实历史费率） |

已知简化：5 分钟 K 线看不到 K 线内部的价格路径；封板判断按整根 K 线（最高价 = 最低价 = 涨跌停价），实际排队成交情况无法得知；
假设现金充足（正T 需要预留买入资金）；不考虑券商个性化费率和交易所规则变化以外的其他制度差异。

## 数据

- 来源：[baostock](http://baostock.com) 免费公开接口，**不复权**（adjustflag=3）5 分钟线和日线。
- **选择把数据快照提交进仓库**（gzip 压缩后单只股票 6 年 5 分钟线约 1.4 MB），理由：
  1. **可复现**：免费数据源不稳定（原型阶段 akshare 的东方财富接口连接被拒，baostock 也限速），
     数据源也可能事后修订数据。提交快照后，任何人都能在同一份数据上得到逐笔相同的结果，测试也能离线运行。
  2. 体积可控：每只股票约 1.5 MB，只提交研究用到的股票，不提交全市场数据。
  3. `data/MANIFEST.json` 记录了每个文件的来源、日期区间、行数和 sha256。更新数据时 diff 一目了然。
  - 代价：仓库会随股票数量增长。如果将来覆盖几十只以上的股票，建议改用 Git LFS 或只提交 MANIFEST，并用 `tlab fetch` 按需重建。
  - 数据版权属于数据提供方；这里只以研究复现为目的保存原始快照。
- 688981 快照来自原型附带的 CSV，已与 baostock 现拉数据核对：日线 1505 行完全一致，
  抽查 4 个月的 5 分钟线（3,408 根）完全一致（`scripts/import_prototype_data.py`）。
- 筛选用的日线缓存放在 `data/cache/`（不入库）。

## 许可

代码使用 MIT 许可。数据版权归原提供方所有。
