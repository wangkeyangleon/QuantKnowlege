# A股量化交易系统 — 1万元实盘模拟

## Context

用户要求用1万元初始资金在A股市场运行量化交易系统，需要独立学习、试错、生存，并每日汇报操作与盈亏。A股市场特点：T+1、不可做空、涨跌停±10%、最低100股/手、千分之一印花税(卖)、佣金约万2.5。

本项目是全新项目，无现有代码。需要从零搭建完整系统。

## 架构设计

```
AiQuant/
├── config.py              # 配置：资金、佣金、标的池
├── data_engine.py         # 数据层：akshare 拉取A股日线数据
├── signal_engine.py       # 信号层：多因子 + 趋势信号生成
├── risk_engine.py         # 风控层：仓位管理、止损止盈
├── execution_engine.py    # 执行层：模拟成交
├── portfolio.py           # 组合层：持仓跟踪、盈亏计算
├── daily_report.py        # 日报：Markdown格式操作与盈亏汇报
├── main.py                # 主入口：每日运行一次
├── scheduler.py           # 调度器：定时触发
└── archive/               # 历史日报归档
```

## 策略设计

### 1万元A股约束
- 标的池：中证1000成分股，股价 5~50元（100股起买）
- 最大持仓：3只股票
- 最低现金保留：15%
- 单笔仓位上限：3000元（~30%）

### 核心策略：趋势动量混合

**信号生成（3维度等权）：**
1. **双均线趋势**（日线）：MA5 > MA20 且 MA20斜率向上 → +1
2. **通道突破**：价格突破20日最高 → +1  
3. **RSI动量**：RSI(14) 在 40-70 区间且上升 → +1
- 信号总分 ≥2 触发买入；总分 ≤0 触发卖出

**选股逻辑：**
- 从标的池中筛选满足买入信号的股票
- 按成交额排序（流动性优先），取前5只候选
- 如果已有持仓未到卖出条件，优先加仓而非新开仓

**止损止盈：**
- 固定止损：-5%（单笔最大亏损150元）
- 移动止盈：从最高点回落3%
- 时间止损：持仓15日未盈利则清仓

## 自适应学习机制

1. **参数网格搜索**：每周末对过去60日数据做参数回测，选择夏普最高的参数组合
   - MA参数：{(5,20), (10,30), (5,30), (10,50)}
   - 止损比例：{3%, 5%, 7%}
2. **简单强化**：统计每种参数组合的近期胜率，倾向使用高胜率组合
3. **市场状态识别**：
   - 趋势市（ADX > 25）：使用趋势策略
   - 震荡市（ADX < 20）：降低仓位，提高入场门槛

## 日报系统

每天收盘后生成日报 `archive/report_YYYY-MM-DD.md`：

```markdown
# 量化交易日报 — 2026-06-07

## 账户概览
| 指标 | 数值 |
|------|------|
| 总资产 | ¥X,XXX |
| 现金 | ¥X,XXX |
| 持仓市值 | ¥X,XXX |
| 累计收益率 | +X.X% |
| 最大回撤 | -X.X% |
| 夏普比率 | X.XX |

## 今日操作
| 时间 | 操作 | 标的 | 价格 | 数量 | 金额 | 原因 |
|------|------|------|------|------|------|------|
| 09:35 | 买入 | 000XXX | ¥XX | 100 | ¥X,XXX | 双均线+RSI信号 |

## 当前持仓
| 代码 | 名称 | 成本价 | 现价 | 数量 | 市值 | 盈亏 | 持仓天数 |
|------|------|--------|------|------|------|------|----------|

## 策略状态
- 当前参数：MA(5,20), 止损-5%
- 市场状态：趋势市 (ADX=28)
- 下次参数优化：周六
```

## 定时调度

通过 `anthropic-skills:schedule` skill 创建定时任务，每个交易日 15:30（收盘后）自动运行 `main.py`。

## 实现步骤

### Step 1: `config.py` — 配置文件
- 初始资金 INITIAL_CAPITAL = 10000
- 佣金率、印花税率、过户费
- 股票池参数
- 策略参数默认值

### Step 2: `data_engine.py` — 数据引擎
- 使用 akshare 获取A股日K线（如前复权）
- 获取中证1000成分股列表
- 缓存数据到本地减少API调用

### Step 3: `signal_engine.py` — 信号引擎
- 计算双均线、通道突破、RSI指标
- 生成买入/卖出信号
- 选股排序逻辑

### Step 4: `risk_engine.py` — 风控引擎
- 计算仓位大小
- 止损止盈检查
- 资金分配

### Step 5: `execution_engine.py` — 执行引擎
- 模拟交易（以当日收盘价成交）
- 考虑涨跌停无法交易的情况

### Step 6: `portfolio.py` — 组合管理
- 持仓跟踪
- 盈亏计算
- 历史记录持久化（JSON）

### Step 7: `daily_report.py` — 日报生成
- 生成Markdown格式日报
- 保存到 archive/

### Step 8: `main.py` — 主流程
- 串联所有模块
- 每日执行：拉数据→算信号→做决策→记录→出报告

### Step 9: 定时任务配置
- 使用 schedule skill 创建交易日定时任务

## 验证方案

1. **单元测试**：测试各引擎的核心逻辑
2. **回测验证**：用历史数据跑一遍，确认信号、成交、盈亏计算正确
3. **日报确认**：检查生成的 report_*.md 内容完整
4. **模拟运行**：连续运行几天，观察系统行为是否符合预期

## 风险提示

此系统为**模拟交易系统**，不连接真实券商。所有交易以收盘价模拟成交。实际交易中需考虑滑点、流动性、涨跌停封单等因素。


# 三层筛选架构 — 实施计划

## Context

当前系统是单层量化信号筛选（MA金叉 + 通道突破 + RSI动量 → 得分排序 → 选前5只）。用户要求升级为三层渐进式筛选：

```
Layer 1: 量化信号          Layer 2: 机器学习预测        Layer 3: DeepSeek AI 终审
(5000+ 只)                 (50 只候选)                 (10 只候选)
    ↓                         ↓                         ↓
MA+Donchian+RSI 扫描   →   RandomForest 预测次日   →   DeepSeek 综合分析
成交额排序取Top 50        收益概率 > 阈值留下        市场+基本面+情绪
                                                         → 最终 0-5 只买入
```

## 新增文件

### 1. `ml_engine.py` — 机器学习预测引擎

**模型选择**: `RandomForestClassifier` (scikit-learn)
- 优点：无需GPU，训练快，特征重要性可解释，不易过拟合
- 目标：二分类 — 次日涨 vs 跌 (y=1 if pct_chg > 0)
- 每天收盘后增量训练，保留最近180日数据窗口

**特征工程** (每只股票每日≈25个特征):
| 类别 | 特征 |
|------|------|
| 价格趋势 | MA偏离度(close/ma5-1, close/ma20-1)，MA斜率 |
| 动量 | 1/3/5/10/20日收益率，RSI(14) |
| 波动率 | ATR(14)/close，Bollinger带宽 |
| 成交量 | 量比(vol/vol_ma5)，量价相关性 |
| 突破类 | 距20日高点%，距20日低点% |
| 市场相对 | 相对大盘超额收益，Beta |

**核心接口**:
```python
train_model(kline_data: dict, days: int = 180) -> sklearn model   # 全量训练
predict_batch(model, candidates: list, kline_data: dict) -> list   # 批量预测 → 返回 [(code, prob, features)]
retrain_if_needed()                                                # 检查是否需要重训
```

**训练策略**:
- 标签: `y = sign(next_day_pct_chg)` (次日涨=1, 跌=0)
- 每周六凌晨自动重训，模型保存到 `.cache/ml_model.pkl`
- 训练数据: 全市场5527只 × 最近180天 ≈ 百万级样本

### 2. `ai_analyst.py` — DeepSeek AI 分析引擎

**API**: `https://api.deepseek.com/v1/chat/completions`
**模型**: `deepseek-chat` (性价比最优)

**输入构造** — 对每只候选股，发送结构化 prompt:
```
你是一位A股量化交易分析师。基于以下数据，判断该股票明日是否值得买入。
请给出: 买入/观望/卖出 + 置信度(0-100) + 简要理由。

股票: {name}({code})
现价: ¥{price}
量化信号: {signal_summary}  (得分{buy_score}/3, 强度{strength})
ML预测: 次日上涨概率 {ml_prob:.0%}

技术面:
- MA5/MA20: {ma_status}
- RSI(14): {rsi}
- 近5日涨幅: {ret5d:+.2f}%
- ADX: {adx} ({regime})

基本面:
- PE: {pe}, PB: {pb}
- 近4季ROE: {roe}

市场环境:
- 市场状态: {regime}
- 涨跌比: {breadth_up}%

请用 JSON 格式回复:
{"decision": "buy|hold|sell", "confidence": 0-100, "reason": "..."}
```

**批量处理**:
- Layer 2 筛选出 10 只候选 → 打包1-2次API调用（节约成本）
- 并行请求（异步）
- 根据返回的 decision 和 confidence 做最终排序

**成本控制**:
- 每次调用约消耗 2000-4000 tokens (输入+输出)
- deepseek-chat 定价极低（百万tokens仅¥1-2），每日成本 < ¥0.1
- 可只在 morning 和 evening 模式调用

### 3. 更新 `config.py` — 新增配置

```python
# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ML 模型
ML_MODEL_PATH = os.path.join(DATA_CACHE_DIR, "ml_model.pkl")
ML_TRAIN_DAYS = 180         # 训练窗口 (天)
ML_RETRAIN_INTERVAL = 7     # 重训间隔 (天)
ML_MIN_PROB_THRESHOLD = 0.55  # Layer 2 留存的最低预测概率

# 三层筛选配置
LAYER1_TOP_N = 50           # 量化信号筛选保留数
LAYER2_TOP_N = 10           # ML预测保留数
LAYER3_FINAL_N = 5          # AI最终决策最多买入数
```

### 4. 更新 `.env` — DeepSeek Key

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 流水线集成 (修改 main.py)

**新的 `run_morning()` / `run_evening()` 流程**:

```
1. 拉取行情数据 (现有)
2. 市场状态分析 (现有)
3. Layer 1 — 量化信号扫描:
   generate_signal() → rank_candidates() → 取Top 50
4. Layer 2 — ML预测过滤:
   ml_engine.predict_batch() → 过滤 prob > 0.55 → 取Top 10
5. Layer 3 — DeepSeek AI 终审:
   ai_analyst.analyze_batch() → 解析 JSON → 取 buy+高置信度 → Top 3-5
6. 风控检查 & 执行交易 (现有)
7. 日报中记录三层筛选详情
```

**日报增强** — 在 AI决策说明 中展示:
```
## 🧠 三层筛选详情

### Layer 1 — 量化信号 (50只候选)
002177(御银股份) MA金叉+RSI上行 得分3/3 | 600011(华能国际) ...

### Layer 2 — ML预测 (10只通过)
| 代码 | 名称 | 信号分 | ML涨概率 | 通过 |
|------|------|--------|----------|------|
| 002177 | 御银股份 | 3/3 | 72% | ✅ |
| 600011 | 华能国际 | 3/3 | 63% | ✅ |

### Layer 3 — DeepSeek AI 终审
002177(御银股份): ✅ 买入 [置信度85] — 趋势明确，成交量放大配合
600011(华能国际): ⏸️ 观望 [置信度60] — 短期涨幅过大，等待回调
```

## 验证方案

1. 先用模拟数据跑通三层 pipeline（不依赖真实 API）
2. 测试 DeepSeek API mock 和真实调用
3. ML 模型从 5527 只 CSV 训练 → 检查 AUC > 0.55
4. 全链路测试: `./start.sh evening`
5. 日报中确认三层筛选详情正确展示



三个子策略
1. 抢权策略 (Pre-Ex-Right)
逻辑: 除权前 15 天买入, 除权前 1 天卖出, 吃统计上的抢权溢价
精选池: 胜率 66.3%, 均收益 +4.47%, 夏普 1.51, 盈利因子 3.24
2. 填权策略 (Post-Ex-Right)
逻辑: 除权日收盘买入, 填权(回到除权前价格)或持有 25 天或止损出场
精选池: 胜率 75.1%, 均收益 +2.55%, 夏普 2.04, 仅持仓 5.7 天
最优性价比 — 最短持仓 + 最高夏普
3. 双轮动策略 (Dual Rotation)
逻辑: 除权前买入→除权前日卖出→除权日再买入→持有至填权
精选池: 胜率 68.8%, 均收益 +6.83%, 夏普 1.67, 盈利因子 4.05
最高绝对收益, 但持仓更长(19.3天)
回测结果对比
指标	标准池(1405只)	精选池(111只)
填权胜率	72.1%	75.1%
填权均收益	+1.73%	+2.55%
填权夏普	1.34	2.04
双轮动均收益	+4.02%	+6.83%
双轮动盈利因子	2.37	4.05
精选池 = 标准池筛选 前胜率>65% & 后胜率>60% & 填权率>85% & 事件>=5

使用方式
# 回测
python run_analysis.py backtest --start 2015-01-01         # 标准池
python run_analysis.py backtest --start 2015-01-01 --strict # 精选池

# 生成当前交易信号
python run_analysis.py signal --top 20
# 程序化调用
from dividend_analyzer.strategy import ExDividendStrategy, run_full_backtest

# 一键回测
results = run_full_backtest(strict_pool=True, start="2020-01-01")

# 自定义策略
s = ExDividendStrategy(
    data_dir="/Users/keyangwang/Documents/work/data/stocks",
    events_csv="reports/all_events.csv",
    summary_csv="reports/market_summary.csv",
)
s.load_selection_pool(
    min_events=5, min_pre_win_rate=0.65,
    min_recovery_rate=0.85, min_post_win_rate=0.60,
)
results = s.backtest(strategy="post_ex", market_filter=True)
实盘集成要点
精选池优于全市场 — 历史胜率是未来表现的有效预测指标
牛市加仓, 熊市空仓 — market_filter=True 自动过滤 2008/2018 级别的熊市
分散持仓 — 事件驱动天然跨时间分散, 建议同时持有 10-20 个不同除权日的事件
填权策略优先 — 夏普最高、持仓最短、容量最大, 可作为核心策略
输出文件 — reports/backtest_*.csv 和 reports/trades_*.csv 包含每笔交易明细, 可直接导入你的回测框架做组合层面验证