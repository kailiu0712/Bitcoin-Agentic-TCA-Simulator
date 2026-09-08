# 两种论文代理与 run 基线：统一口径比较

完整配置、输入文件时间/大小、代码哈希见 manifest.json；使用当前缓存重算，不能与旧 SUMMARY.md 数值直接拼接。

- random_n6 / n20：2025-03 论文随机 trader，再按各 trader 内同向串分组。
- threshold_wait5s / 20s：2025-09 论文 Algorithm 2；分买卖、截断离散幂律子单数量、指数目标等待、4 秒目标超时截止。forward scan 跳过的成交保持未分配（coverage.csv）。这是明确秒单位的真实市场适配，不声称论文已校准这些参数。
- local_run_1s：旧 run 分组逻辑，使用统一末价、逐日归一化和同侧末笔完成点；不是原模块的旧数值。
- aggressive 与 fill 使用完全相同的已验证订单集合；fill 的交换时间与 pre-mid 从所属订单继承，不能视作独立 quote 或独立决策。fill 粒度仅作敏感性。
- 主结果保留正负/零 signed impact；positive_only 仅展示筛正造成的差异。delta 是同尺度分箱均值上的描述拟合，边界命中另有标记，不能把它当成已验证的幂律。
- 每天独立构造，3 个 seed 是同一市场数据的重复分组，seed SD 不是统计显著性。各方法 parent 人群不同，当前是无条件对比，不是匹配 Q/T/参与率后的因果比较。
- 价格轨迹每个 parent 等权，至少 5 child；明确包含 phi=0.5 和 1。衰减使用各时钟内共同完整样本，剔除越界、跨日和超过 30 秒的陈旧报价。零 T 不进入 z 衰减。
- 当前 raw 响应含市场漂移和后续流，未拟合 propagator 或推断永久冲击。10min 与 z=2 不代表同一时间窗口。

## 结果（跨 seed 均值；seed SD 仅为构造敏感性）

| regime | granularity | variant | seeds | parents | T_median_s | participation | delta_mean | delta_seed_sd | delta_positive_only | trajectory_half |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal | aggressive | random_n6 | 3 | 35832 | 30.046 | 0.22012 | 0.21133 | 0.0060277 | 0.125 | 0.46084 |
| normal | aggressive | random_n20 | 3 | 37714 | 96.33 | 0.036947 | 0.23333 | 0.026951 | 0.11633 | 0.48363 |
| normal | aggressive | threshold_wait5s | 3 | 8420.3 | 6.7174 | 0.77954 | 0.108 | 0.056321 | 0.062667 | 0.48705 |
| normal | aggressive | threshold_wait20s | 3 | 5395 | 19.811 | 0.23842 | 0.15767 | 0.01159 | 0.046667 | 0.91523 |
| normal | aggressive | local_run_1s | 1 | 21255 | 0.37346 | 1 | 0.197 | — | 0.157 | 0.38969 |
| normal | fill | random_n6 | 3 | 68227 | 13.542 | 0.17784 | 0.17433 | 0.00057735 | 0.13967 | 0.52293 |
| normal | fill | threshold_wait5s | 3 | 10155 | 7.1121 | 0.15327 | 0.128 | 0.040927 | 0.081667 | 0.4723 |
| stress | aggressive | random_n6 | 3 | 43082 | 27.742 | 0.21836 | 0.21333 | 0.012097 | 0.14367 | 0.46481 |
| stress | aggressive | random_n20 | 3 | 45083 | 89.634 | 0.037014 | 0.25067 | 0.023587 | 0.14667 | 0.48455 |
| stress | aggressive | threshold_wait5s | 3 | 10123 | 6.7175 | 0.75985 | 0.138 | 0.023302 | 0.085667 | 0.3708 |
| stress | aggressive | threshold_wait20s | 3 | 6295 | 19.736 | 0.24068 | 0.17367 | 0.075142 | 0.078333 | 0.35691 |
| stress | aggressive | local_run_1s | 1 | 28359 | 0.36307 | 1 | 0.369 | — | 0.234 | 0.39224 |
| stress | fill | random_n6 | 3 | 79333 | 12.907 | 0.18339 | 0.21933 | 0.010214 | 0.16033 | 0.51222 |
| stress | fill | threshold_wait5s | 3 | 12167 | 6.9992 | 0.15771 | 0.10833 | 0.066161 | 0.11033 | 0.4516 |

## 数据核验

| cached_trades | cached_fills | cached_quotes | fills_removed_outside_regime | fill_groups_unverified | orders_removed_invalid | exchange_backsteps_before_sort | quote_exchange_backsteps_before_sort | matched_orders | matched_fills | source_order_coverage | fill_time | quote_time | regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 161702 | 396688 | 4490391 | 55433 | 4 | 0 | 53 | 0 | 161702 | 341248 | 1 | inherited verified aggressive-order exchange timestamp | stable exchange-time sort; same timestamp last receipt wins | normal |
| 195933 | 415105 | 6199245 | 25031 | 0 | 0 | 32 | 0 | 195933 | 390074 | 1 | inherited verified aggressive-order exchange timestamp | stable exchange-time sort; same timestamp last receipt wins | stress |

## 输出

- comparison_normal.png / comparison_stress.png：六面板对照。
- parents_*.parquet：逐母单含 child 行索引、价格、规模、时长及状态。
- size_curves.csv / trajectories.csv / decay.csv / surface_*.csv：底层曲线和 Q–T 面。
- daily_metrics.csv：逐日指标；coverage.csv：单笔过滤及被算法跳过的行。
- analysis_rows_*.parquet：可直接逐行分析的干净主动订单表。

所有 SEM 列标为 sem_iid，仅描述横截面散布，未处理时间重叠；本报告不作正式显著性宣称。
