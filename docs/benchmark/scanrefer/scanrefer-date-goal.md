# ScanRefer Current Goal and Problem Statement - 2026-05-12

本文是 ScanRefer 方向当前唯一的目标/问题总览文档。它合并并取代下列过程文档中的有效信息：

- `final_investigation_summary_20260504.md`
- `keyframe_funnel_20260504.md`
- `picking_error_diagnosis_20260503.md`

版本结果的审计记录仍以各版本 doc 为准；本文不替代不可变的 per-version benchmark 文档，也不把未过 gate 的实验写成正式结果。

## 1. 一句话目标

我们的目标不是把 ScanRefer 数字通过 oracle 做高，而是得到一个可以和 Camp-A zero-shot 方法公平比较的 ScanRefer VG pipeline：

- 无 GT view oracle：初始图像证据不能由 GT `target_id` visibility 选出。
- Mask3D-pool detection-mode：最终 `proposal_id` 必须来自检测 proposal pool，不使用 GT proposal pool。
- Query-driven Stage 1：用自然语言 query 做 evidence retrieval。
- Stage 2 agent reasoning：agent 基于图像、crop、proposal、空间工具主动查证并提交 proposal。
- 先在 frozen random100 上稳定提升和诊断；达到可解释状态后扩到 full
  val 9508 条。v3.20 已完成 full-val consensus3 验证。

当前最重要的问题已经不是“有没有把目标看见”，而是“目标已经出现在 agent 看过的证据里时，agent 仍然在多干扰物 / label mismatch 场景中选错或提交 `proposal_id=-1`”。

## 2. 当前结论

### 2.1 最高数字：v2 是上界，不是 apples-to-apples SOTA

v2 在 ScanRefer full val 9508 条上的数字是：

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Overall | 69.92 | 62.79 |
| Unique | 83.11 | 76.49 |
| Multiple | 65.00 | 57.68 |

但 v2 的初始 keyframes 使用 GT view oracle：根据 GT `target_id` visibility 选图。它的最终 proposal 仍来自 Mask3D pool，所以不是 GT-pool 泄漏；但 view selection 已经看到了 target visibility，因此不能和 Z3D、SeeGround、CSVG 等 Camp-A zero-shot 结果直接比较。

v2 的正确定位：上界 / 诊断结果。它说明在“看图非常有利”的情况下，Stage 2 + Mask3D pool 可以达到很高分，也说明 full-val evaluator 和 aggregation-GT bbox 路径是可用的。

### 2.2 当前 query-driven full-val headline：v3.20 consensus3

当前最高的 query-driven / Mask3D-pool full-val 结果是 v3.20：

| Split | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| Overall | 55.36 | 49.57 | 0.4745 |
| Unique | 78.58 | 72.31 | - |
| Multiple | 46.71 | 41.09 | - |

v3.20 是三条 fresh no-GT source trajectories 的 proposal-id consensus，
不是单次 agent 轨迹。三条 source 都使用 `pack_scanrefer_v3p_iterative`、
`mask3d_query_driven` keyframes、TADG、no-match guard、evidence-frame guard；
CVRA 关闭。每条 source 先用 aggregation-GT rescore，再由
`scanrefer_consensus_side_by_side.py` 对 `selected_object_id` 做
`plurality_avg_confidence` 投票。selector 不读取 GT / IoU；GT 只在 rescore
和 leaderboard metrics 中使用。

不投票的 source metrics 是：

| Run | Overall@0.25 | Overall@0.50 | Mean IoU |
|---|---:|---:|---:|
| source_a | 54.00 | 48.36 | 0.4633 |
| source_b | 53.52 | 47.93 | 0.4587 |
| source_c | 53.18 | 47.60 | 0.4559 |
| single-run mean | 53.57 | 47.96 | 0.4593 |
| consensus3 | 55.36 | 49.57 | 0.4745 |

因此 v3.20 相对单跑均值提升 +1.80 / +1.61 pp，相对最好的 source_a
提升 +1.37 / +1.21 pp。对外做 zero-shot 对比时，必须同时说明这是
3x inference + voting 结果；如果做成本归一化或单次推理比较，应引用
source_a/b/c 或 single-run mean。

v3.20 consensus 分布：

- 3/3 unanimous：5293 条。
- 2/3 majority：3068 条。
- 1/1/1 all-different：1147 条，走 confidence / deterministic tie-break。
- choice source：source_a=3713、source_b=3062、source_c=2733。

### 2.3 random100 development headline：v3.19 consensus3

当前最高的 query-driven / Mask3D-pool random100 结果是 v3.19：

| Split | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| Overall | 61.00 | 54.00 | 0.5080 |
| Unique | 85.71 | 85.71 | - |
| Multiple | 51.39 | 41.67 | - |

v3.19 不是新的单次 agent 轨迹，而是对已经完成的 v3.10 / v3.11 /
v3.13 三个 no-GT random100 输出做 deterministic plurality consensus：
按 `selected_object_id` 多数票选择，平票时用 agent 自报 confidence
打破。selector 不读取 GT / IoU；GT 只在最后 leaderboard metrics 里计算。

这个结果已经在 frozen random100 上超过 Z3D Mask3D-pool reference 58.9 /
52.7，但必须带 caveat：source set 是在审计同一个 development fold 的历史
产物之后确定的，因此 v3.19 是 random100 开发折 headline，不是 full-val
paper row。v3.20 是它的 fresh full-val 后续验证，但 v3.20 也必须保留
consensus caveat。

当前最佳单次 agent / 单配置 random100 结果仍然是 v3.11：

| Split | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| Overall | 54.00 | 50.00 | 0.4615 |
| Unique | 82.14 | 82.14 | - |
| Multiple | 43.06 | 37.50 | - |

v3.11 在同一个 frozen random100 fold 上相对 v3.10 的 Overall@0.25 持平，Overall@0.50 提升 +2pp，Mean IoU 从 0.4550 提升到 0.4615；但它仍低于 Z3D 的 zero-shot Mask3D-pool reference 58.9 / 52.7。

后续 v3.12 vertical-relation guard 和 v3.13 late-override guard 均已完成同一 random100 gate，但结果都是负向：v3.12 为 54.00 / 48.00，Mean IoU 0.4559；v3.13 为 54.00 / 48.00，Mean IoU 0.4601。它们已作为不可变过程记录和 SQLite row 保留，但不替代 v3.11 headline。

v3.14 只做了 `scene0552_00::14::0` 的 one-sample focus gate，尝试用 EFG 阻止 stale zero-support left/right rank；fresh trace 没有触发该窄条件，仍选错 proposal 24，IoU 0.0。它已作为 rejected focus gate 入库和记录，但不是 random100 结果，也不进入 leaderboard。

v3.15 继续验证另一个窄 EFG 方向：final rationale 引用某个 frame，但 submitted proposal 的 `inspect_proposal.frames_appeared` 不包含这个 frame，而同类候选包含。它在 `scene0203_00::14::3` focus1 上能把 pillow 样本救回，但 10-case focus control 只有 20.00 / 20.00，和 v3.11 在同一切片持平且显著低于现有 v3.13 trace。该候选已作为 rejected focus gate 入库和记录，并从默认代码路径回滚。

v3.16 验证了 TADG 的另一个窄方向：当 agent 用 `not_in_candidates` override 提交某 proposal，但该 proposal 从未出现在任何非空 category lookup 中，而 relation-ranked proposal 出现过时，拒绝这次 override。focus6 结果是 50.00 / 50.00，和 v3.11 在同一切片阈值持平但 mean IoU 从 0.4463 降到 0.4294；更关键的是新分支没有在 motivating fresh trace 上触发，实际触发的是既有 `TADG_STRICT` strong-2D 分支并推向错误 proposal 27。该候选已作为 rejected focus gate 入库和记录，并从默认代码路径回滚。

v3.17 验证了 runtime final-revision guard：当本轮已经 accepted 一个 final proposal 后，如果后续 deferred evidence 导致 agent 改交另一个 proposal，必须提供 `tool_override_reason`。focus12 结果是 41.67 / 25.00，低于 v3.11 同切片的 41.67 / 33.33，也低于 v3.13 的 58.33 / 41.67。该 guard 虽然在 6/12 样本上触发，但 agent 往往补一句 reason 后仍改交错误 proposal，因此该候选已作为 rejected focus gate 入库和记录，并从默认代码路径回滚。

v3.18 尝试把 final evidence ledger 前置到 `submit_final` 之前：agent 需要先调用 `record_final_selection` 记录 winner、candidate set、evidence frame 和 reject reasons。focus12 结果只有 16.67 / 16.67，显著低于 v3.11/v3.13/v3.17。ledger 在 12/12 样本上都被调用，但没有带来更好的 winner 选择，说明“让模型解释当前选择”不是有效 scoring function。该候选已作为 rejected focus gate 入库和记录，并从默认代码路径回滚。

### 2.4 最新 full-val gate：v3.20 已落地，但单轨迹仍未解决

v3.20 full val 已满足过程记录 gate：

| Gate | Status |
|---|---|
| Fold size | `9508/9508` completed |
| Source coverage | source_a/source_b/source_c 均 `9508/9508` |
| Inference inputs | query-driven Stage 1 + Mask3D proposal pool outputs; no GT in selector |
| Consensus selector | `src/evaluation/scripts/scanrefer_consensus_side_by_side.py` |
| Leaderboard metrics | done |
| SQLite ingest | `v3p20_consensus3_full_20260511` |
| Durable doc | `docs/benchmark/scanrefer/v3p20_consensus3_full_20260511.md` |

v3.20 的有效结论是：三轨迹 consensus 的 random100 增益能迁移到 full val，
但 full-val source rows 也说明单轨迹仍停在 48% 左右 Acc@0.50。换句话说，
consensus 是可复现的工程提升，不是单个 agent 的 reasoning 问题已经解决。

为跑完整 full val，运行时做了几项必须保留的工程优化：

- agent runner 支持 `--checkpoint-only` 和 `--max-new-samples`，批量只写
  per-sample checkpoint，避免在长进程中积累完整结果列表。
- `side_by_side.json` 可从 checkpoints 流式组装；metrics 和 consensus 也只
  读取必要字段，避免把 trace-heavy JSON 全量加载进内存。
- Stage 1 selector cache 和 ConceptGraph object cache 都限制为最多 4 个
  scene，避免 scene 数增长导致 RSS 持续上涨。
- 最终成功 completion 使用 `WORKERS_PER_SOURCE=96`、`SOURCE_BATCH_SIZE=96`、
  `SOURCE_PARALLELISM=2`、`RSS_LIMIT_KB=25165824` 和
  `SYSTEM_FREE_MIN_PERCENT=35`；`112/112` 曾出现密集 429 retry，因此回退。

结论：v3.20 是当前 full-val no-GT query-driven headline，可以作为正式结果
记录；但必须按 consensus3 报告。下一步从研究角度有两条路线：

- 工程路线：把 v3.20 的临时 orchestration 固化为可复跑脚本，并记录成本、
  429 rate、RSS 峰值和失败恢复策略。
- 模型/工具路线：继续把 consensus 找到的互补样本转化为单轨迹可用的
  proposal ranking / evidence scoring，而不是继续堆解释型 guard。

### 2.5 v3.11 focus13 的作用：先验消融，已由 random100 gate 验证

v3.11 的补丁最初来自 paired flip focus13 消融：

| Variant | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| v3.10 on same 13 cases | 53.85 | 46.15 | 0.4176 |
| v3.11 TADG anchor-exclusion only | 53.85 | 46.15 | 0.4488 |
| v3.11 EFG 2D-mark experiment | 53.85 | 38.46 | 0.3482 |

v3.11 的有效改动是：TADG candidate coverage 在检查 same-category compare 是否遗漏候选时，不再把当前 anchor 本身算作遗漏候选。这修复了 `scene0660_00::4::4` 这类 “chair next to another same chair” 路径中，anchor 被合理排除在 candidate_ids 外却触发 coverage block 的问题。

该消融本身只提高 mean IoU，没有提高 Acc@0.25 / Acc@0.50；同时还有 `scene0565_00::12::2`、`scene0100_00::26::0`、`scene0149_00::21::2` 三个 v3.10 recovery 回退。因此它当时只能作为候选补丁。后续 fresh random100 gate 证明该补丁在完整 frozen random100 上净正向：Overall@0.50 从 48.00 提升到 50.00，Mean IoU 从 0.4550 提升到 0.4615。

EFG 2D-mark 实验已经回滚：它在同一 focus13 上降低 Acc@0.50 和 mean IoU，没有足够证据保留。

### 2.6 v3.12 的结论：vertical compare 缺失不是主要瓶颈

v3.12 尝试解决 v3.11 的一个明确 regression：当 query 描述目标本身在 anchor 上方/下方时，agent 可能没有调用 `compare_proposals_spatial` 的 `above` / `below` 就直接提交结果。补丁让 TADG 在这类路径上用 `TADG_VERTICAL_RELATION_UNTESTED` 阻止 final answer，迫使 agent 先做 vertical relation check。

random100 gate 结果：

| Split | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| Overall | 54.00 | 48.00 | 0.4559 |
| Unique | 71.43 | 71.43 | - |
| Multiple | 47.22 | 38.89 | - |

相对 v3.11，v3.12 的 paired delta 是：

- Acc@0.25 recoveries / regressions = 7 / 7。
- Acc@0.50 recoveries / regressions = 4 / 6。
- Mean IoU delta = -0.0056。

结论：只强制 vertical compare 不够。`scene0328_00::15::0` 上新 guard 已触发并迫使 agent 做 `below` 比较，但 `below` ranking 仍然把错误 proposal 排在前面。下一步应该修 proposal ranking / compound relation scoring，而不是继续堆 guard。

### 2.7 v3.13 的结论：late-overwrite guard 太脆弱

v3.13 尝试解决另一个明确 failure mode：agent 早期已经通过 relation-ranked evidence 提交了正确 proposal，但后续 deferred visual evidence 注入后又用 left/right override 覆盖成错误 proposal。补丁让 TADG 在 `left_of` / `right_of` rank-mismatch override 中检查历史 `submit_final`，如果当前 rank-1 proposal 已经被成功提交过，就拒绝后续覆盖。

random100 gate 结果：

| Split | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| Overall | 54.00 | 48.00 | 0.4601 |
| Unique | 75.00 | 75.00 | - |
| Multiple | 45.83 | 37.50 | - |

相对 v3.11，v3.13 的 paired delta 是：

- Acc@0.25 recoveries / regressions = 8 / 8。
- Acc@0.50 recoveries / regressions = 6 / 8。
- Mean IoU delta = -0.0014。
- 新增的 “already accepted” strict branch 在 random100 中触发 0 次。

结论：late overwrite 是真实风险，但通过事后扫描历史 `submit_final` 的 guard 太脆弱，不能作为正式机制。下一步应该显式化 final evidence ledger，让最终答案绑定最近一次 relation-ranked winner / reject reason，而不是继续增加机会主义 guard。

### 2.8 v3.14 的结论：stale zero-support left/right guard 过拟合旧 trace

v3.14 不是 full random100 gate，而是针对 `scene0552_00::14::0` 的 focus gate。这个样本在 v3.11 中错选 proposal 24，v3.13 又随机恢复到 proposal 6，因此看起来像一个可修的 left/right spatial-rank 问题。

候选补丁尝试在 EFG 中拦截如下路径：

- latest `compare_proposals_spatial` 是 `left_of` / `right_of`；
- submitted proposal 是 rank-1；
- `supporting_frame_counts` 全为 0；
- compare 之后 agent 看过某些 frame，其中 anchor 和另一个候选共同可见，但 submitted proposal 不可见。

单测通过，但 fresh focus trace 不符合这个条件：agent 在 compare 后先看了 frame 8，其中 anchor 23、submitted candidate 24、alternative candidate 6 都在 `visible_proposals` 中；随后又引用 frame 57，其中 submitted 24 和 anchor 23 都在 `visible_proposals` 中。于是新 guard 没有触发，最终仍提交 24，aggregation-GT IoU 0.0。

结论：zero-support left/right rank 的确是风险，但不能再堆这种依赖旧 trace 形态的窄 guard。下一步更应该把 final evidence ledger 显式化，或者让 left/right spatial tool 在没有 2D shared-frame support 时返回“未验证/需要补证据”，而不是把 3D x-offset 排名当作强证据。

### 2.9 v3.15 的结论：cited-frame inspected-visibility guard 只能修单点

v3.15 针对 `scene0203_00::14::3` 做了另一个 focus gate。该样本在 v3.11 中提交 pillow proposal 73，IoU 0.0，但 rationale 引用 “frame 144”；agent 自己的 `inspect_proposal(73)` 里没有 frame 144，而同类 pillow proposals 2/39 的 `frames_appeared` 包含 frame 144。候选补丁让 EFG 在这种情况下阻止 final，并提示提交实际覆盖 cited frame 的同类 proposal。

结果：

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus1 | 1 | 100.00 | 100.00 | 0.9759 |
| focus10 | 10 | 20.00 | 20.00 | 0.1941 |

focus1 证明这个 failure mode 是真实的，而且可以只用 agent trace 修复，不需要 GT。但 focus10 同时暴露了两个问题：

- 同一 10-case slice 上 v3.15 只和 v3.11 持平，远低于现有 v3.13 trace 的 50 / 40。
- 其中 `scene0377_00::9::2` 从 v3.11/v3.13 正确变成错误，但并不是新 guard 触发导致，说明短 focus rerun 本身仍有较大轨迹抖动。

结论：v3.15 不应进入 random100。继续加“提交后检查 rationale 与 trace 是否矛盾”的窄 EFG 分支收益太局部；下一步应把 final evidence ledger 前置到 selection contract，让 agent 在提交前明确绑定 winner、候选集合、空间/视觉证据和 reject reason。

### 2.10 v3.16 的结论：category-supported TADG guard 没有命中真实 fresh trace

v3.16 针对 `scene0149_00::21::2` 做了 focus gate。旧 trace 里 agent 通过 `not_in_candidates` override 提交 proposal 33，而 relation-ranked candidate 来自已有 category/spatial trace；因此候选补丁尝试用 agent 自己的 category lookup 作为约束：如果 submitted proposal 从未出现在任何非空 category lookup，而 relation-ranked proposal 出现过，就拒绝这次 override。

focus6 结果：

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus6 | 6 | 50.00 | 50.00 | 0.4294 |

同一切片上：

| Version | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| v3.11 | 50.00 | 50.00 | 0.4463 |
| v3.13 | 16.67 | 16.67 | 0.1805 |
| v3.16 | 50.00 | 50.00 | 0.4294 |

关键触发审计：

- 新的 category-supported `not_in_candidates` 分支没有触发。
- `scene0149_00::21::2` fresh trace 触发的是既有 `TADG_STRICT` strong-2D 分支，把 proposal 33 改成 27，但 27 的 IoU 只有 0.0241。
- v3.16 保住了 `scene0025_00::9::0` 和 `scene0660_00::4::4`，修复了 `scene0412_00::13::0`，但把 `scene0629_00::3::3` 从 0.9643 降到 0.0。

结论：v3.16 不应进入 random100。它进一步证明“扫描历史 trace 后加一个窄 guard”的收益不稳定；下一步需要把 final selection contract 前置，让最终答案显式绑定 winner、候选集合、空间证据和 reject reason，而不是在 submit 后追加局部 veto。

### 2.11 v3.17 的结论：final-revision guard 太晚，不能稳定最终选择

v3.17 针对 v3.11 的一个真实失败模式做 focus gate：agent 早期已经 accepted 一个 final proposal，但后续 deferred evidence 注入后又改交另一个 proposal，并且没有明确解释为什么推翻早先答案。候选补丁在 runtime 中记录已接受的 final submission；如果后续 `submit_final` 改变 `proposal_id` 但没有 `tool_override_reason`，就用 `FINAL_REVISION_GUARD` 软拦截。

focus12 结果：

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus12 | 12 | 41.67 | 25.00 | 0.2987 |

同一切片上：

| Version | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| v3.11 | 41.67 | 33.33 | 0.3220 |
| v3.13 | 58.33 | 41.67 | 0.4388 |
| v3.17 | 41.67 | 25.00 | 0.2987 |

关键触发审计：

- 新 guard 在 6/12 样本上触发：`scene0203_00`、`scene0377_00`、`scene0414_00`、`scene0426_00`、`scene0593_00`、`scene0595_00`。
- `scene0203_00::14::3` 最终保持 v3.13 的正确 proposal 2，说明 failure mode 真实且可被 runtime 观察到。
- 但 `scene0377_00::3::1` 从 v3.11/v3.13 的正确 proposal 4 变成 v3.17 的错误 proposal 37，导致 Acc@0.50 净下降。
- 根因是 guard 太晚：agent 可以补充一个简短 `tool_override_reason` 后继续改交，最终选择仍没有被强制绑定到候选集合、空间证据和 reject reason。

结论：v3.17 不应进入 random100。它进一步证明“late confirmation guard”不够，下一步应把 final evidence ledger 前置到 `submit_final` 之前，要求 agent 在真正提交前明确列出 winner、被比较的相邻候选、证据 frame/crop、空间关系判断和为什么排除其他候选。

### 2.12 v3.18 的结论：通用 final-selection ledger 只是解释器，不是排序器

v3.18 把 v3.17 后的想法往前挪了一步：新增 `record_final_selection` 工具，让 agent 在 `submit_final` 前记录最终选择合同，包括 winner、candidate_ids、evidence_frame_ids、reject_reasons、spatial_relation 和 anchor_id；`submit_final` 会检查 ledger winner 是否等于最终 proposal。

focus12 结果：

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus12 | 12 | 16.67 | 16.67 | 0.1564 |

同一切片上：

| Version | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| v3.11 | 41.67 | 33.33 | 0.3220 |
| v3.13 | 58.33 | 41.67 | 0.4388 |
| v3.17 | 41.67 | 25.00 | 0.2987 |
| v3.18 | 16.67 | 16.67 | 0.1564 |

关键审计：

- 12/12 样本都调用了 `record_final_selection`，每个样本 2-4 次。
- `FINAL_SELECTION_LEDGER_*` block 为 0，说明 agent 基本遵守了新工具流程。
- 但 v3.18 把多个已有成功样本打坏：`scene0377_00::3::1` 从 v3.11/v3.13 的正确 proposal 4 变成 41；`scene0426_00::3::2` 从正确 proposal 2 变成 13；`scene0595_00::1::3` 从可命中 proposal 12 变成 18。
- ledger 让输出更可审计，但没有提供新信息或更强 ranking，它只是把当前错误选择包装成结构化解释。

结论：v3.18 不应进入 random100。下一步不能再加“generic reflection/contract”工具；需要 code-side 的候选排序或证据计算，例如对 ambiguous same-category candidates 生成可比较的局部视觉/空间特征，而不是继续要求模型自解释。

## 3. 评测协议

### 3.1 正式目标协议

正式目标协议如下：

- 数据集：ScanRefer val，full val 共 9508 utterances；开发阶段使用 frozen random100。
- 样本选择：random100 使用固定 seed 和固定 sample id 文件，跨版本对齐。
- 输入：自然语言 reference + scene assets。
- Proposal pool：Mask3D / ConceptGraph proposal pool，去掉 wall/floor/ceiling 等 Camp-A convention 下不参与检测的类别。
- 初始证据：query-driven Stage 1 retrieval，不允许 GT target visibility 参与 keyframe 选择。
- Agent：Stage 2 ReAct-style VLM agent，可 request more views/crops、查 proposal、比较空间关系、切换 hypothesis。
- 最终答案：agent 提交一个 proposal id；evaluator 用 ScanNet aggregation-derived GT bbox 计算 axis-aligned 3D IoU。
- 指标：Acc@0.25 / Acc@0.50 x Unique / Multiple / Overall，以及 mean IoU。

### 3.2 输入原则

ScanRefer 主线的输入原则是：推理阶段只能使用 benchmark participant 在 detection-mode zero-shot 设置下合理可得的信息；GT 只能在评测阶段使用。

允许进入推理的输入：

- 自然语言 reference，即 ScanRefer utterance 本身。
- `scene_id` 和对应的 prepared scene assets：RGB/depth/pose/intrinsic、已有 scene index、可渲染或可读取的 scene evidence。
- 检测得到的 proposal pool：Mask3D / ConceptGraph proposals、proposal label、bbox、可见帧、crop、几何中心、空间关系等。
- Query-driven Stage 1 产物：由 reference 解析出的 hypothesis、query-driven keyframes、后续 callback 产生的新 views/crops。
- Agent 自己在本次推理过程中产生的 tool trace、比较结果、候选排序和 reject reason。

禁止进入推理的输入：

- GT `target_id`、ScanRefer `object_id`、GT instance id。
- GT bbox、GT point cloud、GT segmentation mask。
- GT target visibility、`object_to_views[target_id]`、任何用 GT 目标可见性挑出来的 keyframes。
- 用 GT IoU 反推的候选筛选、候选排序、fallback answer 或 case-specific hint。
- 为某几个 failed samples 手工写入的答案、类别映射或 prompt hint。

允许在评测阶段使用的 GT：

- ScanNet aggregation-derived GT bbox 只能在 agent 提交 `proposal_id` 之后用于算 IoU 和 Acc@0.25 / Acc@0.50。
- frozen random100 sample id 文件只能定义评测折，不能携带目标选择信息。

这个原则解释了为什么 v2 是上界而不是正式 zero-shot SOTA：v2 没有用 GT proposal pool，但它把 GT target visibility 用在初始 keyframe selection 上，已经违反“推理输入不得包含 GT visibility”的原则。

### 3.3 正式结果 gate

一个 run 只有满足以下条件，才能写成正式结果：

1. `side_by_side.json` 的样本数等于目标 fold 大小，例如 random100 必须 `n=100`。
2. `failed_count == 0`。
3. 已完成 aggregation-GT rescore。
4. 已生成 leaderboard metrics。
5. 已 ingest 到 `docs/benchmark/scanrefer/runs.sqlite`。
6. 已写新的 `docs/benchmark/scanrefer/<version>_<tag>_<date>.md`。
7. 已更新 `docs/benchmark/scanrefer/README.md` 的 timeline。

如果有 failed sentinels，必须先分类并处理：

- transient exception / HTTP 504：可删除失败 checkpoint 后 full fold cached retry。
- data-root / mesh：先修数据路径或 callback wiring，不能吞成 failed zero。
- model submitted `proposal_id=-1`：如果 pool 中存在 GT-matching candidate，这是 agent 行为问题，不能当 empty-pool。

### 3.4 为什么不能用 partial 或 failed run 做 headline

ScanRefer 的失败样本不是随机缺失。`proposal_id=-1` 往往集中在 label mismatch、多干扰物或描述-类别不一致场景。如果直接忽略失败样本，会改变 denominator 并偏移困难样本分布。v3.6 ignore-2 的 98-sample 数字因此只能用于定位问题，不能作为 benchmark 结果。

## 4. 版本迭代线

目前可以按下列版本/尝试来理解：

| Version | Status | Key change | Headline / diagnosis |
|---|---|---|---|
| v1 | historical | 第一版 ScanRefer full eval | 使用 Phase8 GT-CG bbox，bbox 口径不 paper-comparable |
| v2 | official upper bound | aggregation-GT bbox 修正 | full val 69.92 / 62.79，但有 GT view oracle |
| v3 | historical query-driven | query-driven 初版 | 后来在 v3.2 runtime 修复下 aggregation-GT 纠正为 47 / 40 |
| v3.1 | official smoke | Mask3D-CG visibility keyframes | aggregation-GT corrected 45 / 39；X1 不成立 |
| v3.2 | official | callback / evaluator durability 修复 | random100 46 / 40 |
| v3.3 | official | vertical spatial relations | random100 48 / 42，旧 honest headline |
| v3.4 | negative | `select_among_proposals` forced 1-of-K | random100 44 / 38，退化 |
| v3.5 | negative smoke | CVRA visible non-label proposal augmentation | smoke6 retrieval recall 高，但 0/6 F5a flip；未 full random100 |
| v3.6 | blocked | TADG, CVRA off | n=100 但 failed=2，未 harvest |
| v3.7 | mixed | no-match / evidence-frame guards and frame-view enrichment | random100 46 / 42，durability win but metric loss |
| v3.8 | official | relation-aware spatial ranking and TADG aliases | random100 49 / 43，small fresh win |
| v3.9 | official | anchor/evidence guards and direct-final deferral | random100 53 / 46 |
| v3.10 | superseded | EFG/TADG cleanup plus `submit_final` terminal latch | random100 54 / 48，曾是 honest headline |
| v3.11 | best single-run | TADG same-category anchor-exclusion | random100 54 / 50，最佳单轨迹结果但仍低于 Z3D |
| v3.12 | negative | TADG target above/below missing-compare guard | random100 54 / 48，低于 v3.11，不推广 |
| v3.13 | negative | TADG late left/right override after accepted final guard | random100 54 / 48，新分支触发 0 次，不推广 |
| v3.14 | rejected focus | EFG stale zero-support left/right focus guard | n=1 focus gate 0 / 0，未触发目标分支，已回滚 |
| v3.15 | rejected focus | EFG cited-frame inspected-visibility guard | focus1 修复 scene0203，但 focus10 只有 20 / 20，已回滚 |
| v3.16 | rejected focus | TADG category-supported `not_in_candidates` guard | focus6 为 50 / 50，新分支未触发且 mean IoU 低于 v3.11 切片，已回滚 |
| v3.17 | rejected focus | Runtime final-revision guard | focus12 为 41.67 / 25.00，guard 触发但太晚，低于 v3.11/v3.13 同切片，已回滚 |
| v3.18 | rejected focus | Pre-submit final-selection ledger | focus12 为 16.67 / 16.67，ledger 全部调用但显著降分，已回滚 |
| v3.19 | random100 consensus headline | 三轨迹 no-GT plurality consensus | random100 61 / 54，超过 Z3D random100 reference；development-fold precursor |
| v3.20 | current full-val headline | 三条 fresh full-val no-GT source trajectories + consensus3 | full val 55.36 / 49.57；single-run mean 53.57 / 47.96，必须带 consensus caveat |

full-val headline 当前是 v3.20 consensus3；random100 开发折 headline 是
v3.19 consensus3；最佳单轨迹 random100 headline 仍是 v3.11。各版本不可变记录
以 `docs/benchmark/scanrefer/v*.md` 和 `runs.sqlite` 为准。

## 5. 当前核心问题

### P0. 目标已被看见，但最终选择错

keyframe funnel 的核心发现是：瓶颈不主要在 initial keyframe coverage。

在 v3.3 random100 上，按 per-sample funnel 粗略拆分：

| Stage | Meaning | Count |
|---|---|---:|
| F1 | sample / GT available | 100 |
| F2 | initial query-driven keyframes contain GT-visible evidence | 65 |
| F3 | callbacks / later evidence include more useful target evidence | 83 |
| F4 | agent observed some frame where GT target is visible | 86 |
| F4b | proposal pool contains plausible IoU target candidate | 94 |
| F5a | final chosen proposal hits IoU threshold | 42 |
| F5b | agent selected exact GT-ish proposal in stricter check | 14 |

最关键的 bucket 是 F4 yes but final wrong：约 44 个样本中，agent 已经看过包含目标的 frame，却没有选中正确 proposal。这意味着继续只做“让 Stage 1 多找一点图”的上限有限；主要收益应该来自 proposal selection / cross-check / final decision 约束。

### P1. Multi-distractor proposal ranking 是主瓶颈

Unique 和 Multiple 的差距仍然非常大。v3.19 consensus 已经把 Multiple 往上推了一步，但瓶颈仍在 Multiple：

- v3.11 single-run Unique@0.50 = 82.14，Multiple@0.50 = 37.50。
- v3.19 consensus Unique@0.50 = 85.71，Multiple@0.50 = 41.67。

Unique 高说明当 scene 中同类干扰少时，当前 evidence + agent 能工作。Multiple 低说明问题集中在：

- 同类/近类 proposal 很多。
- query 的空间关系、相对位置、局部外观需要同时满足。
- agent 容易被 label、单帧显著性、局部 crop 误导。
- 一旦工具返回多个候选，agent 没有稳定地把描述约束转成排序。

v3.4 的 `select_among_proposals` 失败进一步说明：简单把候选集合交给一个 VLM 做 1-of-K，不足以解决问题。原因是候选最可见 frame 不一定包含描述中的 anchor；VLM 看 candidate crop 时可能看不到“left of entrance / above refrigerator / right of bookcase”这类关系。

### P2. Label mismatch / no-match 已缓解，但 hard-case final selection 仍不稳

v3.6 最早暴露了“pool 中有 GT-matching proposal，但 agent 认为没有匹配目标”的问题：

- `scene0474_00::15::3`：query 描述 floor-resting high-back chair cushion / whiteboard / bookcase 关系；pool 中 `chair` proposal IoU 0.3538，但 agent 最终 `-1`。
- `scene0678_00::34::3`：query 描述 snack machine / beverage machine / entrance 关系；pool 中 `shelf` proposal IoU 0.7851，但 agent 最终 `-1`。

v3.7-v3.11 已把 `failed_count` 清到 0，并通过 no-match guard、EFG、TADG 和 terminal latch 让这类路径可控；v3.19 进一步说明多个 no-GT 轨迹之间存在可利用的互补性。但 focused rerun 仍显示 `scene0354_00::9::2`、`scene0690_00::15::2` 等样本会在已有候选/已有证据条件下选错相邻 proposal。问题已经从“提交 `-1`”转为“在多个可解释候选之间稳定选对”。

这说明不能把 proposal label 当硬语义真值。Mask3D / ScanNet200 label 可能和 ScanRefer reference phrase 不一致：

- 物体可能被标成 `shelf`，但人类描述为 snack machine。
- 物体可能被标成 `chair`，但描述强调 cushion、back、location 或上下文 anchor。
- object name、semantic class、visual appearance、spatial referent 四者并不总是同一个词。

下一版如果只继续增加 guard，而不改最终排序证据闭环，仍然会在这类样本上失败。

### P3. Callback recovery 的上限比 picking 修复低

v3.2 之后 callback durability、raw PNG path、structured payload extraction、tool trace persistence 都已经修复。v3.6-v3.11 又陆续修了 ScanRefer hypothesis callback 中 `use_visual_context=False` 的 text-only re-query wiring、mesh-root failures、newly queued image injection、selector cache、transient retry classification、same-turn final overwrite 和 same-category anchor coverage 误报。v3.19 没有改变 Stage 1 coverage，而是利用 Stage 2 final-selection 的多轨迹互补性。

这些修复很重要，因为它们让 run 可重复、失败可诊断。但从 funnel 看，callback/F4-fail 的直接上限约 14pp，而 F4 yes/picker wrong bucket 约 44pp。后续不能把主要精力继续投到“更多 callback”或“更多初始 views”，除非它直接服务于最终 proposal 排序。

### P4. CVRA 暴露了 addressability 定义过宽

v3.5 CVRA 的 smoke6 结果是：

- retrieval recall 6/6：target pid 能进入 `label_hits ∪ clip_visible_aug`。
- F5a flips 0/6：最终正确选择没有增加。
- 5/6 agent pick 与 v3.3 baseline 相同。

这说明“可通过 bbox IoU 找到候选”不等于“agent 在描述约束下应该选择它”。某些 CVRA candidate 与 GT bbox 足够重叠，但空间 referent 与 query 矛盾；`compare_proposals_spatial` 正确拒绝它们。未来做 candidate augmentation 时，必须把 spatial referent 验证纳入 addressability，而不是只看 label miss + bbox IoU。

### P5. TADG 当前问题不是 telemetry，而是行为闭环

TADG 的 telemetry 规则已经澄清：

- raw `TADG_BLOCK` 算 trigger。
- `tool_input.tadg_message.startswith("TADG_OVERRIDE_ACCEPTED:")` 算 trigger。
- force-pass 算 trigger。
- 普通 pass 中带 `tool_override_reason` 不算 trigger。

v3.11 的阻塞点不在 telemetry，而在最终行为：TADG/EFG 可以阻止一部分明显不一致的 final answer，但还不能保证 agent 在同类候选、反向空间关系、anchor self、局部 crop 与全局 frame 冲突时稳定选中正确 proposal。v3.19 通过外部 consensus 缓解了这个问题，但还没有把这种选择能力蒸馏回单轨迹工具或 prompt contract。

### P6. 过程文档曾经让结论分散

旧文档里有三类内容混在一起：

- 已修复的 evaluator/runtime bug。
- 已被实验否定的假设。
- 仍然成立的核心瓶颈。

例如早期 `39/15` 数字来自 Phase8 GT-CG bbox 口径，后来 aggregation-GT 修正后不应继续作为当前 headline；`keyframe_funnel` 的价值则是“目标已看见但仍选错”的结构性结论，应保留。本文把这些结论合并到一个入口，旧过程文件删除，避免未来 agent 继续从过时过程结论出发。

## 6. 不应该继续做什么

以下方向会污染目标或已被当前证据压低优先级：

1. 不做 GT-pool ablation 作为主线。它会破坏 detection-mode 定义，只能作为非常明确的诊断上界。
2. 不把 v2 当 zero-shot SOTA 宣称。v2 有 GT view oracle。
3. 不在 `failed_count > 0` 时跑 official ingest / docs harvest；failed sample 必须先重试或分类清楚。
4. 不把 v3.6 ignore-2、任何 partial/random100 子集、或 retry merge 前的 raw pre-aggregation 数字当正式结果。
5. 不继续单纯扩大 keyframe 数量来期待主要提升，除非同时改变最终排序逻辑。
6. 不把 Mask3D label 当硬约束；它只能是弱先验。
7. 不把普通 `tool_override_reason` 计为 TADG trigger。
8. 不手工修某几个 failed/regressed 样本的答案；必须改通用机制并重新跑 full random100。

## 7. 下一步建议：post-v3.20

v3.14、v3.15、v3.16、v3.17 和 v3.18 的 focus gates 已证明一个负面结论：
继续堆依赖旧 trace 形态的窄 guard 或 generic reflection 工具不够稳。v3.19
和 v3.20 的正向结果说明，当前瓶颈更像“单轨迹最终选择不稳定”，而不是
“某一个明显缺失的工具调用”。下一步应该围绕两件事推进：第一，把 v3.20
临时 orchestration 固化为可复跑 full-val consensus runner；第二，从
consensus 的互补样本中蒸馏出单轨迹可用的 code-side proposal ranking /
evidence scoring。

### 7.1 目标

下一版的最低目标：

- random100 `failed_count == 0`。
- 如果是单轨迹改动，不低于 v3.11 single-run headline，并解释是否能缩小到 v3.19 consensus 的差距。
- 如果是 consensus/orchestrator 改动，不低于 v3.20 full-val headline，并
  给出成本、并发、429 rate、RSS 峰值和失败恢复记录。
- 特别关注 Multiple split，而不是只看 Overall。
- 保留 no-match / EFG / TADG / terminal latch 的 durability，不重新引入 same-turn final overwrite 或 `proposal_id=-1` 失败。

理想目标：

- 修复 v3.10/v3.11/v3.13 互补样本中的 paired regressions，同时保住 v3.19 consensus recoveries。
- Multiple@0.50 有清晰提升。
- Tool traces 能解释为什么最终 proposal 赢过相邻干扰物。

### 7.2 候选机制

可以优先考虑以下改动，按侵入性从小到大：

1. Durable full-val consensus orchestrator：把 v3.20 的 checkpoint-only
   batching、RSS guard、source coverage check、consensus build、metrics 和 SQLite
   ingest 固化到 `scripts/`，不要再依赖 `tmp/` launcher。
2. Consensus disagreement audit：优先看 v3.20 的 `consensus_group_size=1`
   以及 source 分歧样本，逐个归因 Stage 1、tool result、guard、final reasoning
   或 metric。
3. Code-side proposal scoring：把 plurality consensus 暴露出的稳定信号转成单轨迹可计算特征，例如 co-visible anchor support、spatial relation score、frame coverage、category/label mismatch penalty。
4. Co-visible anchor ranking：对多干扰物候选，优先找 candidate 与 query anchor 同框的 view，再比较空间关系；避免 v3.4 那种只看 candidate 最可见 crop 的缺陷。
5. Parser fidelity spot-check：只在 Stage 1 明确造成错误时修 parser；不要把所有错误都归因到 retrieval。

### 7.3 测试与评估要求

下一版不能只看两个失败样本。建议 gate：

1. Focused paired set：覆盖 v3.19 consensus recoveries、v3.11/v3.10/v3.13 disagreement cases、scene0354/scene0690 等 still-hard cases。
2. Focused unit tests：覆盖每个新增 guard 或 final-evidence contract，先红后绿。
3. Full random100 fresh/checkpoint-resumed rerun，或可复现 consensus build：最终重新生成 `side_by_side.json`，不能只报告 focused win。
4. Gate：`n=100` 且 `failed_count=0`。
5. aggregation-GT rescore。
6. metrics + SQLite ingest。
7. 新 per-version doc，并同步 README / leaderboard。

任何只在少数样本上通过、但 random100 Overall/Multiple 退化的改动，都不应成为 headline。

## 8. 当前重要 artifacts

### 8.1 正式文档

- `docs/benchmark/scanrefer/README.md`：benchmark timeline 和当前 headline。
- `docs/benchmark/scanrefer/leaderboard.md`：外部方法对比。
- `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md`：v2 full-val upper bound。
- `docs/benchmark/scanrefer/v3p2_callbacks_durable_20260504.md`：runtime/evaluator durability。
- `docs/benchmark/scanrefer/v3p3_vertical_spatial_20260504.md`：旧 honest headline。
- `docs/benchmark/scanrefer/v3p4_select_among_proposals_20260504.md`：negative 1-of-K judge。
- `docs/benchmark/scanrefer/v3p5_cvra_negative_20260505.md`：negative CVRA smoke。
- `docs/benchmark/scanrefer/v3p7_final_selection_20260508.md`：durability win / metric mixed。
- `docs/benchmark/scanrefer/v3p8_relation_ranking_20260508.md`：relation-ranking fresh win。
- `docs/benchmark/scanrefer/v3p9_anchor_guard_20260508.md`：anchor/evidence guard checkpoint。
- `docs/benchmark/scanrefer/v3p10_terminal_latch_20260508.md`：旧单轨迹 headline。
- `docs/benchmark/scanrefer/v3p11_anchor_exclusion_focus13_20260508.md`：v3.11 focus13 ablation 和 rejected EFG 2D-mark experiment。
- `docs/benchmark/scanrefer/v3p11_anchor_exclusion_20260508.md`：当前最佳单轨迹 headline。
- `docs/benchmark/scanrefer/v3p12_vertical_relation_guard_20260508.md`：negative vertical-compare guard。
- `docs/benchmark/scanrefer/v3p13_late_override_20260508.md`：negative late-overwrite guard。
- `docs/benchmark/scanrefer/v3p14_stale_left_right_focus_20260508.md`：rejected one-sample stale-left/right focus gate。
- `docs/benchmark/scanrefer/v3p15_cited_frame_guard_focus_20260508.md`：rejected cited-frame inspected-visibility focus gate。
- `docs/benchmark/scanrefer/v3p16_category_supported_tadg_focus_20260508.md`：rejected category-supported TADG focus gate。
- `docs/benchmark/scanrefer/v3p17_final_revision_guard_focus_20260508.md`：rejected final-revision focus gate。
- `docs/benchmark/scanrefer/v3p18_final_selection_ledger_focus_20260508.md`：rejected final-selection ledger focus gate。
- `docs/benchmark/scanrefer/v3p19_consensus3_20260508.md`：random100 no-GT consensus headline。
- `docs/benchmark/scanrefer/v3p20_consensus3_full_20260511.md`：当前 full-val no-GT consensus headline。

### 8.2 v3.20 artifacts

这些文件是当前 v3.20 full-val consensus gate 的原始材料：

- `tmp/v3p20_full_source_a_eval_agg_gt/`
- `tmp/v3p20_full_source_b_eval_agg_gt/`
- `tmp/v3p20_full_source_c_eval_agg_gt/`
- `tmp/v3p20_consensus3_full_eval/`
- `tmp/v3p20_consensus3_full_eval/leaderboard_metrics.json`
- `tmp/scanrefer_artifacts/full_val_sample_ids.json`
- `docs/benchmark/scanrefer/runs.sqlite` run id `v3p20_consensus3_full_20260511`

### 8.3 v3.19 artifacts

这些文件是当前 v3.19 consensus gate 的原始材料：

- `tmp/v3p19_consensus3_random100_eval_agg_gt/`
- `tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt/`
- `tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt/`
- `tmp/v3p13_late_override_random100_eval_agg_gt/`
- `tmp/scanrefer_artifacts/random100_sample_ids.json`
- `docs/benchmark/scanrefer/runs.sqlite` run id `v3p19_consensus3_random100_20260508`

### 8.4 v3.11 artifacts

这些文件是当前最佳单轨迹 v3.11 gate 的原始材料：

- `tmp/v3p11_tadg_anchor_exclusion_random100_eval/`
- `tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt/`
- `tmp/v3p11_tadg_anchor_exclusion_random100_eval.log`
- `tmp/scanrefer_artifacts/random100_sample_ids.json`
- `docs/benchmark/scanrefer/runs.sqlite` run id `v3p11_tadg_anchor_exclusion_random100_20260508`

### 8.5 注意：full-run 内存处理

v3.20 前的 `scanrefer_leaderboard_metrics` 走 `ScanRefVGDataset.from_path`
时会加载 Phase8 gzip scene object pickles，并按 scene 缓存 `objs`。在
98 samples / 65 scenes 这类切片上也可能出现非常高的内存占用。v3.20 已把
metrics 改成 streaming per-sample 读取 + lightweight metadata loader，把
runner 改成 checkpoint-only batch + bounded caches。后续 full-run 仍应放在
tmux 中，并记录 RSS 峰值、429 retry rate 和 source coverage。

SQLite 本身也需要控体积：v3.7-v3.18 的中间开发折 tool response 文本过长，
完整写入会把 `runs.sqlite` 推到 100MB 以上。当前 DB 保留 run/sample rows、
tool-call rows、tool names 和 tool inputs；这些中间 run 的超长
`tool_calls.response_text` 会被截断成可审计摘要。完整 raw artifact 路径以
各 per-version doc 为准。v3.19/v3.20 consensus trace 本身就是 compact
`consensus_select`，不需要截断。

## 9. 当前 north star

ScanRefer 主线接下来应该围绕一个问题推进：

> 当正确 proposal 已在 Mask3D pool 中、目标也已出现在 agent 看过的 evidence 中时，agent 为什么仍然没能把自然语言描述、视觉外观、空间 referent 和 detector proposal 对齐？

只要这个问题不解决，继续扩大 retrieval、增加候选、或增加单帧 VLM judge 都很可能重演 v3.4/v3.5 的负结果。v3.19 和 v3.20 说明多轨迹 consensus 能缓解 final-selection 抖动；真正有价值的下一版应该要么把 v3.20 consensus 做成可复跑、可成本核算的工程流程，要么把 consensus 的信号蒸馏成单轨迹更稳定的 proposal selection，尤其是在 Multiple split、label mismatch、spatial anchor 依赖强的样本上。
