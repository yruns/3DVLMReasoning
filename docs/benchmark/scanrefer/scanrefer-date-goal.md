# ScanRefer Current Goal and Problem Statement - 2026-05-07

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
- 先在 frozen random100 上稳定提升和诊断，再扩到 full val 9508 条。

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

### 2.2 当前可信 query-driven headline：v3.3

当前最 honest 的 query-driven / Mask3D-pool random100 结果是 v3.3：

| Split | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|
| Overall | 48.00 | 42.00 | 0.4130 |
| Unique | 75.00 | 75.00 | - |
| Multiple | 37.50 | 29.17 | - |

它是当前 apples-to-apples 开发折上的最好正式结果。v3.4、v3.5 都是负结果，没有替代 v3.3。

### 2.3 最新 v3.6/TADG：未形成正式结果

v3.6 TADG random100 已跑到 `n=100`，但最终 `failed_count=2`：

| Sample | Final failure | Important diagnosis |
|---|---|---|
| `scannet/scene0474_00::15::3` | agent submitted `proposal_id=-1` | pool 中存在 GT-matching proposal `pid=6`, label `chair`, IoU 0.3538 |
| `scannet/scene0678_00::34::3` | agent submitted `proposal_id=-1` | pool 中存在 GT-matching proposal `pid=10`, label `shelf`, IoU 0.7851 |

这两个失败不是 HTTP 504、mesh root、empty proposal pool，也不是 evaluator 找不到 GT。它们是 agent no-match / label-mismatch selection failures：正确或足够接近的 proposal 在 pool 里，但 agent 没有把它识别为描述对象，最终选择了 `-1`。

因此 v3.6 当前不能作为正式 benchmark 结果：

- 未跑 official agg-GT metrics。
- 未跑 SQLite ingest。
- 未写 per-version durable benchmark doc。
- 不应在 README/leaderboard 中作为 official row。

为了快速分析，我们曾忽略这 2 个失败样本计算过 98-sample 指标，但这个数字只能作为诊断：

| Run | Denominator | Acc@0.25 | Acc@0.50 | Mean IoU | Status |
|---|---:|---:|---:|---:|---|
| v3.3 same-98 baseline | 98 | 47.96 | 41.84 | 0.4125 | diagnostic slice |
| v3.6 TADG ignore-2 | 98 | 46.94 | 42.86 | 0.4106 | diagnostic only |

结论：即使忽略两个失败，v3.6 也不是清晰 win；它在 Acc@0.50 上 +1.02pp，但 Acc@0.25 -1.02pp、mean IoU 略低。TADG 当前更像暴露了 no-match / label-mismatch 问题，而不是已经解决了主瓶颈。

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

目前可以按 9 个版本/尝试来理解：

| Version | Status | Key change | Headline / diagnosis |
|---|---|---|---|
| v1 | historical | 第一版 ScanRefer full eval | 使用 Phase8 GT-CG bbox，bbox 口径不 paper-comparable |
| v2 | official upper bound | aggregation-GT bbox 修正 | full val 69.92 / 62.79，但有 GT view oracle |
| v3 | historical query-driven | query-driven 初版 | 后来在 v3.2 runtime 修复下 aggregation-GT 纠正为 47 / 40 |
| v3.1 | official smoke | Mask3D-CG visibility keyframes | aggregation-GT corrected 45 / 39；X1 不成立 |
| v3.2 | official | callback / evaluator durability 修复 | random100 46 / 40 |
| v3.3 | official headline | vertical spatial relations | random100 48 / 42，当前 honest 最好 |
| v3.4 | negative | `select_among_proposals` forced 1-of-K | random100 44 / 38，退化 |
| v3.5 | negative smoke | CVRA visible non-label proposal augmentation | smoke6 retrieval recall 高，但 0/6 F5a flip；未 full random100 |
| v3.6 | blocked | TADG, CVRA off | n=100 但 failed=2，未 harvest |

正式文档化到 v3.5；v3.6 是当前正在尝试的第 9 个版本/方向，但还没有形成正式结果。

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

Unique 和 Multiple 的差距非常大：

- v3.3 Unique@0.50 = 75.00
- v3.3 Multiple@0.50 = 29.17

Unique 高说明当 scene 中同类干扰少时，当前 evidence + agent 能工作。Multiple 低说明问题集中在：

- 同类/近类 proposal 很多。
- query 的空间关系、相对位置、局部外观需要同时满足。
- agent 容易被 label、单帧显著性、局部 crop 误导。
- 一旦工具返回多个候选，agent 没有稳定地把描述约束转成排序。

v3.4 的 `select_among_proposals` 失败进一步说明：简单把候选集合交给一个 VLM 做 1-of-K，不足以解决问题。原因是候选最可见 frame 不一定包含描述中的 anchor；VLM 看 candidate crop 时可能看不到“left of entrance / above refrigerator / right of bookcase”这类关系。

### P2. Label mismatch / no-match 是最新暴露出的硬问题

v3.6 的两个剩余失败都属于“pool 中有 GT-matching proposal，但 agent 认为没有匹配目标”：

- `scene0474_00::15::3`：query 描述 floor-resting high-back chair cushion / whiteboard / bookcase 关系；pool 中 `chair` proposal IoU 0.3538，但 agent 最终 `-1`。
- `scene0678_00::34::3`：query 描述 snack machine / beverage machine / entrance 关系；pool 中 `shelf` proposal IoU 0.7851，但 agent 最终 `-1`。

这说明不能把 proposal label 当硬语义真值。Mask3D / ScanNet200 label 可能和 ScanRefer reference phrase 不一致：

- 物体可能被标成 `shelf`，但人类描述为 snack machine。
- 物体可能被标成 `chair`，但描述强调 cushion、back、location 或上下文 anchor。
- object name、semantic class、visual appearance、spatial referent 四者并不总是同一个词。

下一版如果继续沿用“找不到严格语义 label 就提交 `-1`”的策略，会继续在这类样本上失败。

### P3. Callback recovery 的上限比 picking 修复低

v3.2 之后 callback durability、raw PNG path、structured payload extraction、tool trace persistence 都已经修复。v3.6 又修了 ScanRefer hypothesis callback 中 `use_visual_context=False` 的 text-only re-query wiring，mesh-root failures 也已清掉。

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

v3.6 的阻塞点不在 telemetry，而在最终行为：agent 可以在有候选的情况下提交 `-1`。TADG 如果只是拦“工具 top-1 与 final answer 不一致”，但没有对 no-match / OOD / label mismatch 做明确闭环，仍然无法保证 failed=0 或提升 final proposal selection。

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
3. 不在 `failed_count > 0` 时跑 official ingest / docs harvest。
4. 不把 v3.6 ignore-2 当正式结果。
5. 不继续单纯扩大 keyframe 数量来期待主要提升，除非同时改变最终排序逻辑。
6. 不把 Mask3D label 当硬约束；它只能是弱先验。
7. 不把普通 `tool_override_reason` 计为 TADG trigger。
8. 不手工修两个 failed 样本的答案；必须改通用机制并重新跑 full random100。

## 7. 下一版建议：v3.7 focus

v3.7 应该围绕 label-noise / no-match / multi-distractor final selection 做小而硬的改动。目标不是堆更多工具，而是让 agent 在提交最终答案前完成一个可验证的闭环。

### 7.1 目标

v3.7 的最低目标：

- random100 `failed_count == 0`。
- 不降低 v3.3 headline，至少在 same fold 上有 paired improvement。
- 特别关注 Multiple split，而不是只看 Overall。
- 对 `proposal_id=-1` 建立 guard，避免有可见候选时轻易 no-match。

理想目标：

- 修复 v3.6 两个 no-match 样本，同时不引入 false positive。
- Multiple@0.50 有清晰提升。
- Tool traces 能解释为什么最终 proposal 赢过相邻干扰物。

### 7.2 候选机制

可以优先考虑以下改动，按侵入性从小到大：

1. Prompt/playbook 修正：明确 proposal label 是弱先验；reference phrase 和 detector label 不一致时，应以 visual evidence + spatial referent 为准。
2. Final-submit guard：当 agent 想提交 `-1` 时，检查是否已经看过候选 proposal、是否有 near-category / spatially plausible proposal、是否有 high-IoU-like candidate evidence；若存在候选，要求 agent 先给出最接近 proposal 并说明 reject reason。
3. No-match 二阶段：把 `-1` 改成需要二次确认的特殊路径，而不是普通 final answer。二次确认必须引用 evidence 缺口，例如目标不可见、pool 确实没有对应实例、空间关系全部矛盾。
4. Category synonym / OOD bridge：对 `snack machine` vs `shelf`、`cushion` vs `chair` 这类 reference phrase，允许从 visual/anchor 角度召回候选，而不是只依赖 object label。
5. Co-visible anchor ranking：对多干扰物候选，优先找 candidate 与 query anchor 同框的 view，再比较空间关系；避免 v3.4 那种只看 candidate 最可见 crop 的缺陷。

### 7.3 测试与评估要求

v3.7 不能只看两个失败样本。建议 gate：

1. Focused test：覆盖 ScanRefer hypothesis callback 保持 text-only re-query，不回退到 mesh/BEV。
2. Focused no-match regression：构造或复用 v3.6 两个 case 的 trace/pool 条件，验证 final `-1` guard 会触发。
3. Full random100 cached rerun：删除需要重跑的 checkpoint，不改成功样本；最终重新生成 `side_by_side.json`。
4. Gate：`n=100` 且 `failed_count=0`。
5. aggregation-GT rescore。
6. metrics + SQLite ingest。
7. 新 per-version doc。

任何只在两个样本上通过、但 random100 Overall/Multiple 退化的改动，都不应成为 headline。

## 8. 当前重要 artifacts

### 8.1 正式文档

- `docs/benchmark/scanrefer/README.md`：benchmark timeline 和当前 headline。
- `docs/benchmark/scanrefer/leaderboard.md`：外部方法对比。
- `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md`：v2 full-val upper bound。
- `docs/benchmark/scanrefer/v3p2_callbacks_durable_20260504.md`：runtime/evaluator durability。
- `docs/benchmark/scanrefer/v3p3_vertical_spatial_20260504.md`：当前 honest headline。
- `docs/benchmark/scanrefer/v3p4_select_among_proposals_20260504.md`：negative 1-of-K judge。
- `docs/benchmark/scanrefer/v3p5_cvra_negative_20260505.md`：negative CVRA smoke。

### 8.2 v3.6 临时 artifacts

这些文件是诊断材料，不是正式 benchmark doc：

- `tmp/v3p6_tadg_surface26_handoff.md`
- `tmp/v3p6_tadg_telemetry_semantics_audit.md`
- `tmp/v3p6_tadg_random100_eval/side_by_side.json`
- `tmp/v3p6_tadg_random100_eval/resume_20260507_201053.log`
- `tmp/v3p6_tadg_random100_eval/retry_20260507_210311.log`
- `tmp/v3p6_tadg_random100_eval_ignore2_agg_gt/lightweight_leaderboard_metrics.json`

### 8.3 注意：metrics 脚本内存问题

`scanrefer_leaderboard_metrics` 走 `ScanRefVGDataset.from_path` 时会加载 Phase8 gzip scene object pickles，并按 scene 缓存 `objs`。在 98 samples / 65 scenes 这类切片上也可能出现非常高的内存占用。做诊断切片时应优先用已有 pack sample 的 `is_unique` 和 per-sample IoU 做 lightweight calculation；正式 full metrics 则应放到 tmux 中并监控内存。

## 9. 当前 north star

ScanRefer 主线接下来应该围绕一个问题推进：

> 当正确 proposal 已在 Mask3D pool 中、目标也已出现在 agent 看过的 evidence 中时，agent 为什么仍然没能把自然语言描述、视觉外观、空间 referent 和 detector proposal 对齐？

只要这个问题不解决，继续扩大 retrieval、增加候选、或增加单帧 VLM judge 都很可能重演 v3.4/v3.5 的负结果。真正有价值的 v3.7 应该让最终 proposal selection 更稳定，尤其是在 Multiple split、label mismatch、spatial anchor 依赖强的样本上。
