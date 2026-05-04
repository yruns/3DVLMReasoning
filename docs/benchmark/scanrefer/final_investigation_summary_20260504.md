# ScanRefer 排查总报告 - 2026-05-04

## 最终结果

这轮排查把 ScanRefer query-driven random100 的可信结果从最初文档里引用的
`39.0 / 15.0` 修正并提升到了：

| 版本 | 评估口径 | Acc@0.25 | Acc@0.50 | mean IoU | 说明 |
|---|---|---:|---:|---:|---|
| v3.1 原始引用 | Phase8 GT-CG bbox | 39.0 | 15.0 | 0.1976 | 旧诊断文档中的数字，后续确认不是 paper-comparable 口径 |
| v3.1 corrected | ScanNet aggregation GT | 45.0 | 39.0 | 0.3820 | 同一批预测重新用 aggregation GT 打分 |
| v3.2 callback-durable | ScanNet aggregation GT | 46.0 | 40.0 | 0.3857 | 修复 callback 图片注入、stride/path/extractor/trace durability |
| **v3.3 vertical-spatial** | **ScanNet aggregation GT** | **48.0** | **42.0** | **0.4130** | 增加 `above` / `below` 空间工具关系并重跑 random100 |

v3.3 的分列指标：

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 75.00 | 37.50 | **48.00** |
| Acc@0.50 | 75.00 | 29.17 | **42.00** |

结论很明确：

- `39/15` 不是当前 query-driven pipeline 的真实 paper-comparable 状态；主要是评估 GT bbox 口径错了。
- v3.3 相比修正后的 v3.1 是 `+3pp / +3pp`，相比 v3.2 是 `+2pp / +2pp`。
- v3.x 仍显著落后 v2 random100 GT-view-oracle 上界 `67/59`，差距约 `19pp / 17pp`。
- 剩余瓶颈仍然是 multiple / multi-distractor proposal ranking，不是 initial keyframe 是否覆盖同类候选。

## 本次实际做过的尝试

### 1. 重新审计旧诊断文档

起点文件是：

- `docs/benchmark/scanrefer/picking_error_diagnosis_20260503.md`

我没有直接接受里面的猜想，而是按代码、数据、artifact、SQLite 重新核对。旧文档的问题是把 v3.1 的 `39/15` 当作主要模型失败，但该数字使用的是 Phase8 GT-CG bbox。这个 bbox 与 ScanRefer / Mask3D 相关论文常用的 ScanNet aggregation GT bbox 不一致。

验证方式：

- 对 `tmp/v3p_random100_eval/side_by_side.json` 做 aggregation-GT rescore。
- 写入 SQLite run：`v3p_agg_gt_random100_20260503`。

结果：

```text
v3p_smoke_random100_20260503      | 100 | 0.3900 | 0.1500 | 0.1976
v3p_agg_gt_random100_20260503     | 100 | 0.4500 | 0.3900 | 0.3820
```

结论：第一优先级问题不是模型突然只有 15% Acc@0.50，而是 evaluator/GT-source 口径错了。

### 2. 检查 proposals / detector / benchmark 口径

确认当前 ScanRefer zero-shot track 仍按 Mask3D-pool detection-mode 来做：

- proposal pool 来自 Mask3D / ConceptGraph 对象。
- 不做 GT-pool ablation，因为这会破坏 Camp-A detection-mode 定义。
- v2 虽然 bbox 评估口径 paper-comparable，但 keyframe selection 读了 GT `target_id` visibility，因此只是 GT-view-oracle 上界，不应当作为 apples-to-apples zero-shot headline。

结论：保留 Mask3D-pool 是正确方向；不应通过换成 GT proposals 来“提升”指标。

### 3. 修复 Stage 2 callback 图片注入

发现的问题：

- `view_keyframe_marked` 会把新图片放到 `vg_pending_images`。
- 但 agent 同一 turn 内如果又调用 `submit_final`，旧 run loop 会立即接受 final submission。
- 结果是：工具返回文本说有新图片，但真正的像素还没被注入下一轮，final 就结束了。

修复：

- `src/agents/stage2_deep_agent.py`
- `src/agents/runtime/deepagents_agent.py`

行为改为：

1. 如果 `submit_final` 时发现 `runtime.consume_evidence_update()` 为真；
2. 构造 evidence update message，把新图片注入；
3. 清空 `runtime.final_submission`；
4. 继续下一轮，让模型看完图片后重新 submit。

同时更新 evidence update prompt，明确告诉模型：如果刚才已经 submit 但还没看到新图，该提交未被接受。

验证：

- 新增测试 `test_run_defers_submit_final_until_pending_images_are_injected`。
- 100-case run 日志里可以看到 `deferring submit_final until newly queued visual evidence is injected`。

结果：

- v3.2 aggregation-GT：`46.0 / 40.0`。
- 这是 correctness fix，同时也让 callback 图片真正进入模型上下文。

### 4. 修复 ScanRefer callback selector 的 stride 错配

发现的问题：

- ScanRefer / NR3D 本地 visibility index 是 `stride=1`。
- callback 里构造 `KeyframeSelector` 时用了 `stride=10`。
- 这会让 callback 产出的 view/frame id 与本地 raw frame / visibility 文件错位。

修复：

- `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
- `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`

结论：这是 callback 可用性的硬 bug。修复后 Stage 1 callback 返回的 frame id 更稳定。

### 5. 修复 raw frame 路径解析

发现的问题：

- `KeyframeSelector._set_image_paths()` 只兼容旧的 `*-rgb.jpg`。
- 本地 ScanRefer / NR3D raw layout 是 `*-rgb.png`。
- 这会导致 callback 选出 view 后无法找到真实图片路径。

修复：

- `src/query_scene/keyframe_selector.py`

新增 PNG fallback，并用测试覆盖：

- `test_set_image_paths_accepts_raw_rgb_png_layout`

结论：这是另一个 callback 图片不可用的基础设施 bug。

### 6. 修复 structured `proposal_id` finalizer payload 解析

发现的问题：

- 某些 sample 的最终 payload 只有结构化 `proposal_id`，没有直接的 `bbox_3d`。
- 旧 `extract_pack_v1_prediction()` 只认 `bbox_3d` / `selected_object_id`，因此把这类样本当作失败。

修复：

- `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
- 从 `result.final_bundle.extra_metadata["vg_proposal_pool"]["proposals"]` 里按 `proposal_id` 回填 bbox。

验证：

- `test_extract_pack_v1_prediction_resolves_structured_proposal_payload`

结论：这是 evaluation extraction bug，修复后减少无意义 failed sample。

### 7. 增加 per-sample tool trace 持久化

旧状态：

- `tool_calls` / `llm_calls` SQLite 为空。
- 很难做 per-question regression analysis。

修复：

- `run_scanrefer_vg_side_by_side.py` 把 `tool_trace` 写进 side-by-side / checkpoint。
- `scripts/ingest_scanrefer_run.py` 读取 `tool_trace` 并写入 SQLite `tool_calls`。

结果：

```text
v3p2_callbacks_durable_random100_20260504 | tool_calls = 1899
v3p3_vertical_spatial_random100_20260504  | tool_calls = 1891
```

结论：后续能按 qid / tool_name 做真实 failure analysis，不再只依赖 console grep。

### 8. 修复 leaderboard random100 显式切片

发现的问题：

- leaderboard aggregator 原先没有明确接收 random100 sample ids。
- 这类工具容易在 full val / subfold 之间产生口径不清。

修复：

- `src/evaluation/scripts/scanrefer_leaderboard_metrics.py`
- 增加 `--sample-ids`。

验证：

- `test_load_sample_id_filter_accepts_runner_json_shapes`

结论：之后 random100 的 Unique / Multiple / Overall 指标可复现，不依赖隐式默认行为。

### 9. 更新 full v3 driver，强制 aggregation-GT rescore + SQLite ingest

修复：

- `scripts/run_scanrefer_v3_query_driven.sh`

现在 full v3 driver 会按顺序执行：

1. pack-prep
2. agent run
3. aggregation-GT rescore
4. leaderboard metrics
5. SQLite ingest

结论：后续不会再把 Phase8 GT-CG smoke number 当成 headline。

### 10. 增加 `above` / `below` 空间关系工具

发现的问题：

- 多个失败样本包含 vertical language，例如 `above` / `below` / `under`。
- 原工具只有 `closest_to` / `farthest_from`。
- 模型会把 “cabinet above refrigerator” 这种关系硬套成 `closest_to`，导致排序语义错误。

修复：

- `src/agents/packs/vg_embodiedscan/tools.py`
- `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`

新增：

- `relation="above"`：按 candidate bbox center 相对 anchor 的正 z offset 排序。
- `relation="below"`：按负 z offset 排序。
- 返回 `vertical_offsets` 和 `horizontal_distances`，便于 trace 解释。

测试：

- `test_compare_proposals_spatial_above_uses_z_axis`
- `test_compare_proposals_spatial_below_uses_z_axis`

100-case 验证：

- run id：`v3p3_vertical_spatial_random100_20260504`
- aggregation-GT：`48.0 / 42.0`
- tool trace 中关系调用：

```text
closest_to: 50
farthest_from: 8
above: 4
below: 2
```

结论：这是本轮除 evaluator 修正外最明确的正向功能改动。它不是完全解决 multi-distractor ranking，但在 random100 上带来净 `+2pp / +2pp`。

## 每轮实验结果

### v3.1 original smoke

| 指标 | 数值 |
|---|---:|
| Acc@0.25 | 39.0 |
| Acc@0.50 | 15.0 |
| mean IoU | 0.1976 |
| 问题 | 使用 Phase8 GT-CG bbox，不是 paper-comparable |

这个结果保留为 audit trail，不再作为真实 headline。

### v3.1 corrected

| 指标 | 数值 |
|---|---:|
| Acc@0.25 | 45.0 |
| Acc@0.50 | 39.0 |
| mean IoU | 0.3820 |

关键结论：评估口径修正本身解释了大部分 `39/15` 异常。

### v3.2 callback-durable

| 指标 | 数值 |
|---|---:|
| Acc@0.25 | 46.0 |
| Acc@0.50 | 40.0 |
| mean IoU | 0.3857 |
| Unique@0.50 | 71.43 |
| Multiple@0.50 | 27.78 |
| no_prediction | 2 |
| tool_calls | 1899 |

关键结论：runtime/callback/extractor 修复是必要的 correctness work，但 headline 只小幅提升。

### v3.3 vertical-spatial

| 指标 | 数值 |
|---|---:|
| Acc@0.25 | 48.0 |
| Acc@0.50 | 42.0 |
| mean IoU | 0.4130 |
| Unique@0.50 | 75.00 |
| Multiple@0.50 | 29.17 |
| no_prediction | 5 |
| tool_calls | 1891 |

相对 v3.2：

- Acc@0.25：`+2pp`
- Acc@0.50：`+2pp`
- mean IoU：`+0.0273`
- Acc@0.50 gains：7 个样本从 `<0.50` 到 `>=0.50`
- Acc@0.50 losses：5 个样本从 `>=0.50` 到 `<0.50`
- net：`+2` 个 Acc@0.50 样本

注意：最初 motivating case `scene0011_00::20::2` 并没有因此变好。它确实使用了 `above`，但选到 proposal 41，aggregation-GT IoU 变成 0.0。v3.3 的总提升来自整体 rerun + 新 tool affordance 的净效果，不是这个单例被修好。

## 最终结论

### 结论 1：旧文档的核心数字需要降级为 audit trail

`39/15` 不能作为当前 query-driven ScanRefer 的最终结论。它暴露了问题，但真实 paper-comparable baseline 是 aggregation-GT rescore 后的 `45/39`。

### 结论 2：initial keyframe coverage 不是主因

v3.1 的 Mask3D-CG candidate visibility 已经把 pack fallback 从 v3 的 38% 降到 0%，但 corrected headline 仍只有 `45/39`。所以“初始 keyframe 看不到同类目标”不是主要瓶颈。

### 结论 3：callback 基础设施之前确实有硬 bug

包括：

- same-turn `submit_final` 抢在新图片注入前结束；
- callback selector stride 错；
- raw PNG 路径不兼容；
- structured `proposal_id` payload 被当失败；
- `tool_calls` 没有持久化。

这些 bug 修复后，系统更可信，也小幅提升到 `46/40`。

### 结论 4：多实例候选排序仍是主要瓶颈

v3.3 后：

- Unique@0.50 = 75.00
- Multiple@0.50 = 29.17

差距集中在 Multiple split。也就是说，模型大多能处理 near-unique reference，但在同类对象很多时，仍然经常选错 proposal。

### 结论 5：空间工具还不够，但增加 vertical relation 有正收益

`above` / `below` 带来 `+2pp / +2pp`，说明工具 affordance 能影响 agent 行为。后续如果继续推进，优先方向应当是更系统地支持：

- camera/view-relative left/right/front/behind；
- ordinal / row language，例如 “fourth from the right”；
- compound anchor disambiguation，例如 “chair next to the desk near the window”；
- between / closest-to-both；
- relation tool 输出可视化 frame refs，减少纯 bbox-center 排序误导。

### 结论 6：v2 仍只能作为上界，不是公平 headline

v2 full val `69.92 / 62.79` 和 random100 `67/59` 都使用 GT-view-oracle keyframe selector。它证明 Mask3D pool + Stage 2 agent 有上界空间，但不能作为 zero-shot Camp-A apples-to-apples 结果。

## 当前 artifact 与复现入口

主要文档：

- `docs/benchmark/scanrefer/v3p1_mask3d_query_driven_20260503.md`
- `docs/benchmark/scanrefer/v3p2_callbacks_durable_20260504.md`
- `docs/benchmark/scanrefer/v3p3_vertical_spatial_20260504.md`
- `docs/benchmark/scanrefer/leaderboard.md`
- `docs/benchmark/scanrefer/README.md`

核心 raw artifacts：

- v3.1 original：`tmp/v3p_random100_eval/`
- v3.1 aggregation-GT：`tmp/v3p_random100_eval_agg_gt/`
- v3.2：`tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt/`
- v3.3：`tmp/v3p3_vertical_spatial_random100_eval_agg_gt/`
- frozen fold：`tmp/scanrefer_artifacts/random100_sample_ids.json`
- SQLite：`docs/benchmark/scanrefer/runs.sqlite`

关键 SQLite 查询：

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p_smoke_random100_20260503',
  'v3p_agg_gt_random100_20260503',
  'v3p2_callbacks_durable_random100_20260504',
  'v3p3_vertical_spatial_random100_20260504'
)
ORDER BY run_id;
```

当前实测输出：

```text
v3p_smoke_random100_20260503              | 100 | 0.3900 | 0.1500 | 0.1976
v3p_agg_gt_random100_20260503             | 100 | 0.4500 | 0.3900 | 0.3820
v3p2_callbacks_durable_random100_20260504 | 100 | 0.4600 | 0.4000 | 0.3857
v3p3_vertical_spatial_random100_20260504  | 100 | 0.4800 | 0.4200 | 0.4130
```

## 本轮验证

代码级验证：

```bash
PYTHONPATH=src .venv/bin/pytest \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py \
  src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py \
  src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py \
  src/evaluation/scripts/tests/test_ingest_scanrefer_run.py \
  src/query_scene/tests/test_keyframe_selector_hypothesis.py::TestKeyframeSelectorHypothesis::test_set_image_paths_accepts_raw_rgb_png_layout \
  src/agents/tests/test_stage2_deep_agent.py::TestStage2DeepAgent::test_run_defers_submit_final_until_pending_images_are_injected \
  -q
```

结果：

```text
48 passed in 5.70s
```

Lint：

```bash
.venv/bin/ruff check <touched python files>
```

结果：

```text
All checks passed!
```

Patch whitespace：

```bash
git diff --check
```

结果：无输出，clean。

## 后续建议

如果继续提高 ScanRefer，优先级建议如下：

1. **做 multi-distractor relation toolkit**：left/right/front/behind/ordinal/between，不要继续只靠 `closest_to`。
2. **把 relation tool 和可视化 evidence 绑定**：工具返回 ranked ids 后，同时推荐包含 candidate+anchor 的 frames，逼 agent 做 visual cross-check。
3. **减少 no-prediction**：v3.3 有 5 个 no-prediction，虽然总指标提升，但这是可修的稳定性问题。
4. **做 random100 variance run**：v3.3 是单次 LLM run；建议同代码跑 2-3 次估计方差。
5. **只有 random100 明显继续提升后再跑 full 9508**：当前 Multiple@0.50 只有 29.17，full run 成本高，先在 development fold 上继续攻 multiple split 更划算。
