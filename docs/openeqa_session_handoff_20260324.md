# OpenEQA Session Handoff (2026-03-24)

本文是本次 OpenEQA / Stage 1 / Stage 2 工作会话的详细交接文档。目标是让下一个 code agent 进入仓库后，不需要重建上下文，就能直接接力继续做实验、清理代码或扩展评测。

---

## 1. 本次会话的主任务

本次会话实际覆盖了 5 条主线：

1. 理解当前项目的 Stage 1 / Stage 2 结构，以及本地 OpenEQA 数据现状。
2. 补齐本地 OpenEQA raw 图像数据，把官方原始帧整理到每个 scene 的 `raw/` 目录。
3. 基于本地 prepared scenes 跑通单 scene 的 Stage 1、Stage 2、以及端到端 pipeline，并扩成小批量 pilot。
4. 下载官方 `open-eqa-v0.json`，把“官方 OpenEQA question -> 本地 scene -> 单题 / 小批量 pilot”接起来，并接入官方评测逻辑。
5. 调查 Stage 1 parser 延迟，确认 gemini pool 的可用性，并将“所有非 Agents 的 LLM 默认走 gemini pool”这条约束真正落到代码里。

---

## 2. 已完成事项

### 2.1 项目结构和研究链路已摸清

已经确认：

- Stage 1 主入口是 [`select_keyframes_v2()`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/keyframe_selector.py#L1389)。
- Stage 2 运行时主体是 [`DeepAgentsStage2Runtime`](/Users/bytedance/project/3DVLMReasoning/src/agents/runtime/deepagents_agent.py#L33)，`Stage2DeepResearchAgent` 主要是兼容 wrapper。
- Stage 1 -> Stage 2 的桥接入口是 [`build_stage2_evidence_bundle()`](/Users/bytedance/project/3DVLMReasoning/src/agents/stage1_adapters.py#L54)。

结论：

- Stage 1 是 evidence retrieval，不是最终答题。
- Stage 2 把 Stage 1 当 soft prior，不把它当真值。
- 当前仓库已经适合做 scene-level 方法实验，但本地 `data/OpenEQA` 不是官方 benchmark repo 原生目录布局。

---

### 2.2 本地 OpenEQA raw 数据已补齐

这部分工作已经完成，且本地核查过。

结果：

- `data/OpenEQA/scannet/` 下本地共有 `89` 个 scene。
- 每个 scene 现在都具备：
  - `conceptgraph/`
  - 同级 `raw/`
- 原先残留的失效 symlink 已清理。

本地数据结论：

- 就“prepared-scene / local full-pipeline pilot”而言，原始图像数据已经准备好了。
- 但就“直接喂官方 `OpenEQADataset.from_path()` 跑官方 repo benchmark”而言，仍需要 question / episode adapter，而不是直接指向 `data/OpenEQA/scannet/*/conceptgraph`。

相关说明文档：

- [`current_repo_state.md`](/Users/bytedance/project/3DVLMReasoning/docs/current_repo_state.md)

---

### 2.3 已新增单 scene pilot，并完成 Stage 1 / Stage 2 / E2E 验证

新增脚本：

- [`openeqa_single_scene_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_single_scene_pilot.py#L1)

作用：

- 不改原始数据。
- 在 `tmp/` 下创建 runtime overlay。
- 将 `raw/*-rgb.png` 映射成 Stage 1 所需的 `results/frameXXXXXX.jpg`
- 将 `raw/*-depth.png` 映射成 `results/depthXXXXXX.png`
- 分别跑：
  - `stage1`
  - `stage2`
  - `e2e`

关键逻辑：

- overlay 构建：[`ensure_runtime_scene()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_single_scene_pilot.py#L165)
- Stage 1 入口：[`run_stage1()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_single_scene_pilot.py#L263)
- Stage 2 / E2E 入口：[`run_stage2()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_single_scene_pilot.py#L298)

已实跑 artifact：

- [`stage1.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_pilot_runs/124-scannet-scene0131_02/stage1.json)
- [`stage2.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_pilot_runs/124-scannet-scene0131_02/stage2.json)
- [`e2e.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_pilot_runs/124-scannet-scene0131_02/e2e.json)

另做过一次 stress E2E：

- [`stress e2e.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_pilot_runs_stress/124-scannet-scene0131_02/e2e.json)

关键结论：

- 单 scene pipeline 是可跑的。
- `request_more_views` 的闭环已被实测触发。
- Stage 2 单独跑固定证据时会诚实地返回 `insufficient_evidence`，E2E 接上回调后可以补证据并完成。

---

### 2.4 已新增小批量 batch pilot，并实跑 5-scene 验证

新增脚本：

- [`openeqa_batch_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_batch_pilot.py#L1)

作用：

- 从本地 `conceptgraph` 自动生成 query 候选。
- 用真实 Stage 1 验证 query 可用性。
- 生成 query set。
- 对小批量 scene 逐个跑 Stage 2 和 E2E。

已实跑结果：

- query set: [`query_set.jsonl`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_batch_pilot_5/query_set.jsonl)
- batch summary: [`batch_summary.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_batch_pilot_5/batch_summary.json)

5-scene 小批量结论：

- Stage 1: `5/5` 成功
- Stage 2-alone: `5/5` 都是 `insufficient_evidence`
- E2E: `4/5` 变为 `completed`
- 说明 Stage 2 的价值主要在“请求更多证据”，而不是静态地消费 Stage 1 初始结果

---

### 2.5 官方 `open-eqa-v0.json` 已下载并接入本地 pipeline

官方数据文件现在在：

- [`data/open-eqa-v0.json`](/Users/bytedance/project/3DVLMReasoning/data/open-eqa-v0.json)

已确认：

- 全量 `1636` 条问题
- 其中 `1079` 条是 `scannet-v0`
- 本地 `89/89` 个 ScanNet clip 都能从 `episode_history` 映射到 `data/OpenEQA/scannet/<clip_id>`

此外已经补了 loader 兼容：

- [`openeqa_loader.py`](/Users/bytedance/project/3DVLMReasoning/src/benchmarks/openeqa_loader.py#L135)

当前 loader 支持：

- `data_root / "data" / "open-eqa-v0.json"`
- `data_root / "open-eqa-v0.json"`

并且会在缺少 `scene_id` 时，从 `episode_history` 推出本地 scene。

---

### 2.6 已新增官方 question adapter，支持官方题目驱动 Stage 1 / Stage 2 / E2E

新增脚本：

- [`openeqa_official_question_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py#L1)

它当前支持：

- `--question-id`
- `--clip-id`
- `--category`
- `--max-samples`
- `--require-stage1-success`
- `--unique-scenes`
- `--evaluate`
- `--eval-model`
- `--official-repo-root`

关键逻辑：

- 官方 ScanNet question 过滤与 clip 映射：[`load_official_scannet_samples()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py#L141)
- scene 去重：[`build_candidate_pool()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py#L173)
- official question -> Stage 1 fallback query：[`build_stage1_query_candidates()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py#L195)
- 单题执行：[`run_one_sample()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py#L279)
- 评测集成：[`main()`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py#L377)

已实跑：

1. 单题：

- [`official_batch_summary.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_single/official_batch_summary.json)

2. 小批量 3 题：

- [`official_batch_summary.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_batch3/official_batch_summary.json)

3. 单题 + 官方评测：

- [`official_batch_summary.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_eval_single_gemini/official_batch_summary.json)

结论：

- 官方 question 已经能直接驱动本地 scene 的 Stage 1 / Stage 2 / E2E。
- 目前官方问题更偏 QA 风格而非 retrieval 风格，所以 Stage 1 常见结果是 `proxy_grounded`，不是 `direct_grounded`。

---

### 2.7 官方 OpenEQA 评测代码已 clone 并复用

官方 repo 已 clone 到：

- [`external/open-eqa`](/Users/bytedance/project/3DVLMReasoning/external/open-eqa)

当前 commit：

- `cfa3fce`

官方评测主入口：

- [`evaluate-predictions.py`](/Users/bytedance/project/3DVLMReasoning/external/open-eqa/evaluate-predictions.py)

我新增了一个包装器：

- [`openeqa_official_eval.py`](/Users/bytedance/project/3DVLMReasoning/src/benchmarks/openeqa_official_eval.py#L1)

作用：

- 保留官方 `openeqa.evaluation.llm_match` 评测逻辑
- 只替换 judge 模型客户端
- 使用当前项目的 Azure-compatible chat client
- 当前评测 judge 默认使用 `gemini-2.5-pro`

关键入口：

- [`evaluate_predictions_with_official_llm_match()`](/Users/bytedance/project/3DVLMReasoning/src/benchmarks/openeqa_official_eval.py#L111)

已验证单题评测结果：

- stage2 predictions: [`official_predictions_stage2.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_eval_single_gemini/official_predictions_stage2.json)
- stage2 metrics: [`official_predictions_stage2-metrics.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_eval_single_gemini/official_predictions_stage2-metrics.json)
- e2e predictions: [`official_predictions_e2e.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_eval_single_gemini/official_predictions_e2e.json)
- e2e metrics: [`official_predictions_e2e-metrics.json`](/Users/bytedance/project/3DVLMReasoning/tmp/openeqa_official_eval_single_gemini/official_predictions_e2e-metrics.json)

单题评测分数：

- `stage2 = 50.0`
- `e2e = 75.0`

---

### 2.8 已确认 Stage 1 parser 的慢点，并验证 gemini pool 可用

调查对象：

- Stage 1 parser 为什么慢
- gemini pool 是否可用
- 慢点到底在本地还是远端

代码定位：

- QueryParser 同步入口：[`parse()`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py#L618)
- pool 重试入口：[`_parse_with_pool_retry()`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py#L674)
- 真正发请求的位置：
  - 文本：[`llm.invoke(prompt)`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py#L769)
  - 带图：[`llm.invoke([message])`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py#L767)
- 图片编码：[`_image_to_data_url()`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py#L441)
- gemini pool：[`GeminiClientPool`](/Users/bytedance/project/3DVLMReasoning/src/utils/llm_client.py#L81)

实测结论：

- gemini pool 正常可用
- 最小 pool client 调用可通
- `pool.invoke_with_retry()` 可通
- 慢点主要在远端 parser 生成，不在本地图片编码或 JSON 校验

这次测到的典型数字：

- 最小 pool client: `5.80s`
- `pool.invoke_with_retry`: `1.36s`
- `QueryParser(use_pool=True)` 文本 parse: `19.07s`
- `QueryParser(use_pool=True)` 带 1 张 BEV 图的 multimodal parse: `35.05s`

因此当前结论是：

- pool 的主要价值是更稳定、更抗限流、更抗坏 key
- 不是保证 parser 一定非常快
- parser 慢的根因仍然是“大 prompt + JSON 结构化输出 + multimodal remote inference”

---

### 2.9 已将“非 Agents 默认走 gemini pool”落到代码

这是本次最后一项比较大的代码改动。

现在的规则是：

- 非 `agents/` 路径下，默认 LLM 应该用 `gemini-2.5-pro`
- 且不需要手动传 `use_pool=True`
- 如果模型是 `gemini-2.5-pro`，则默认自动启用 pool
- `Agents` 默认配置不动，仍保留现有 Stage 2 的 GPT 默认

核心改动：

1. `get_langchain_chat_model()` 自动启 pool：

- [`llm_client.py`](/Users/bytedance/project/3DVLMReasoning/src/utils/llm_client.py#L528)

2. QueryParser 自动启 pool：

- [`query_parser.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py#L393)
- [`parsing/parser.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/parsing/parser.py#L77)

3. KeyframeSelector 自动启 pool：

- [`keyframe_selector.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/keyframe_selector.py#L231)
- [`retrieval/keyframe_selector.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/retrieval/keyframe_selector.py#L231)

4. Stage 1 默认模型切到 `gemini-2.5-pro`：

- [`batch_eval.py`](/Users/bytedance/project/3DVLMReasoning/src/evaluation/batch_eval.py#L234)
- [`ablation_config.py`](/Users/bytedance/project/3DVLMReasoning/src/evaluation/ablation_config.py#L161)
- OpenEQA 单场景 pilot：[`openeqa_single_scene_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_single_scene_pilot.py#L52)
- teacher models：[`open_world_sample_builder.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/open_world_sample_builder.py#L37)

运行时确认过：

- `QueryParser("gemini-2.5-pro", ...)` 默认 `use_pool=True`
- `QueryParser("gpt-5.2-2025-12-11", ...)` 默认 `use_pool=False`

已跑过验证：

- [`test_ablation_config.py`](/Users/bytedance/project/3DVLMReasoning/src/evaluation/tests/test_ablation_config.py)
- [`test_run_uncertainty_ablation.py`](/Users/bytedance/project/3DVLMReasoning/src/evaluation/ablations/tests/test_run_uncertainty_ablation.py)

---

## 3. 当前仓库中的关键新增 / 修改文件

### 3.1 新增文件

- [`src/agents/examples/openeqa_single_scene_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_single_scene_pilot.py)
- [`src/agents/examples/openeqa_batch_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_batch_pilot.py)
- [`src/agents/examples/openeqa_official_question_pilot.py`](/Users/bytedance/project/3DVLMReasoning/src/agents/examples/openeqa_official_question_pilot.py)
- [`src/benchmarks/openeqa_official_eval.py`](/Users/bytedance/project/3DVLMReasoning/src/benchmarks/openeqa_official_eval.py)
- [`external/open-eqa`](/Users/bytedance/project/3DVLMReasoning/external/open-eqa)

### 3.2 关键修改文件

- [`src/benchmarks/openeqa_loader.py`](/Users/bytedance/project/3DVLMReasoning/src/benchmarks/openeqa_loader.py)
- [`src/utils/llm_client.py`](/Users/bytedance/project/3DVLMReasoning/src/utils/llm_client.py)
- [`src/query_scene/query_parser.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/query_parser.py)
- [`src/query_scene/parsing/parser.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/parsing/parser.py)
- [`src/query_scene/keyframe_selector.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/keyframe_selector.py)
- [`src/query_scene/retrieval/keyframe_selector.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/retrieval/keyframe_selector.py)
- [`src/query_scene/open_world_sample_builder.py`](/Users/bytedance/project/3DVLMReasoning/src/query_scene/open_world_sample_builder.py)
- [`src/evaluation/batch_eval.py`](/Users/bytedance/project/3DVLMReasoning/src/evaluation/batch_eval.py)
- [`src/evaluation/ablation_config.py`](/Users/bytedance/project/3DVLMReasoning/src/evaluation/ablation_config.py)
- 多个 `src/evaluation/scripts/*.py`
- 多个 `src/evaluation/ablations/*.py`

---

## 4. 当前可直接使用的命令

### 4.1 单 scene pilot

```bash
cd /Users/bytedance/project/3DVLMReasoning
.venv/bin/python -m agents.examples.openeqa_single_scene_pilot \
  --mode all \
  --clip-id 124-scannet-scene0131_02 \
  --k 1 \
  --stage2-query "List the major objects around the keyboard and describe the overall workstation layout. If the current images do not show enough context, request more views before answering."
```

### 4.2 本地自动 query 的 5-scene batch

```bash
cd /Users/bytedance/project/3DVLMReasoning
.venv/bin/python -m agents.examples.openeqa_batch_pilot \
  --num-scenes 5 \
  --output-root tmp/openeqa_batch_pilot_5
```

### 4.3 官方 question 单题

```bash
cd /Users/bytedance/project/3DVLMReasoning
.venv/bin/python -m agents.examples.openeqa_official_question_pilot \
  --question-id e845f82a-d55c-42f8-88e3-8e9fcaf5bd02 \
  --output-root tmp/openeqa_official_single
```

### 4.4 官方 question 单题 + 官方评测

```bash
cd /Users/bytedance/project/3DVLMReasoning
.venv/bin/python -m agents.examples.openeqa_official_question_pilot \
  --question-id b0b40220-5e47-40ea-9298-167b140c242c \
  --evaluate \
  --llm-model gemini-2.5-pro \
  --output-root tmp/openeqa_official_eval_single_gemini
```

### 4.5 官方 question 去重小批量

```bash
cd /Users/bytedance/project/3DVLMReasoning
.venv/bin/python -m agents.examples.openeqa_official_question_pilot \
  --max-samples 3 \
  --unique-scenes \
  --require-stage1-success \
  --evaluate \
  --llm-model gemini-2.5-pro \
  --output-root tmp/openeqa_official_unique_eval3
```

说明：

- 这条 3-scene `unique-scenes + evaluate` 全量命令本次没有跑完，我中途停了，因为第一题 Stage 1 parser 延迟较高，不值得继续无信息等待。
- 但 `--unique-scenes` 的去重逻辑和 `--evaluate` 的单题评测链路都已经分别验证过。

---

## 5. 当前未完成事项

### 5.1 官方 question 的 `unique-scenes` 小批量评测没有完整跑完

现状：

- 逻辑已经实现
- 去重逻辑已验证
- 官方评测链路已验证
- 但 3-scene 串行完整结果没有最终落地

建议下一步：

- 直接重跑 `tmp/openeqa_official_unique_eval3`
- 或者先加一个“复用已有 artifact 的 resume / evaluate-only”模式，再跑

---

### 5.2 还没有做真正的 benchmark 级答案统计

现状：

- 已能生成官方格式 predictions
- 已能用官方 judge 打分
- 但现在只做了单题和很小的小批量
- 还没有产出正式的 batch metrics、category breakdown、completion uplift 报告

建议下一步：

1. 先跑 `10` 个 `unique-scenes`
2. 对 `stage2` 和 `e2e` 分别统计：
   - official score
   - completion rate
   - insufficient_evidence rate
   - tool-call rate
   - 平均 final keyframes

---

### 5.3 还没补 `resume-from-artifacts` / `evaluate-only` 模式

当前问题：

- `openeqa_official_question_pilot.py` 现在每次 `--evaluate` 都会重新跑 Stage 1 / Stage 2 / E2E
- 对于高延迟 parser，不够高效

建议改法：

- 新增 `--reuse-existing-artifacts`
- 如果 `runs/<clip>/<qid>/stage2.json`、`e2e.json` 已存在，就直接提取 prediction，跳过 pipeline，只跑 official judge

---

### 5.4 还有一批测试和文档没有同步清理

已改并验证的测试很少，主要是：

- `test_ablation_config.py`
- `test_run_uncertainty_ablation.py`

还未系统清理的残留点：

- 若干 docstring / example 字符串仍提到 `gpt-5.2-2025-12-11`
- `query_scene/tests/test_open_world_sample_builder.py` 里仍有旧 teacher model 断言
- 某些 `agents/examples/` 中仍显式写死 `gpt-5.2` 和 `use_pool=False`

注意：

- 用户本次要求是“除了 Agents，其他都必须默认走 gemini pool”，所以我没有批量改 `agents/examples/`
- 但如果后续做统一清理，这块要再扫一遍

---

### 5.5 OpenEQA 官方 question 仍然有 retrieval mismatch

这是方法层面的，不是代码 bug。

现状：

- 官方 OpenEQA question 更接近 QA，而不是 object retrieval query
- 所以 Stage 1 对官方题常常只能产出 `proxy_grounded`
- 这不是 adapter mapping 错，而是 query form 与 retrieval system 的目标不完全一致

建议：

- 如果要做更强的官方题 Stage 1，应该加一层：
  - QA question -> retrieval-oriented subquery rewrite
- 当前脚本里已有很轻量的 heuristic rewrite，但明显还不够

---

## 6. 风险点

### 6.1 当前 worktree 很脏，且包含大量未提交文件

当前 `git status --short` 显示：

- 多个 `src/evaluation/*`
- 多个 `src/query_scene/*`
- 新增 `external/`
- 新增多个 pilot 脚本
- `tmp/` 下有大量运行产物

风险：

- 下一个 agent 在提交前必须先区分：
  - 本次会话新增的代码
  - 用户原有未提交改动
  - 临时实验产物

建议：

- 先做一次有选择的 `git status` / `git diff --stat`
- 不要直接全量 commit

---

### 6.2 `external/open-eqa` 是 clone 下来的官方 repo

风险：

- 如果后续做 package、lint 或 CI，`external/` 目录可能引入噪音
- 若仓库不希望 vendor 进正式代码库，需要后续决定：
  - 保留 clone
  - 改成 submodule
  - 只保留最小评测代码副本

当前状态：

- 先按“可用优先”保留 clone

---

### 6.3 官方评测的 judge 已不是官方默认 OpenAI GPT judge

事实：

- 评测 prompt 和评分逻辑是官方的
- 但 judge 模型客户端被替换为当前项目的 Azure-compatible `gemini-2.5-pro`

风险：

- 与论文 / leaderboard 上官方默认 judge 不完全等价
- 分数具有可参考性，但不应在对外叙述里直接说“与官方线上得分完全一致”

---

### 6.4 Stage 1 parser 延迟仍然高

虽然已确认 pool 可用，但风险仍在：

- 单条 parser 可能是 `20s-35s`
- 带图 parse 更慢
- 服务端波动明显，单次差异很大

后果：

- 小批量实验会被 parser 吞掉大量时间
- 如果没有 artifact reuse，成本很高

---

### 6.5 非 agent 默认值已切成 gemini，但未全量回归

已做：

- 关键文件 `py_compile`
- 2 组 pytest

未做：

- 没有对所有 `evaluation/scripts` 和 `query_scene` examples 做全量集成回归
- 没有全量扫完所有默认值相关测试

建议：

- 若下一个 agent 要继续合并这批改动，至少再跑：
  - `pytest src/query_scene/tests -q`
  - `pytest src/evaluation/tests -q`
  - 按需抽 1-2 个 evaluation script 做 smoke test

---

## 7. 给下一个 code agent 的推荐起手动作

优先顺序建议如下：

1. 先读本文件，再看：
   - [`current_repo_state.md`](/Users/bytedance/project/3DVLMReasoning/docs/current_repo_state.md)
   - [`stage2_agent_handoff.md`](/Users/bytedance/project/3DVLMReasoning/docs/stage2_agent_handoff.md)

2. 先确认当前改动和 worktree：
   - `git status --short`
   - `git diff --stat`

3. 如果目标是继续 OpenEQA 官方评测：
   - 先跑 `--unique-scenes --evaluate` 的 3-scene 或 10-scene
   - 然后补 `resume-from-artifacts`

4. 如果目标是继续工程收口：
   - 先扫所有非 `agents/` 的剩余默认值 / 文档 / 测试残留
   - 再决定要不要把 `agents/examples/` 也统一掉

5. 如果目标是继续科研实验：
   - 先做 `10` scene official question batch
   - 再统计 `stage2` vs `e2e` 的 judge score uplift

---

## 8. 本次会话的最重要结论

一句话总结：

> 本地 OpenEQA raw 数据已经补齐，单 scene 和小批量 Stage 1 / Stage 2 / E2E 已跑通，官方 `open-eqa-v0.json` 与本地 scene 的映射已经接上，官方评测代码也已复用并能跑单题 judge；同时，非 Agents 的默认 LLM 路径已经收口到 `gemini-2.5-pro + gemini pool`，但批量 official eval、artifact reuse 和全量回归仍未完成。

