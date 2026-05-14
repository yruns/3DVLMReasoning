# VG 初始纯净图与按需标注图设计

日期：2026-05-14
分支：`feat/nr3d-v4-agent-guards-fair-views`
范围：NR3D 与 ScanRefer 的 pack-v1 visual grounding evidence flow

## 问题

当前 pack-v1 VG 样本会把带标注的 keyframe 作为初始视觉证据注入给
Stage-2 Agent。样本准备阶段会在 `annotated/frame_<frame_id>.png` 存在时，
把选中的 keyframe 归一化到这个 annotated 图，所以 Agent 第一轮看到的图已经
包含彩色 bbox 和 `#id label`。

这会让第一轮视觉观察变得很吵。wall、floor、desk、person 这类大框会在
Agent 还没有判断哪一帧值得细看之前，就显著干扰整体场景理解。

## 目标行为

初始 evidence 应该由“纯净场景图 + 独立文本候选清单”组成。Agent 只有在认为
某个 frame 有价值并调用 `view_keyframe_marked(frame_id=...)` 后，才会看到带
bbox 和 label 的 marked 图。

初始每帧文本必须包含：

- `frame_id`
- 按 left-to-right 排列的 proposal
- 每个 proposal 用 `#<proposal_id> <category>` 表示

初始每帧文本必须不包含：

- `bbox_2d`
- visibility score
- annotated image path

marked-frame 工具可以继续在 Agent 显式请求 marked 图之后返回更详细的
marked-frame geometry，用于 guard 和 final 决策校验。

## 推荐方案

采用双通道 evidence policy：

1. 初始图保持纯净
   - pack sample JSON 里的 `keyframes[*].image_path` 指向原始 RGB frame，
     不再指向 `annotated/frame_<frame_id>.png`。
   - annotated frames 仍然正常渲染，并保留在现有 annotated 目录中供工具使用。

2. 初始候选清单走文本
   - 在 Stage-2 初始 prompt 中增加 VG 专用的 initial frame summary。
   - 候选列表从 bundle metadata 里已有的 `vg_proposal_pool.frame_index`
     和 proposal category 派生。
   - 如果 proposal 在该 frame 上有 2D 几何信息，则按 2D center x 排序；
     如果缺少 frame-view geometry，则只为文本排序保留 frame-index 原始顺序。

3. marked 图按需注入
   - `view_keyframe_marked(frame_id)` 仍然是显式入口，负责把
     `annotated/frame_<frame_id>.png` append 到 `vg_pending_images`。
   - 只有工具调用之后，下一轮 Agent 消息才会收到 marked 图。

## 数据流

样本准备阶段：

1. 初始 frame id 的选择逻辑保持不变。
2. 写入 sample `keyframes` 时使用原始 RGB image path。
3. annotated frames 仍然照常渲染。
4. `vg_proposal_pool.annotated_image_dir` 和 `vg_proposal_pool.frame_index`
   保持不变，确保 VG tools 继续可用。

runtime prompt 组装阶段：

1. 编码并注入纯净的 `keyframes[*].image_path` 图像。
2. 对 VG task 增加一个 `Initial frame proposal inventory` 文本 section。
3. 这个 section 每个 keyframe 一行，例如：
   `frame_id=69 left_to_right: #12 monitor, #31 keyboard, #4 chair`
4. 这个初始 section 不包含 box、visibility score 或 annotated image path。

工具使用阶段：

1. `list_keyframes_with_proposals()` 可以返回同样的纯文本候选清单，不返回
   annotated image path。
2. `view_keyframe_marked(frame_id)` 仍然是 annotated image 第一次被注入的入口。
3. 当 rationale 引用 marked visual evidence 时，final-answer guards 继续依赖
   marked-frame tool trace 进行校验。

## 备选方案

### 只改 `keyframes[*].image_path`

这样可以让初始图变纯净，但 Agent 会失去快速的 `#id category` 候选清单，除非
它马上调用工具。这会浪费一轮推理，也会让初始 prompt 信息不足。

### 只改 prompt 文本

这样可以给 Agent 更好的候选 inventory，但初始图仍然包含噪声标注，无法解决
视觉污染问题。

### 同时改 pack 图像路径和 VG 文本清单

这最符合目标推理流程：先用纯净图理解场景，通过结构化文本知道候选，再在需要时
请求 marked evidence。推荐采用这个方案。

## 测试

目标测试需要覆盖：

- NR3D prepared sample 的 keyframes 保留原始 RGB path。
- ScanRefer prepared sample 的 keyframes 保留原始 RGB path。
- annotated frames 仍然被渲染，并且能通过
  `vg_proposal_pool.annotated_image_dir` 找到。
- 初始 VG prompt 包含 left-to-right 的 `#id category` inventory。
- 初始 VG prompt 不包含 `bbox_2d`、visibility score 或 annotated image path。
- `view_keyframe_marked(frame_id)` 仍然会通过 `vg_pending_images` queue marked 图。

代码改完后，先跑 focused unit tests，再跑固定 NR3D random100 对比；在这些验证
完成前，不基于这个改动宣称 benchmark 指标提升。

## 注意事项

这个设计只改变 evidence 呈现方式，不改变 proposal generation、visibility
计算或 Stage-1 frame selection。任何指标变化都应该记录为同一 fold、同一
proposal pool 下的 Stage-2 evidence-format ablation。
