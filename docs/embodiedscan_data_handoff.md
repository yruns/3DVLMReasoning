# EmbodiedScan 数据准备交接文档

**日期**: 2026-04-04
**分支**: `feat/embodiedscan-grounding`
**任务**: 为 EmbodiedScan 3D Visual Grounding 任务准备数据

---

## 已完成

### 1. 标注文件已下载 (`data/embodiedscan/`)

从 HuggingFace (`cjfcsjt/embodiedscan`) 下载，全部就绪：

| 文件 | 大小 | 说明 |
|------|------|------|
| `embodiedscan_infos_train.pkl` | 306M | 训练集场景元数据（相机位姿、3D bbox、288类别） |
| `embodiedscan_infos_val.pkl` | 81M | 验证集场景元数据 |
| `embodiedscan_infos_test.pkl` | 55M | 测试集场景元数据 |
| `embodiedscan_val_vg.json` | 14M | **验证集 VG 标注（有GT，本地评测用）** |
| `embodiedscan_val_mini_vg.json` | 2.9M | 验证集 VG 迷你子集 |
| `embodiedscan_val_vg_all.json` | 15M | 验证集 VG（含复杂描述） |
| `embodiedscan_test_vg.json` | 11M | 测试集 VG（无GT，需提交服务器） |
| `embodiedscan_train_vg.json` | 56M | 训练集 VG（不需要） |
| `embodiedscan_train_mini_vg.json` | 12M | 训练集 VG 迷你子集（不需要） |
| `embodiedscan_train_vg_all.json` | 61M | 训练集 VG（不需要） |

### 2. EmbodiedScan 官方仓库已 clone

位置: `data/EmbodiedScan_repo/`，包含图像提取脚本。

### 3. 分支已创建

`feat/embodiedscan-grounding`，从 master 分出。

---

## 待完成：下载并提取 Val 集的 RGB-D 图像

我们只做评测（VLM agent，不训练），所以只需要 **val 集**的图像数据。

### Val 集需求

| 数据源 | 场景数 | VG 条目 | 图像数 | 预估存储 |
|--------|--------|---------|--------|---------|
| **ScanNet** | 222 | 25,440 | 34,703 | ~10 GB |
| 3RScan | 214 | 19,891 | 56,924 | ~17 GB |
| Matterport3D | 258 | 11,723 | 49,456 | ~14 GB |
| **总计** | **694** | **57,054** | **141,083** | **~41 GB** |

**建议优先下载 ScanNet**（最大子集，最熟悉的数据源）。

### ScanNet 图像准备步骤

#### Step 1: 下载 `.sens` 文件

需要 ScanNet 下载权限。获取 `SCANNET_BASE_URL` 后：

```bash
export SCANNET_BASE_URL="https://kaldir.vc.in.tum.de/scannet/v2/scans"
```

需要下载的 222 个场景 ID 可通过以下命令获取：

```python
import pickle
with open('data/embodiedscan/embodiedscan_infos_val.pkl', 'rb') as f:
    val_info = pickle.load(f)
scannet_scenes = sorted(set(
    d['sample_idx'].replace('scannet/', '')
    for d in val_info['data_list']
    if d['sample_idx'].startswith('scannet/')
))
# 输出: ['scene0006_00', 'scene0006_01', ..., 共222个]
```

下载命令（每个场景约 200MB-2GB）：

```bash
for scene_id in $(python3 -c "
import pickle
with open('data/embodiedscan/embodiedscan_infos_val.pkl','rb') as f:
    info = pickle.load(f)
for d in info['data_list']:
    if d['sample_idx'].startswith('scannet/'):
        print(d['sample_idx'].replace('scannet/',''))
" | sort -u); do
    mkdir -p data/embodiedscan/scannet/scans/$scene_id
    wget -nc -P data/embodiedscan/scannet/scans/$scene_id \
        "${SCANNET_BASE_URL}/${scene_id}/${scene_id}.sens"
done
```

#### Step 2: 提取 posed images

使用 EmbodiedScan 的转换脚本（`--fast` 只提取每10帧）：

```bash
cd /path/to/3DVLMReasoning
python data/EmbodiedScan_repo/embodiedscan/converter/generate_image_scannet.py \
    --dataset_folder data/embodiedscan/scannet/ \
    --fast \
    --nproc 8
```

这会在 `data/embodiedscan/scannet/posed_images/<scene_id>/` 下生成：
- `NNNNN.jpg` — RGB 图像
- `NNNNN.png` — 深度图
- `NNNNN.txt` — 相机位姿 (4x4 cam2world)
- `intrinsic.txt` — RGB 相机内参
- `depth_intrinsic.txt` — 深度相机内参

#### Step 3: 验证

```python
import os
posed = 'data/embodiedscan/scannet/posed_images'
scenes = [d for d in os.listdir(posed) if d.startswith('scene')]
print(f'已提取场景: {len(scenes)} / 222')
# 检查单个场景
files = os.listdir(f'{posed}/{scenes[0]}')
jpgs = [f for f in files if f.endswith('.jpg')]
print(f'{scenes[0]}: {len(jpgs)} 张图像')
```

### 3RScan 和 Matterport3D（可选，后续）

如果需要完整 val 集评测，还需要：

- **3RScan**: 从 https://github.com/WaldJohannaU/3RScan 下载，然后运行:
  ```bash
  python data/EmbodiedScan_repo/embodiedscan/converter/generate_image_3rscan.py \
      --dataset_folder data/embodiedscan/3rscan/
  ```

- **Matterport3D**: 从 https://github.com/niessner/Matterport 下载（无需额外提取）

---

## 最终目标目录结构

```
data/embodiedscan/
├── embodiedscan_infos_val.pkl          ✅ 已有
├── embodiedscan_infos_test.pkl         ✅ 已有
├── embodiedscan_val_vg.json            ✅ 已有
├── embodiedscan_test_vg.json           ✅ 已有
├── embodiedscan_val_mini_vg.json       ✅ 已有
├── ...其他标注文件                       ✅ 已有
├── scannet/
│   ├── scans/<scene_id>/<scene_id>.sens    ❌ 需下载
│   └── posed_images/<scene_id>/            ❌ 需从.sens提取
│       ├── NNNNN.jpg
│       ├── NNNNN.png
│       ├── NNNNN.txt
│       ├── intrinsic.txt
│       └── depth_intrinsic.txt
├── 3rscan/  (可选)                          ❌
└── matterport3d/  (可选)                    ❌
```

## PKL 数据格式参考

每个场景在 PKL 中的结构：

```python
{
    "sample_idx": "scannet/scene0415_00",
    "cam2img": [[fx, 0, cx, 0], [0, fy, cy, 0], ...],  # 4x4 内参
    "axis_align_matrix": <4x4>,  # 轴对齐矩阵
    "images": [
        {
            "img_path": "scannet/posed_images/scene0415_00/00000.jpg",
            "depth_path": "scannet/posed_images/scene0415_00/00000.png",
            "cam2global": <4x4>,  # 相机外参
            "visible_instance_ids": [1, 2, 5, ...]
        },
        ...  # 平均每场景 143 个视图
    ],
    "instances": [
        {
            "bbox_3d": [cx, cy, cz, dx, dy, dz, alpha, beta, gamma],  # 9-DOF
            "bbox_label_3d": 42,  # 类别ID（288类）
            "bbox_id": 5  # 实例ID，与 VG JSON 的 target_id 对应
        },
        ...
    ]
}
```

VG JSON 格式：

```json
{
    "scan_id": "scannet/scene0415_00",
    "target_id": 2,          // 对应 PKL 中 bbox_id
    "distractor_ids": [1],
    "text": "find the bag that is closer to the bathtub",
    "target": "bag",
    "anchors": ["bathtub"],
    "anchor_ids": [9],
    "tokens_positive": [[9, 12]]
}
```

## 注意事项

1. **PKL 中的 img_path 是相对路径**，相对于 `data/embodiedscan/` 目录（即 `data_root`）
2. **generate_image_scannet.py 依赖**: `imageio`, `numpy`, `mmengine`（需要在 conda 环境中安装）
3. **GPU 1 坏了** — 不要用 `CUDA_VISIBLE_DEVICES=1`
4. 提取脚本会 `os.chdir` 到 dataset_folder，输出到 `posed_images/` 子目录
