# ScanNet Data Exporter

Utilities for extracting data from ScanNet .sens files.

Developed and tested with Python 3.8+.

## Usage

```bash
python -m src.scripts.scannet_process.reader \
    --filename [.sens file to export data from] \
    --output_path [output directory to export data to] \
    --export_depth_images \
    --export_color_images \
    --export_poses \
    --export_intrinsics
```

### Options

- `--export_depth_images`: Export all depth frames as 16-bit PNGs (depth shift 1000)
- `--export_color_images`: Export all color frames as 8-bit RGB JPEGs
- `--export_poses`: Export all camera poses (4x4 matrix, camera to world)
- `--export_intrinsics`: Export camera intrinsics (4x4 matrix)

## Example

```bash
SCENE_ID=scene0011_00
SENS_PATH=$HOME/data/scannet/scans/${SCENE_ID}/${SCENE_ID}.sens
OUTPUT_PATH=$HOME/data/scannet/scans/${SCENE_ID}/

python -m src.scripts.scannet_process.reader \
    --filename $SENS_PATH \
    --output_path $OUTPUT_PATH \
    --export_depth_images \
    --export_color_images \
    --export_poses \
    --export_intrinsics
```

## Source

Based on the official ScanNet SDK:
https://github.com/ScanNet/ScanNet/tree/master/SensReader/python
