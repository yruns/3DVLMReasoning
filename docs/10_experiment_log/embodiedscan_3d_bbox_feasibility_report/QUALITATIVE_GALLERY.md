# Qualitative Gallery: RGB, Point Clouds, GT and Best Predictions

This gallery visualizes every target in the V-DETR batch30 run. Each case
contains one RGB observation panel and four point-cloud panels. Green boxes
are GT, red boxes are the best V-DETR proposal for that input condition,
and blue boxes are the best 2D-CG proposal over the same scene.

![Qualitative contact sheet](resources/qualitative/qualitative_contact_sheet.png)

## Case Index

| Case | Scene | Target | Category | 2D-CG | Single | Multi | Pose crop | Full scene | Figure |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Q01 | scene0451_00 | 72 | picture | 0.742 | 0.000 | 0.000 | 0.000 | 0.045 | [case_01_picture_scene0451_00_target72.png](resources/qualitative/case_01_picture_scene0451_00_target72.png) |
| Q02 | scene0451_00 | 32 | curtain | 0.660 | 0.028 | 0.087 | 0.000 | 0.490 | [case_02_curtain_scene0451_00_target32.png](resources/qualitative/case_02_curtain_scene0451_00_target32.png) |
| Q03 | scene0451_00 | 26 | curtain | 0.199 | 0.323 | 0.266 | 0.000 | 0.600 | [case_03_curtain_scene0451_00_target26.png](resources/qualitative/case_03_curtain_scene0451_00_target26.png) |
| Q04 | scene0451_02 | 23 | curtain | 0.054 | 0.000 | 0.000 | 0.000 | 0.263 | [case_04_curtain_scene0451_02_target23.png](resources/qualitative/case_04_curtain_scene0451_02_target23.png) |
| Q05 | scene0451_02 | 17 | table | 0.370 | 0.000 | 0.000 | 0.035 | 0.513 | [case_05_table_scene0451_02_target17.png](resources/qualitative/case_05_table_scene0451_02_target17.png) |
| Q06 | scene0451_02 | 4 | curtain | 0.519 | 0.000 | 0.000 | 0.367 | 0.484 | [case_06_curtain_scene0451_02_target4.png](resources/qualitative/case_06_curtain_scene0451_02_target4.png) |
| Q07 | scene0451_04 | 14 | curtain | 0.519 | 0.392 | 0.097 | 0.000 | 0.270 | [case_07_curtain_scene0451_04_target14.png](resources/qualitative/case_07_curtain_scene0451_04_target14.png) |
| Q08 | scene0451_04 | 22 | curtain | 0.294 | 0.001 | 0.026 | 0.000 | 0.436 | [case_08_curtain_scene0451_04_target22.png](resources/qualitative/case_08_curtain_scene0451_04_target22.png) |
| Q09 | scene0451_04 | 88 | cabinet | 0.459 | 0.000 | 0.000 | 0.000 | 0.496 | [case_09_cabinet_scene0451_04_target88.png](resources/qualitative/case_09_cabinet_scene0451_04_target88.png) |
| Q10 | scene0392_01 | 33 | window | 0.304 | 0.449 | 0.014 | 0.000 | 0.665 | [case_10_window_scene0392_01_target33.png](resources/qualitative/case_10_window_scene0392_01_target33.png) |
| Q11 | scene0392_01 | 42 | sink | 0.433 | 0.062 | 0.104 | 0.000 | 0.566 | [case_11_sink_scene0392_01_target42.png](resources/qualitative/case_11_sink_scene0392_01_target42.png) |
| Q12 | scene0392_01 | 20 | sink | 0.622 | 0.000 | 0.000 | 0.744 | 0.577 | [case_12_sink_scene0392_01_target20.png](resources/qualitative/case_12_sink_scene0392_01_target20.png) |
| Q13 | scene0114_00 | 6 | chair | 0.339 | 0.000 | 0.106 | 0.678 | 0.591 | [case_13_chair_scene0114_00_target6.png](resources/qualitative/case_13_chair_scene0114_00_target6.png) |
| Q14 | scene0114_00 | 5 | chair | 0.433 | 0.526 | 0.515 | 0.000 | 0.697 | [case_14_chair_scene0114_00_target5.png](resources/qualitative/case_14_chair_scene0114_00_target5.png) |
| Q15 | scene0114_00 | 9 | cabinet | 0.591 | 0.000 | 0.000 | 0.740 | 0.854 | [case_15_cabinet_scene0114_00_target9.png](resources/qualitative/case_15_cabinet_scene0114_00_target9.png) |
| Q16 | scene0114_02 | 1 | window | 0.178 | 0.000 | 0.000 | 0.000 | 0.351 | [case_16_window_scene0114_02_target1.png](resources/qualitative/case_16_window_scene0114_02_target1.png) |
| Q17 | scene0114_02 | 11 | window | 0.004 | 0.000 | 0.033 | 0.000 | 0.501 | [case_17_window_scene0114_02_target11.png](resources/qualitative/case_17_window_scene0114_02_target11.png) |
| Q18 | scene0114_02 | 19 | desk | 0.326 | 0.723 | 0.689 | 0.323 | 0.780 | [case_18_desk_scene0114_02_target19.png](resources/qualitative/case_18_desk_scene0114_02_target19.png) |
| Q19 | scene0614_00 | 33 | window | 0.071 | 0.255 | 0.528 | 0.000 | 0.754 | [case_19_window_scene0614_00_target33.png](resources/qualitative/case_19_window_scene0614_00_target33.png) |
| Q20 | scene0614_00 | 13 | desk | 0.461 | 0.000 | 0.000 | 0.000 | 0.896 | [case_20_desk_scene0614_00_target13.png](resources/qualitative/case_20_desk_scene0614_00_target13.png) |
| Q21 | scene0614_00 | 12 | cabinet | 0.362 | 0.000 | 0.000 | 0.865 | 0.272 | [case_21_cabinet_scene0614_00_target12.png](resources/qualitative/case_21_cabinet_scene0614_00_target12.png) |
| Q22 | scene0247_00 | 15 | door | 0.368 | 0.000 | 0.000 | 0.000 | 0.533 | [case_22_door_scene0247_00_target15.png](resources/qualitative/case_22_door_scene0247_00_target15.png) |
| Q23 | scene0247_00 | 40 | door | 0.513 | 0.000 | 0.002 | 0.012 | 0.595 | [case_23_door_scene0247_00_target40.png](resources/qualitative/case_23_door_scene0247_00_target40.png) |
| Q24 | scene0247_00 | 54 | table | 0.401 | 0.000 | 0.000 | 0.000 | 0.748 | [case_24_table_scene0247_00_target54.png](resources/qualitative/case_24_table_scene0247_00_target54.png) |
| Q25 | scene0040_00 | 8 | desk | 0.485 | 0.148 | 0.336 | 0.195 | 0.598 | [case_25_desk_scene0040_00_target8.png](resources/qualitative/case_25_desk_scene0040_00_target8.png) |
| Q26 | scene0040_00 | 9 | desk | 0.520 | 0.000 | 0.000 | 0.569 | 0.625 | [case_26_desk_scene0040_00_target9.png](resources/qualitative/case_26_desk_scene0040_00_target9.png) |
| Q27 | scene0040_00 | 45 | chair | 0.294 | 0.329 | 0.416 | 0.000 | 0.769 | [case_27_chair_scene0040_00_target45.png](resources/qualitative/case_27_chair_scene0040_00_target45.png) |
| Q28 | scene0040_01 | 14 | desk | 0.533 | 0.495 | 0.719 | 0.000 | 0.824 | [case_28_desk_scene0040_01_target14.png](resources/qualitative/case_28_desk_scene0040_01_target14.png) |
| Q29 | scene0040_01 | 12 | desk | 0.828 | 0.000 | 0.000 | 0.864 | 0.798 | [case_29_desk_scene0040_01_target12.png](resources/qualitative/case_29_desk_scene0040_01_target12.png) |
| Q30 | scene0040_01 | 40 | cabinet | 0.365 | 0.000 | 0.000 | 0.000 | 0.139 | [case_30_cabinet_scene0040_01_target40.png](resources/qualitative/case_30_cabinet_scene0040_01_target40.png) |

## Case Figures

### Q01: scene0451_00 target 72 (picture)

- RGB frame: `12`
- IoU summary: 2D-CG `0.742`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.045`

![Q01 qualitative visualization](resources/qualitative/case_01_picture_scene0451_00_target72.png)

### Q02: scene0451_00 target 32 (curtain)

- RGB frame: `153`
- IoU summary: 2D-CG `0.660`, single `0.028`, multi `0.087`, pose crop `0.000`, full scene `0.490`

![Q02 qualitative visualization](resources/qualitative/case_02_curtain_scene0451_00_target32.png)

### Q03: scene0451_00 target 26 (curtain)

- RGB frame: `198`
- IoU summary: 2D-CG `0.199`, single `0.323`, multi `0.266`, pose crop `0.000`, full scene `0.600`

![Q03 qualitative visualization](resources/qualitative/case_03_curtain_scene0451_00_target26.png)

### Q04: scene0451_02 target 23 (curtain)

- RGB frame: `29`
- IoU summary: 2D-CG `0.054`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.263`

![Q04 qualitative visualization](resources/qualitative/case_04_curtain_scene0451_02_target23.png)

### Q05: scene0451_02 target 17 (table)

- RGB frame: `215`
- IoU summary: 2D-CG `0.370`, single `0.000`, multi `0.000`, pose crop `0.035`, full scene `0.513`

![Q05 qualitative visualization](resources/qualitative/case_05_table_scene0451_02_target17.png)

### Q06: scene0451_02 target 4 (curtain)

- RGB frame: `46`
- IoU summary: 2D-CG `0.519`, single `0.000`, multi `0.000`, pose crop `0.367`, full scene `0.484`

![Q06 qualitative visualization](resources/qualitative/case_06_curtain_scene0451_02_target4.png)

### Q07: scene0451_04 target 14 (curtain)

- RGB frame: `13`
- IoU summary: 2D-CG `0.519`, single `0.392`, multi `0.097`, pose crop `0.000`, full scene `0.270`

![Q07 qualitative visualization](resources/qualitative/case_07_curtain_scene0451_04_target14.png)

### Q08: scene0451_04 target 22 (curtain)

- RGB frame: `115`
- IoU summary: 2D-CG `0.294`, single `0.001`, multi `0.026`, pose crop `0.000`, full scene `0.436`

![Q08 qualitative visualization](resources/qualitative/case_08_curtain_scene0451_04_target22.png)

### Q09: scene0451_04 target 88 (cabinet)

- RGB frame: `95`
- IoU summary: 2D-CG `0.459`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.496`

![Q09 qualitative visualization](resources/qualitative/case_09_cabinet_scene0451_04_target88.png)

### Q10: scene0392_01 target 33 (window)

- RGB frame: `23`
- IoU summary: 2D-CG `0.304`, single `0.449`, multi `0.014`, pose crop `0.000`, full scene `0.665`

![Q10 qualitative visualization](resources/qualitative/case_10_window_scene0392_01_target33.png)

### Q11: scene0392_01 target 42 (sink)

- RGB frame: `237`
- IoU summary: 2D-CG `0.433`, single `0.062`, multi `0.104`, pose crop `0.000`, full scene `0.566`

![Q11 qualitative visualization](resources/qualitative/case_11_sink_scene0392_01_target42.png)

### Q12: scene0392_01 target 20 (sink)

- RGB frame: `162`
- IoU summary: 2D-CG `0.622`, single `0.000`, multi `0.000`, pose crop `0.744`, full scene `0.577`

![Q12 qualitative visualization](resources/qualitative/case_12_sink_scene0392_01_target20.png)

### Q13: scene0114_00 target 6 (chair)

- RGB frame: `52`
- IoU summary: 2D-CG `0.339`, single `0.000`, multi `0.106`, pose crop `0.678`, full scene `0.591`

![Q13 qualitative visualization](resources/qualitative/case_13_chair_scene0114_00_target6.png)

### Q14: scene0114_00 target 5 (chair)

- RGB frame: `1`
- IoU summary: 2D-CG `0.433`, single `0.526`, multi `0.515`, pose crop `0.000`, full scene `0.697`

![Q14 qualitative visualization](resources/qualitative/case_14_chair_scene0114_00_target5.png)

### Q15: scene0114_00 target 9 (cabinet)

- RGB frame: `73`
- IoU summary: 2D-CG `0.591`, single `0.000`, multi `0.000`, pose crop `0.740`, full scene `0.854`

![Q15 qualitative visualization](resources/qualitative/case_15_cabinet_scene0114_00_target9.png)

### Q16: scene0114_02 target 1 (window)

- RGB frame: `135`
- IoU summary: 2D-CG `0.178`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.351`

![Q16 qualitative visualization](resources/qualitative/case_16_window_scene0114_02_target1.png)

### Q17: scene0114_02 target 11 (window)

- RGB frame: `99`
- IoU summary: 2D-CG `0.004`, single `0.000`, multi `0.033`, pose crop `0.000`, full scene `0.501`

![Q17 qualitative visualization](resources/qualitative/case_17_window_scene0114_02_target11.png)

### Q18: scene0114_02 target 19 (desk)

- RGB frame: `6`
- IoU summary: 2D-CG `0.326`, single `0.723`, multi `0.689`, pose crop `0.323`, full scene `0.780`

![Q18 qualitative visualization](resources/qualitative/case_18_desk_scene0114_02_target19.png)

### Q19: scene0614_00 target 33 (window)

- RGB frame: `4`
- IoU summary: 2D-CG `0.071`, single `0.255`, multi `0.528`, pose crop `0.000`, full scene `0.754`

![Q19 qualitative visualization](resources/qualitative/case_19_window_scene0614_00_target33.png)

### Q20: scene0614_00 target 13 (desk)

- RGB frame: `28`
- IoU summary: 2D-CG `0.461`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.896`

![Q20 qualitative visualization](resources/qualitative/case_20_desk_scene0614_00_target13.png)

### Q21: scene0614_00 target 12 (cabinet)

- RGB frame: `73`
- IoU summary: 2D-CG `0.362`, single `0.000`, multi `0.000`, pose crop `0.865`, full scene `0.272`

![Q21 qualitative visualization](resources/qualitative/case_21_cabinet_scene0614_00_target12.png)

### Q22: scene0247_00 target 15 (door)

- RGB frame: `416`
- IoU summary: 2D-CG `0.368`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.533`

![Q22 qualitative visualization](resources/qualitative/case_22_door_scene0247_00_target15.png)

### Q23: scene0247_00 target 40 (door)

- RGB frame: `193`
- IoU summary: 2D-CG `0.513`, single `0.000`, multi `0.002`, pose crop `0.012`, full scene `0.595`

![Q23 qualitative visualization](resources/qualitative/case_23_door_scene0247_00_target40.png)

### Q24: scene0247_00 target 54 (table)

- RGB frame: `233`
- IoU summary: 2D-CG `0.401`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.748`

![Q24 qualitative visualization](resources/qualitative/case_24_table_scene0247_00_target54.png)

### Q25: scene0040_00 target 8 (desk)

- RGB frame: `30`
- IoU summary: 2D-CG `0.485`, single `0.148`, multi `0.336`, pose crop `0.195`, full scene `0.598`

![Q25 qualitative visualization](resources/qualitative/case_25_desk_scene0040_00_target8.png)

### Q26: scene0040_00 target 9 (desk)

- RGB frame: `46`
- IoU summary: 2D-CG `0.520`, single `0.000`, multi `0.000`, pose crop `0.569`, full scene `0.625`

![Q26 qualitative visualization](resources/qualitative/case_26_desk_scene0040_00_target9.png)

### Q27: scene0040_00 target 45 (chair)

- RGB frame: `174`
- IoU summary: 2D-CG `0.294`, single `0.329`, multi `0.416`, pose crop `0.000`, full scene `0.769`

![Q27 qualitative visualization](resources/qualitative/case_27_chair_scene0040_00_target45.png)

### Q28: scene0040_01 target 14 (desk)

- RGB frame: `186`
- IoU summary: 2D-CG `0.533`, single `0.495`, multi `0.719`, pose crop `0.000`, full scene `0.824`

![Q28 qualitative visualization](resources/qualitative/case_28_desk_scene0040_01_target14.png)

### Q29: scene0040_01 target 12 (desk)

- RGB frame: `55`
- IoU summary: 2D-CG `0.828`, single `0.000`, multi `0.000`, pose crop `0.864`, full scene `0.798`

![Q29 qualitative visualization](resources/qualitative/case_29_desk_scene0040_01_target12.png)

### Q30: scene0040_01 target 40 (cabinet)

- RGB frame: `81`
- IoU summary: 2D-CG `0.365`, single `0.000`, multi `0.000`, pose crop `0.000`, full scene `0.139`

![Q30 qualitative visualization](resources/qualitative/case_30_cabinet_scene0040_01_target40.png)

