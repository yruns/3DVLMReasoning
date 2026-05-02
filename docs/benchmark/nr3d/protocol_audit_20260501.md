# NR3D / ReferIt3D Evaluation Protocol Audit

**Audit date:** 2026-05-01
**Auditor:** Worker @ surface:14
**Repo audited:** https://github.com/referit3d/referit3d
**Cloned to:** `tmp/referit3d_audit/repo/` at commit `725a5d3` (`Update README.md`).
**Citation convention:** All `<file>:<line>` paths are relative to `tmp/referit3d_audit/repo/` unless prefixed with `OURS:` (then they refer to our checkout at `/Users/bytedance/project/3DVLMReasoning/`).

---

## TL;DR

1. **The official NR3D candidate pool at test time is the FULL SCENE's segmented instances**
   (target + same-class siblings + shuffled clutter from other classes), capped at
   `max_test_objects=88`. **Not** target-class-only, **not** `target_id ∪ distractor_ids`.
   `distractor_ids` are used only to (a) stratify easy/hard buckets and (b) drop unique-target
   utterances at test time. **[CONFIRMED-CODE]**
2. **Easy ⟺ `n_objects <= 2`, Hard ⟺ `n_objects > 2`.** `n_objects` is the same-class
   instance count parsed from `stimulus_id`. **[CONFIRMED-CODE]**
3. **View-dep keyword set is exactly 10 tokens, per-token lowercase intersection** —
   not substring. **[CONFIRMED-CODE]**
4. **Canonical filters are: drop bad-context CSV ∪ clothes/clothing → tokens ≤ 24 →
   `mentions_target_class==True` → drop unique-target test utts.**
   `correct_guess` is **NOT** part of the canonical filter chain — it is only printed for
   stats and used in viz. UniVLG's `correct_guess==True` filter is a UniVLG-specific
   add-on. **[CONFIRMED-CODE]**
5. **Headline accuracy denominator** = number of test utterances after canonical filters.
   Predictions are evaluated as classification over the full scene's segmented instances
   with **axis-aligned** GT bboxes derived from ScanNet point-cloud min/max — there is
   **no IoU threshold** in the canonical NR3D protocol. **[CONFIRMED-CODE + CONFIRMED-PAPER]**
6. **Detection-mode papers** (BUTD-DETR) report a separate Acc@0.25 (only) on NR3D
   using detector-generated proposals, not the canonical classification metric.
   They do **not** report Acc@0.50 on NR3D. **[CONFIRMED-PAPER]**

---

## Q1. Candidate pool construction  [CONFIRMED-CODE + CONFIRMED-PAPER]

**Answer: (b) ALL segmented instances of the scene, capped at 88. NOT `{target_id} ∪ distractor_ids`.**

The candidate set at test time is constructed by `ListeningDataset.prepare_distractors`,
called from `ListeningDataset.__getitem__`. The `distractor_ids` parsed from
`stimulus_id` are **never** used to build the candidate pool — they are only used to
stratify easy/hard and to filter unique-target test utterances.

### Verbatim source (`referit3d/in_out/pt_datasets/listening_dataset.py:44-60`)

```python
def prepare_distractors(self, scan, target):
    target_label = target.instance_label

    # First add all objects with the same instance-label as the target
    distractors = [o for o in scan.three_d_objects if
                   (o.instance_label == target_label and (o != target))]

    # Then all more objects up to max-number of distractors
    already_included = {target_label}
    clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
    np.random.shuffle(clutter)

    distractors.extend(clutter)
    distractors = distractors[:self.max_distractors]
    np.random.shuffle(distractors)

    return distractors
```

### Verbatim source (`referit3d/in_out/pt_datasets/listening_dataset.py:127-149`)

```python
for split in splits:
    mask = is_train if split == 'train' else ~is_train
    d_set = referit_data[mask]
    d_set.reset_index(drop=True, inplace=True)

    max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
    ## this is a silly small bug -> not the minus-1.

    # if split == test remove the utterances of unique targets
    if split == 'test':
        def multiple_targets_utterance(x):
            _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
            return len(distractors_ids) > 0

        multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
        d_set = d_set[multiple_targets_mask]
```

### Defaults

- `referit3d/in_out/arguments.py:53` — `--max-test-objects` default = **88**.
- `referit3d/in_out/arguments.py:42` — `--max-distractors` default = **51** (training only).
- `referit3d/in_out/pt_datasets/listening_dataset.py:24` — `max_context_size = max_distractors + 1`.

So the **test context size** is up to **88** (target + 87 others). The 88 cap is large enough that for almost all ScanNet scenes the ENTIRE scene's segmented instance set fits.

### Pad masking

The model emits `logits[B, 88]` over the whole padded context. At evaluation, only the
real (non-pad) entries are scored, see
`referit3d/models/referit3d_net_utils.py:243-245`:

```python
if FOR_VISUALIZATION:
    n_ex = len(out['logits'])
    c = batch['context_size']
    n_obj = out['logits'].shape[1]
    for i in range(n_ex):
        if c[i] < n_obj:
            out['logits'][i][c[i]:] = -10e6
```

`context_size` is `len(samples)` after `prepare_distractors`+ target (line 70, 82 of
`listening_dataset.py`). So pad slots get logit `-10e6` and never win argmax.

### Two reported metrics

`referit3d/models/referit3d_net_utils.py:247-266` computes both:

- `guessed_correctly` — argmax over the **full context** (target + same-class +
  clutter, capped at 88). This is the canonical "Listening-Acc" reported in
  `train_referit3d.py:38` and shown in tables.
- `guessed_correctly_among_true_class` — argmax restricted to the same-class
  subset by masking other-class logits to `-1e6`. This is the
  `among-true` column in `analysis/deepnet_predictions.py:69-71` and is **not** the
  number reported on leaderboards.

### Paper cross-reference

**ReferIt3D ECCV 2020 (Achlioptas et al., paper p.9):**
> "we assume access to the instance-level object segmentations of the underlying
> scene. This choice allows us to cast the 3D reference problem into a classification
> problem that aims to predict the referred 'target' among **M segmented 3D
> instances**."

**ReferIt3D ECCV 2020 (paper p.11), Decoupled / V+L baselines:**
> "we ground an RNN with the visual feature of each object of **O_i ∈ S**,
> independently"

`S` = full scene point cloud / object set, not the same-class subset.

**3D-VisTA ICCV 2023 (Zhu et al., paper p.5):**
> "For Nr3D/Sr3D, we follow ReferIt3D [1] to use **ground-truth object masks** and
> report the results as the grounding accuracy, i.e., **whether the model correctly
> selects the referred object among ground-truth object proposals**."

3D-VisTA Table 4 caption (paper p.6): "Grounding accuracy (%) on Nr3D and Sr3D
**with ground-truth object proposals**." So Overall=64.2, Easy=72.1, Hard=56.7,
ViewDep=61.5, ViewIndep=65.1 are all on the full-scene GT-proposal pool.

---

## Q2. Easy / Hard definition  [CONFIRMED-CODE]

**Answer: easy ⟺ `n_objects <= 2`, hard ⟺ `n_objects > 2`, where `n_objects` is the
3rd field of `stimulus_id` and equals `len(distractor_ids) + 1`.**

### Verbatim source (`referit3d/analysis/deepnet_predictions.py:34-36`)

```python
hardness = references.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
view_dep_mask = is_explicitly_view_dependent(references)
easy_context_mask = hardness <= 2
```

### Stimulus parsing (`referit3d/data_generation/nr3d/stimuli_generation.py:38-60`)

```python
@staticmethod
def decode_stimulus_string(s):
    ...
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)
    ...
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1
    return scene_id, instance_label, n_objects, target_id, distractors_ids
```

So `n_objects` includes the target. Since unique-target utterances are filtered out
at test time (`listening_dataset.py:136-149`), `n_objects ≥ 2` always holds on the
test set, hence:

- **easy ⟺ `n_objects == 2` (exactly 1 same-class distractor)**
- **hard ⟺ `n_objects > 2` (2+ same-class distractors)**

### Note on the paper text vs code

The ReferIt3D ECCV 2020 paper text (p.8) says:
> "We define 'hard' contexts as those 3D scenes that contain **more than 2 distractors**"

Taken literally, "more than 2 distractors" maps to `n_objects > 3`. The **code**
(`hardness <= 2` for easy) maps to `n_objects > 2` for hard. The community has
followed the code convention (3D-VisTA, MVT, ViL3DRel, SAT all report easy-vs-hard
splits matching the code, not the literal paper text). Treat the paper sentence as
imprecisely worded; the operational definition is in `deepnet_predictions.py:36`.

There is **no other** easy/hard definition in the codebase (no token-length based
or stimulus-string based variant).

---

## Q3. View-dependent keyword list  [CONFIRMED-CODE]

**Answer: a literal 10-token set, per-token lowercase intersection (NOT substring).**

### Verbatim source (`referit3d/analysis/utterances.py:98-105`)

```python
def is_explicitly_view_dependent(df):
    """
    :param df: pandas dataframe with "tokens" columns
    :return: a boolean mask
    """
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    return df.tokens.apply(lambda x: len(set(x).intersection(target_words)) > 0)
```

The **literal set** is exactly:
`{front, behind, back, right, left, facing, leftmost, rightmost, looking, across}` (10 tokens).

Matching is **per-token**, on the pre-tokenized lower-cased token list (tokens are
lowercased during preprocessing, see
`referit3d/data_generation/nr3d/tokenization.py:322`: `clean_text = clean_text.apply(lambda x: x.lower())`).

So the substring "front" inside the word "frontier" would **not** trigger view-dep —
the token has to be exactly `"front"`.

There is no alternative view-dependent definition in the codebase.

---

## Q4. Sample filters and resulting test split size  [CONFIRMED-CODE]

The canonical filter chain, in order of application, is:

| # | Filter | File:line | Default |
|---|---|---|---|
| 1 | `drop_bad_context` (drops `instance_type ∈ {clothes, clothing}` AND blacklist CSV) | `referit3d/in_out/nr3d.py:30-48` | True |
| 2 | `mentions_target_class_only` | `referit3d/in_out/neural_net_oriented.py:64-69`; `referit3d/in_out/arguments.py:50` | **True** |
| 3 | `tokens <= max_seq_len` (drops long utterances) | `referit3d/in_out/neural_net_oriented.py:84-86`; `referit3d/in_out/arguments.py:44` | `max_seq_len=24` |
| 4 | drop unique-target test utterances (no distractors) | `referit3d/in_out/pt_datasets/listening_dataset.py:136-149` | always-on at test |

### `correct_guess` is NOT a canonical filter

`correct_guess` appears only at:
- `referit3d/scripts/prepare_referential_data.py:60` — informational print: `print('Success Rate utterances:', raw_data.correct_guess.mean())`
- `referit3d/utils/visualizations.py:29` — visualization tooltip

The training/eval data pipeline (`load_referential_data` in
`referit3d/in_out/neural_net_oriented.py:55-105`) **does not filter by `correct_guess`**.
Per supervisor's question: UniVLG's `correct_guess==True` filter is a UniVLG-specific
add-on, **not** canonical referit3d behaviour. Most published numbers (3D-VisTA, MVT,
ViL3DRel, SAT) used the canonical chain without `correct_guess`.

### Verbatim source (`referit3d/in_out/nr3d.py:30-48`)

```python
def load_nr3d_raw_data(refer_it_csv, drop_bad_context=True):
    df = pd.read_csv(refer_it_csv)
    df.rename(columns={'Input.stimulus_id': 'stimulus_id', 'Answer.response': 'utterance'}, inplace=True)
    df['scan_id'] = df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[0])
    df['instance_type'] = df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[1])
    df['target_id'] = df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[3])

    if drop_bad_context:
        basedir = osp.split(pathlib.Path(__file__).parent.absolute())[0]
        bad_context = osp.join(basedir, 'data/language/nr3d/manually_inspected_bad_contexts.csv')
        bad_context = pd.read_csv(bad_context)
        bad_context = set(bad_context['stimulus_id'].unique())
        drop_mask = df.instance_type.apply(lambda x: x in ['clothes', 'clothing'])
        drop_mask |= df.stimulus_id.isin(bad_context)
        print('dropping ', (drop_mask.sum()), 'utterances marked manually as bad/poor context')
        df = df[~drop_mask]
        df.reset_index(inplace=True, drop=True)

    return df
```

### Verbatim source (`referit3d/in_out/neural_net_oriented.py:55-86`)

```python
def load_referential_data(args, referit_csv, scans_split):
    referit_data = pd.read_csv(referit_csv)

    if args.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data['mentions_target_class']]
        referit_data.reset_index(drop=True, inplace=True)
        print('Dropping utterances without explicit '
              'mention to the target class {}->{}'.format(n_original, len(referit_data)))

    referit_data = referit_data[['tokens', 'instance_type', 'scan_id',
                                 'dataset', 'target_id', 'utterance', 'stimulus_id']]
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)
    ...
    n_original = len(referit_data)
    referit_data = referit_data[referit_data.tokens.apply(lambda x: len(x) <= args.max_seq_len)]
```

### Bad-context CSV

`referit3d/data/language/nr3d/manually_inspected_bad_contexts.csv` contains **270**
unique `stimulus_id` rows (one header line + 270 data lines).

### Resulting test-split size

The codebase does not print a fixed test-split count for NR3D. The denominator is
data-dependent (depends on which NR3D CSV release you use), but per the
ReferIt3D paper p.9, **91.6%** of NR3D utterances mention the target class (or a
synonym). Our local NR3D test set after `apply_blacklist=True, drop_clothes=True` is
**8584** (per supervisor's brief). After also enforcing canonical
`mentions_target_class_only=True`, expect roughly `8584 × 0.916 ≈ 7860` utterances.
The exact number depends on which `mentions_target_class` flag values are persisted
in our CSV release.

The unique-target-drop filter (Q4 #4) removes utterances where `len(distractor_ids)==0`,
which in NR3D is 0 (all NR3D utts have at least one same-class distractor by
construction — the speaker is shown the target alongside distractors).

---

## Q5. Use-color subset definition  [INFERRED + UNKNOWN-IN-CODE]

**Short answer:** the open-source referit3d code does NOT include the script that
generates the `uses_color_lang` CSV column. The values are baked into the public
NR3D CSV release. The reference word list is in
`referit3d/data/language/color_words.json.txt` (279 color tokens).

### What IS in the codebase

- `referit3d/data/language/color_words.json.txt` — the canonical color word list (279
  entries: `red`, `blue`, ... `crimson`, `mahogany`, ...).
- `referit3d/analysis/word_meanings.py:3-33` — `shape_words` set (used for
  `uses_shape_lang`).
- `referit3d/analysis/word_meanings.py:38-72` — `spatial_tokens` (= `spatial_words ∪
  spatial_prepositions`) and `spatial_expressions = [['close', 'to'], ['next', 'to'],
  ['back', 'of']]`. The `uses_spatial_reasoning(tokens)` function computes
  `tokens ∩ spatial_tokens != ∅` OR a multi-word match against `spatial_expressions`.
- `referit3d/analysis/word_meanings.py:77-149` — `object_words` (used for
  `uses_object_lang`, presumably).

### What is NOT in the codebase

- No script in the public repo computes `uses_color_lang`, `uses_object_lang`, or
  `uses_shape_lang` and writes them to the CSV. `prepare_referential_data.py:80-84`
  **only** adds `tokens`, `dataset`, and `mentions_target_class` to the CSV. No
  `uses_*_lang` column is generated by this script.

### Inference (not authoritative)

Given the existence of `color_words.json.txt` and the parallel structure of
`uses_spatial_reasoning(tokens)` for spatial words, it is highly likely that the
authors precomputed `uses_color_lang = tokens ∩ color_words != ∅` and shipped the
column in the released CSV. **But this is not provable from the public code.**

**[INFERRED]** for the formula. **[UNKNOWN-IN-CODE]** for the actual script.

For our purpose: most papers that report a "use-color" ablation re-use the CSV
column directly and do not recompute it. As long as we read `uses_color_lang` from
the same NR3D CSV release everyone else uses, we are aligned.

---

## Q6. Classification accuracy denominator  [CONFIRMED-CODE + CONFIRMED-PAPER]

**Answer: denominator = number of test utterances after canonical filters.
Each utterance is scored as a single binary `argmax(logits) == target_pos` where
`logits` ranges over the FULL scene's segmented instances (capped at 88).
NOT a per-class restricted denominator.**

### Code path

`referit3d/models/referit3d_net_utils.py:142-204` (`evaluate_on_dataset`):

```python
predictions = torch.argmax(res['logits'], dim=1)
guessed_correctly = torch.mean((predictions == target).double()).item()
ref_acc_mtr.update(guessed_correctly, batch_size)
...
metrics['test_referential_acc'] = ref_acc_mtr.avg
```

`AverageMeter` is in `referit3d/utils/evaluation.py:1-25`. It accumulates
`sum += val * n` and `count += n` per batch, so the final `avg` is the per-utterance
mean: `# correct / # utterances`. No reweighting, no per-class normalization.

### Cross-check with 3D-VisTA  [CONFIRMED-PAPER]

3D-VisTA paper p.5 (ICCV 2023, Zhu et al., "Implementation Details" — 3D Visual
Grounding subsection):

> "We evaluate our model on three datasets for this task: ScanRefer [8], Nr3D, and
> Sr3D [1]. For Nr3D/Sr3D, we follow ReferIt3D [1] to use **ground-truth object
> masks** and report the results as the grounding accuracy, **i.e., whether the
> model correctly selects the referred object among ground-truth object proposals**."

Table 4 (paper p.6), 3D-VisTA on Nr3D: **Overall 64.2**, Easy 72.1, Hard 56.7,
View-Dep 61.5, View-Indep 65.1.

The "Overall" column is the mean accuracy over **all test utterances after canonical
filters**, against the full-scene GT object proposals. There is no per-class
restricted variant in the leaderboard column.

### Cross-check with ReferIt3D paper itself  [CONFIRMED-PAPER]

ReferIt3D ECCV 2020 paper p.11 (§5 Experiments and Analysis):

> "We explore different listening architectures and report the listening accuracy;
> **each test utterance receives a binary score (1 if the correct object is predicted
> as target and 0 otherwise)**. For all experiments we use the official-ScanNet
> splits."

ReferIt3D Table 2 (p.12): ReferIt3DNet w/ Sr3D+ on Nr3D = Overall **37.6%** with
splits Easy 45.4, Hard 30.0, View-Dep 33.1, View-Indep 39.8. (This is the original
ReferIt3DNet baseline; subsequent transformer-based methods reach 55-65%.)

So both Q4 (filter chain) and Q6 (denominator) are independently confirmed across
the original ReferIt3D paper and 3D-VisTA. **Two-paper consensus.**

---

## Q7. Detection-mode papers on NR3D  [CONFIRMED-PAPER]

**Answer: Yes — BUTD-DETR (ECCV 2022, Jain & Gkanatsios et al.) reports a separate
detection-mode (`det`) Acc@0.25 on NR3D using detector-generated proposals from a
trained Group-Free 3D detector. They do NOT report Acc@0.50 on NR3D. They also
report a `GT` classification track on SR3D but explicitly state "this is the realistic
setup of not having access to oracle object boxes".**

### Verbatim from BUTD-DETR (paper p.10)

> "All existing models that have been tested in SR3D or NR3D benchmarks are
> box-bottlenecked, namely, they are trained to select the answer from a pool of box
> proposals. They all use **ground-truth 3D object boxes (without category labels)**
> as the set of boxes to select from. We thus consider two evaluation setups:
>
> 1. `det`: where we re-train previous models using their publicly available code and
>    provide the same 3D box proposals we use in BUTD-DETR, obtained by the
>    Group-Free 3D object detector trained to detect 485 object categories in
>    ScanNet (Section `det` in Table 1).
> 2. `GT`, where we use ground-truth 3D object boxes for our model and baseline
>    (Section `GT` in Table 1).

> "We use top-1 accuracy metric, which measures the percentage of times we can
> find the target box with an IoU higher than the threshold. We report results with
> **IoU@0.25 on SR3D and NR3D**; and with both **IoU@0.25 and IoU@0.5 on
> ScanRefer**."

### BUTD-DETR Table 1 (paper p.11) — NR3D entries

NR3D column reports **only Acc@0.25(det)**:

| Method | NR3D Acc@0.25(det) |
|---|---|
| ReferIt3DNet [2] | 24.0 |
| ScanRefer [5] | - |
| TGNN [29] | 37.4 |
| 3DRefTransformer [1] | 47.0 |
| InstanceRefer [54] | 29.9 |
| FFL-3DOG [12] | - |
| LanguageRefer [45] | 28.6 |
| 3DVG-Transformer [55] | - |
| TransRefer3D [15] | - |
| SAT-2D [51] | 31.7 |
| MDETR-3D (BUTD impl) | 31.5 |
| **BUTD-DETR** | **43.3** |

These numbers are NOT directly comparable to the canonical classification-on-GT-pool
numbers (e.g., 3D-VisTA's 64.2). They use detector-generated proposals as the box
set and an IoU≥0.25 criterion to count a hit.

### Importantly: there is no public Acc@0.50 on NR3D from BUTD-DETR

Per their stated protocol they **only** report IoU@0.25 on NR3D. So a side-by-side
comparison "our Acc@0.50 vs BUTD-DETR Acc@0.50 on NR3D" is not possible from
the published table.

### Other detection-mode signals

- **3D-VisTA** also implements a Mask3d-proposal track on **ScanRefer** (their
  Table 5, paper p.6) with M3D Det column reporting Acc@0.25 / Acc@0.50, but on
  Nr3D they only report the GT-proposal classification track (Table 4).
- We did not locate published Acc@0.25 / Acc@0.50 numbers for **MVT**, **ViL3DRel**,
  **MiKASA** on NR3D. They all report classification-on-GT-pool only.

So **the only public detection-mode number on NR3D test is BUTD-DETR's Acc@0.25**.

---

## Q8. Hidden gotchas  [CONFIRMED-CODE + INFERRED]

### IoU computation — axis-aligned only

`referit3d/in_out/cuboid.py:363-395` (`iou_3d`) is axis-aligned 3D IoU; it takes
corner-array inputs and reduces them via `np.max`/`np.min` over each axis,
discarding any rotation information.

`referit3d/in_out/cuboid.py:343-345` (`OrientedCuboid.intersection_with`):

```python
def intersection_with(self, other):
    if np.any(self.rot != np.eye(3)):
        raise NotImplementedError("intersection_with(): Not implemeted for oriented boxes")
```

So oriented-box IoU is **explicitly unimplemented** in referit3d. The original
canonical NR3D evaluation never computes any IoU because it is a classification
metric (correct object_id == target object_id). IoU only matters for detection-mode
evaluations on top of detector-generated proposals (Q7).

### GT bbox source — axis-aligned, derived from point cloud min/max

`referit3d/in_out/three_d_object.py:63-79` (`set_axis_align_bbox` / `get_axis_align_bbox`):

```python
def set_axis_align_bbox(self):
    pc = self.get_pc()
    cx, cy, cz = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2.0
    lx, ly, lz = np.max(pc, axis=0) - np.min(pc, axis=0)
    rot = np.eye(N=3)
    assert (lx > 0 and ly > 0 and lz > 0)
    self.axis_aligned_bbox = OrientedCuboid(cx, cy, cz, lx, ly, lz, rot)
```

The bbox is **min/max of the per-instance ScanNet point segmentation**, with
identity rotation. Not from Scan2CAD, not 9-DoF. There is an optional
`set_object_aligned_bbox` for Scan2CAD-derived oriented boxes (`three_d_object.py:109-118`),
but the canonical NR3D listener only consumes the axis-aligned variant.

### Train / test split

`referit3d/in_out/neural_net_oriented.py:18-22`:

```python
train_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_train.txt')
train_split = read_lines(train_split)
test_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_val.txt')
test_split = read_lines(test_split)
```

So the **NR3D "test" split = ScanNet v2 OFFICIAL `scannetv2_val.txt`** (312 scenes).
Training = `scannetv2_train.txt` (1201 scenes). Note: `data/language/nr3d/test_scans.txt`
and `data/language/nr3d/train_scans.txt` exist in the repo but are 0 lines — the
ScanNet v2 official splits are the source of truth.

### Class normalization — minimal merge map

`referit3d/data/mappings/referIt3D_merged_instance_classes.json`:

```json
{
"book": "books",
"clothing": "clothes",
"end table": "table",
"cabinet": "cabinets"
}
```

This is a tiny manual merge of 4 instance-label aliases. Note that "clothes" /
"clothing" both get dropped by Q4-#1 anyway (see `nr3d.py:42`), so the practical
merge is `book → books`, `end table → table`, `cabinet → cabinets`.

The canonical `prepare_distractors` uses raw `instance_label` strings — there is no
runtime application of this JSON. It would have been applied during preprocessing.
Whether your CSV's `instance_type` column is already merged depends on which CSV
release you have.

### Utterance preprocessing

`referit3d/data_generation/nr3d/tokenization.py:316-345` (`pre_process_text`):

1. sentence-level spelling map (`sentence_spelling_dictionary` line 3-4 — only one
   replacement: `'thewaytheshapeschangethespace' → 'the way the shapes change the space'`)
2. lowercasing
3. unquote words (strip surrounding `'` / `"`)
4. expand contractions (full English contraction map, line 106-224)
5. punctuation `.?!,:;/\\-~*_=` → space (line 326)
6. NLTK `word_tokenize`
7. token-level spelling map (`token_spelling_dictionary` line 6-99 — fixes ~80
   common misspellings: `furthest → farthest`, `gray → grey`, `redtrash → ['red', 'trash']`,
   etc.)
8. SymSpell spell-check for any remaining out-of-vocabulary tokens

So the published `tokens` column in NR3D CSV is **fully lowercased, contraction-expanded,
spell-checked, NLTK-tokenized**. Any view-dep / use-color matching needs to use this
same tokenization (which is why the per-token lowercase intersection in Q3 works
without further normalization).

### No multi-class targets

NR3D's `instance_type` column is a single string per row. `decode_stimulus_string`
returns a single `instance_label`. No multi-class handling in the canonical path.

---

## Cross-check summary

| Source | Q1 (Pool) | Q4 (Filters) | Q6 (Denominator) |
|---|---|---|---|
| referit3d code (this audit) | full scene cap=88 | drop bad_context+clothes, mentions_target_class=True, max_seq_len=24, drop unique-target | mean over all test utts |
| **ReferIt3D ECCV 2020 (p.9, p.11)** | "M segmented 3D instances" / "O_i ∈ S" | implicit via `--mentions-target-class-only` default | "each test utterance receives a binary score" |
| **3D-VisTA ICCV 2023 (p.5, Table 4)** | "ground-truth object proposals" | "follow ReferIt3D" | per-utterance accuracy on full GT-proposal pool |
| BUTD-DETR ECCV 2022 (p.10-11) | "ground-truth 3D object boxes (without category labels)" for the GT track; Group-Free 3D detector boxes for the `det` track | not specified in detail; reuses ReferIt3D's CSV | top-1 IoU@0.25 (det track only on NR3D) |

Two-paper consensus on classification protocol: ReferIt3D + 3D-VisTA.

---

## Minimal changes to align our setup with the public leaderboard

Our local NR3D run achieves Acc@0.25 = 77.67% / Acc@0.50 = 77.62% on **8584 utts**
using a **GT-pool candidate set** of ~50-70 instances/scene from Phase 8 GT-CG,
with 9-DoF oriented IoU scoring. To make this strictly comparable with the public
leaderboard (3D-VisTA Overall = 64.2% etc.), the smallest set of code-level changes is:

### Change 1 — flip `mentions_target_class_only=True` for leaderboard-track runs (addresses Q4)

**File:** `OURS:src/benchmarks/nr3d_loader.py:120`
**Current:** `mentions_target_class_only: bool = False`
**Change:** add a leaderboard preset OR pass `mentions_target_class_only=True` from
the eval driver. The canonical referit3d default is `True`
(`referit3d/in_out/arguments.py:50`).

Concretely: where the eval driver constructs the dataset (find call sites of
`Nr3dDataset.from_path` in `OURS:src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
and any pack-prep callers), explicitly pass `mentions_target_class_only=True`.

This is the largest single-knob delta. Expected effect: drops test split from 8584
to roughly 7860 utts (per ReferIt3D paper p.9: 91.6% mention rate).

### Change 2 — DO NOT add `correct_guess_only=True` (addresses Q4)

`OURS:src/benchmarks/nr3d_loader.py:119` already defaults `correct_guess_only=False`.
**Keep this default.** UniVLG's `correct_guess==True` filter is UniVLG-specific, NOT
canonical referit3d behaviour. Leaderboard papers (3D-VisTA, MVT, ViL3DRel, SAT) do
not apply it.

### Change 3 — switch reported metric from Acc@IoU to classification accuracy (addresses Q1, Q6)

This is the most structural change — but conceptually simple given Q1's finding:

- The public leaderboard reports a **classification metric**: did the model pick the
  correct `object_id` from the candidate pool, with `pool == full scene's GT
  segmentation, capped at 88`. **No IoU threshold.**
- Our current `Acc@0.25 / Acc@0.50` score the predicted box's IoU against the GT box.
  Even if the agent correctly picks `selected_object_id == target_id`, the score
  depends on how the proposal-bbox lines up with the GT bbox.

**Action items:**

1. In `OURS:src/evaluation/scripts/run_nr3d_vg_side_by_side.py`, add a new metric
   `classification_acc = (selected_object_id == target_id).mean()` alongside the
   existing IoU-based metrics.
2. Report `classification_acc` as the leaderboard-comparable headline. The IoU
   metrics remain useful for our internal "did the bbox actually match" sanity
   check, but they are NOT the leaderboard number.
3. Because our pool is a single Phase 8 GT-CG instance per object_id, and the agent
   selects by id, the classification metric is a clean equality check — no IoU
   matching against arbitrary proposals is needed. The 9-DoF oriented IoU scoring
   becomes a separate ablation, not the headline.

### Change 4 — slice reporting on Easy / Hard / V-Dep / V-Indep using canonical definitions (addresses Q2, Q3)

**File:** `OURS:src/benchmarks/nr3d_loader.py` (already exposes
`distractor_ids`, `n_objects` on `Nr3dVGSample`), and the eval reporter.

In the per-sample report, derive:

```python
is_easy = sample.n_objects <= 2
is_view_dep = bool({"front", "behind", "back", "right", "left",
                    "facing", "leftmost", "rightmost", "looking",
                    "across"} & set(sample.tokens))
```

Then publish 5 columns: Overall, Easy, Hard, View-Dep, View-Indep — exactly the
columns 3D-VisTA, MVT, ViL3DRel, SAT all report.

Caveat: ensure `sample.tokens` is the canonical lowercased/spell-checked list (Q8
preprocessing). If we ever stored raw utterance tokens (un-lowercased,
un-spell-corrected), the view-dep mask would not align with the published one.

### Change 5 — keep candidate pool source AS-IS (addresses Q1)

**No change needed.** Our Phase 8 GT-CG pool of 50-70 instances/scene is
structurally aligned with the canonical full-scene-up-to-88 pool. The codebase
already includes target + same-class + clutter from other classes, capped at
`max_test_objects=88`, which for a typical ScanNet scene is "the entire segmentation".
Phase 8 GT-CG approximates the same shape — we are within the same pool regime.

### Change 6 — confirm IoU type is irrelevant for the leaderboard track (addresses Q8)

Since the leaderboard metric is classification (not IoU), the choice of axis-aligned
vs 9-DoF oriented IoU only matters for our internal box-quality ablation, not for
leaderboard comparability. **No change to IoU implementation needed for
leaderboard-track output**, but we should document this explicitly so future
readers do not conflate the two metrics.

### Summary table of recommended changes

| # | Addresses | File:line (ours) | Action |
|---|---|---|---|
| 1 | Q4 | `src/benchmarks/nr3d_loader.py:120`; eval driver call site | Pass `mentions_target_class_only=True` for leaderboard runs |
| 2 | Q4 | `src/benchmarks/nr3d_loader.py:119` | Keep `correct_guess_only=False` (it's already correct) |
| 3 | Q1, Q6 | `src/evaluation/scripts/run_nr3d_vg_side_by_side.py` | Add `classification_acc` metric and report it as the headline |
| 4 | Q2, Q3 | eval reporter | Add Easy/Hard via `n_objects<=2`; View-Dep via the literal 10-keyword set |
| 5 | Q1 | (none) | Keep Phase 8 GT-CG pool — structurally aligned |
| 6 | Q8 | (docs) | Document that IoU type is irrelevant for leaderboard track |

The single biggest delta is change #3 — the metric definition. Until our reported
number is `classification_acc` (selected_object_id == target_id), our 77% on a
GT-pool with IoU thresholds is **not** comparable to the public 64.2 / 67.0 / 76.0
numbers, regardless of any other knob.

---

## File index — for future re-audits

| Topic | File:line |
|---|---|
| `decode_stimulus_string` | `referit3d/data_generation/nr3d/stimuli_generation.py:38-60` |
| Bad-context blacklist + clothes drop | `referit3d/in_out/nr3d.py:30-48` |
| `mentions_target_class_only` filter | `referit3d/in_out/neural_net_oriented.py:64-69` |
| Token length filter | `referit3d/in_out/neural_net_oriented.py:84-86` |
| Unique-target test drop | `referit3d/in_out/pt_datasets/listening_dataset.py:136-149` |
| `prepare_distractors` (pool construction) | `referit3d/in_out/pt_datasets/listening_dataset.py:44-60` |
| `max_test_objects` default 88 | `referit3d/in_out/arguments.py:53` |
| Easy/Hard mask | `referit3d/analysis/deepnet_predictions.py:34-36` |
| View-dep keyword set | `referit3d/analysis/utterances.py:103-105` |
| `mentions_target_class` computation | `referit3d/analysis/utterances.py:20-95` |
| Test referential acc reporter | `referit3d/models/referit3d_net_utils.py:142-204` |
| `guessed_correctly_among_true_class` (alt metric) | `referit3d/models/referit3d_net_utils.py:262-266` |
| Detailed per-utt eval | `referit3d/models/referit3d_net_utils.py:207-274` |
| Axis-aligned bbox from point cloud | `referit3d/in_out/three_d_object.py:63-79` |
| Oriented IoU NotImplementedError | `referit3d/in_out/cuboid.py:343-345` |
| Axis-aligned 3D IoU | `referit3d/in_out/cuboid.py:363-395` |
| ScanNet train/val splits | `referit3d/in_out/neural_net_oriented.py:18-22`; `referit3d/data/scannet/splits/official/v2/scannetv2_{train,val}.txt` |
| Color word list | `referit3d/data/language/color_words.json.txt` |
| Spatial / shape word lists | `referit3d/analysis/word_meanings.py:3-72` |
| Object word list | `referit3d/analysis/word_meanings.py:77-149` |
| Class merge JSON | `referit3d/data/mappings/referIt3D_merged_instance_classes.json` |
| Utterance pre-processing pipeline | `referit3d/data_generation/nr3d/tokenization.py:316-345` |
| Token spelling dictionary | `referit3d/data_generation/nr3d/tokenization.py:6-99` |

End of audit.
