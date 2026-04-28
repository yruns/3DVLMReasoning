# v15 chassis-repro — Detailed Severe Regression Cases

**Companion to:** `v15_chassis_repro_20260427_regressions.md` (which gave the high-level patterns).
**Source:** SQLite `docs/benchmark/openeqa/runs.sqlite` + raw stage2.json artifacts.
**Scope:** the 38 cases where chassis stage2_score=1 AND v15 stage2_score ∈ {4,5} — the questions chassis *broke* that v15 had right.

Each case shows: question + GT, the v15 vs chassis trace (tools used, confidence, final answer, uncertainties). Pattern bucket assigned by post-hoc inspection.

## Index

- **attribute recognition**: 2 cases
- **functional reasoning**: 2 cases
- **object localization**: 4 cases
- **object recognition**: 9 cases
- **object state recognition**: 4 cases
- **spatial understanding**: 12 cases
- **world knowledge**: 5 cases
- **TOTAL**: 38 cases

## Category: attribute recognition

### `5178b12e-a949-4788-bbfe-36ecd035cec8`  (v15=5 → chassis=1)

- **scene**: 113-scannet-scene0207_00
- **question**: What color is the duvet cover?
- **GT answer**: dark green

**v15** (kf 3 → 7, conf 0.82):

  1. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['duvet cover', 'comforter', 'bed covering']. Each crop is 2.0x the object bbox with r

  **answer**: The duvet cover appears dark green, with white bedding also visible underneath/on the bed.
  **uncertainties**: The bedding is somewhat blurry, so it is slightly ambiguous whether the queried 'duvet cover' refers to the dark green c

**chassis** (kf 3 → 7, conf 0.81):

  1. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['duvet cover', 'bed cover', 'comforter']. Each crop is 2.0x the object bbox with red 

  **answer**: White or off-white.
  **uncertainties**: The images are somewhat blurry and dim, so the exact shade is approximate (white vs. off-white/cream).

---

### `e35875b7-5844-4f4b-9354-1ac985495869`  (v15=4 → chassis=1)

- **scene**: 047-scannet-scene0747_00
- **question**: What color are the patio chairs?
- **GT answer**: White

**v15** (kf 3 → 5, conf 0.67):

  1. **request_more_views** — `mode=targeted | Need views that clearly show any patio/outdoor chairs visible through the window`
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [111]. Total keyframes now: 4.
  2. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['patio chair', 'outdoor chair', 'chairs on patio', 'balcony chair']. Each crop is 2.0

  **answer**: The patio chairs look white or light beige.
  **uncertainties**: The outdoor chairs are seen through bright window light, so the exact shade is slightly washed out. | Only part of the patio seating is clearly visible, so the description is approximate between white and light beige.

**chassis** (kf 3 → 28, conf 0.20):

  1. **request_more_views** — `mode=targeted | Need views that clearly show any patio/outdoor chairs. Current keyframes only sh`
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [111]. Total keyframes now: 4.
  2. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['patio chair', 'outdoor chair', 'chair on patio', 'chairs outside']. Each crop is 2.0
  3. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  4. **request_more_views** — `mode=explore | Need the newly added keyframe and any additional unexplored views that show the `
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [86, 84]. Total keyframes now: 7.
  5. **request_more_views** — `mode=targeted | Need a view through or at the patio/deck area that clearly shows any outdoor cha`
     → No additional views available. Current keyframes already provide best coverage.
  6. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  7. **retrieve_object_context** — `['patio chair', 'outdoor chair', 'chair', 'door', 'window', 'deck', 'patio']`
     → {   "Queen Anne dining chair": "A traditional Queen Anne style dining chair with a dark, polished wood frame. It feature
  8. **switch_or_expand_hypothesis** — `{'request_text': "The user asks specifically about patio chairs, but current ret`
     → No new_query specified for hypothesis switching.
  9. **switch_or_expand_hypothesis** — `{'request_text': "Switch retrieval to focus on patio/outdoor chairs, deck chairs`
     → No new_query specified for hypothesis switching.
  10. **request_more_views** — `mode=targeted | Find views that maximize visibility through the large windows/doors to the outsi`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 9.
  11. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['window view outside', 'patio area', 'deck area', 'outdoor seating', 'chair outside w
  12. **request_more_views** — `mode=targeted | Need the generated outside-facing views returned as inspectable keyframes; speci`
     → No additional views available. Current keyframes already provide best coverage.
  13. **switch_or_expand_hypothesis** — `{'request_text': 'New query: patio chairs. Retrieve evidence focused specificall`
     → No new_query specified for hypothesis switching.
  14. **switch_or_expand_hypothesis** — `{'request_text': "Please re-run retrieval with new_query='patio chairs' and prio`
     → No new_query specified for hypothesis switching.
  15. **retrieve_object_context** — `['chaise lounge', 'chair', 'dining chair', 'outdoor seating', 'patio', 'deck']`
     → {   "Queen Anne dining chair": "A traditional Queen Anne style dining chair with a dark, polished wood frame. It feature
  16. **request_crops** — ``
     → Generated 6 annotated crop(s) for ['window', 'glass door', 'outside area', 'deck railing', 'possible chair outside']. Ea
  17. **request_more_views** — `mode=targeted | Return any neighboring or alternate views around frame 0 that look out through t`
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [20]. Total keyframes now: 19.
  18. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['patio furniture', 'outdoor chair', 'deck chair', 'chair outside window']. Each crop 
  19. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  20. **switch_or_expand_hypothesis** — `{'request_text': 'Expand retrieval beyond indoor chairs. New query patio chairs `
     → No new_query specified for hypothesis switching.
  21. **request_more_views** — `mode=explore | Show maximally different views around the newly found outside-facing area near f`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [266, 231]. Total keyframes now: 24.
  22. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['patio chair', 'outdoor chair', 'deck chair', 'patio furniture', 'chair']. Each crop 

  **answer**: Insufficient evidence to determine the patio chairs' color.
  **uncertainties**: The most recently generated crops/keyframes were not displayed here for direct pixel inspection. | No visible image in this exchange contains a clearly identifiable patio chair.

---

## Category: functional reasoning

### `32d69796-26fb-4467-8f17-786ebf010ece`  (v15=5 → chassis=1)

- **scene**: 137-scannet-scene0598_01
- **question**: I need to charge my laptop, where can I find a power outlet?
- **GT answer**: on the brown desk

**v15** (kf 3 → 9, conf 0.48):

  1. **request_more_views** — `mode=targeted | Need views where a power strip or wall/floor power outlet is actually visible. C`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  3. **retrieve_object_context** — `['power strip', 'power outlet', 'table', 'desk', 'computer']`
     → {   "table divider rail": "A metal rail system mounted on a large wooden table. It consists of several vertical, silver-
  4. **switch_or_expand_hypothesis** — `{'request_text': "Expand retrieval from 'power strip' to include wall outlet, fl`
     → No new_query specified for hypothesis switching.
  5. **request_more_views** — `mode=explore | Explore any unseen regions of the scene that might show wall outlets, floor outl`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [200, 199]. Total keyframes now: 5.
  6. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['power outlet', 'power strip', 'desk power module', 'cable grommet', 'under-table are
  7. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  8. **request_more_views** — `mode=targeted | Need targeted views of the large wooden study table, especially the metal divide`
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [186]. Total keyframes now: 7.
  9. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['table divider rail', 'desk power module', 'cable grommet', 'power outlet', 'power st

  **answer**: Check the large wooden study table. Near the metal divider rail on top of the table, there appears to be a built-in power/cable access module, which is the best visible place to plug in your laptop.
  **uncertainties**: A standard wall outlet or floor outlet is still not clearly visible. | The evidence suggests an integrated table power/cable module rather than a clearly readable socket face. | Newly added frames/crops were not individually rendered with file paths in the tool output, so the conclusion relies on 

**chassis** (kf 3 → 7, conf 0.67):

  1. **request_more_views** — `mode=targeted | Need additional views that clearly show any power outlets or charging ports in t`
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [135]. Total keyframes now: 4.
  2. **request_more_views** — `mode=explore | Explore additional unseen room regions near frame 135 and other areas of the mee`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [9, 11]. Total keyframes now: 6.
  3. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['power outlet', 'electrical outlet', 'wall outlet', 'floor outlet', 'table power port
  4. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "proxy_grounded",     "hypothesis_kind": "proxy",     "hypothesis_rank": 2,     "parse
  5. **request_more_views** — `mode=targeted | Need a view centered on the detected candidate outlet from the crop, plus any ad`
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: You can use the power outlet mounted low on the wall at the right side of the room, near the window-side wall/corner.
  **uncertainties**: The original three keyframes do not show the outlet; it is only supported by the added crop evidence. | I do not have a wider localized frame clearly showing the outlet's position relative to the whole room, so the location 

---

### `e1f9b6b4-cf8f-417f-aa85-e32d6db75a77`  (v15=5 → chassis=1)

- **scene**: 179-scannet-scene0300_01
- **question**: What can I use to find where a particular book is located?
- **GT answer**: Use the catalog on the computer

**v15** (kf 3 → 5, conf 0.75):

  1. **request_more_views** — `mode=explore | Current views only show iMac computers and desks; no object for finding a book's`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [144, 143]. Total keyframes now: 5.

  **answer**: You could use the computer terminal/iMac on the desk to look up where a particular book is located.
  **uncertainties**: The newly added frames are not displayed inline here, so citation beyond the clearly visible computer views is limited. | The exact software or catalog function on the computers is not visible; this answer relies on the computer being the lik

**chassis** (kf 3 → 3, conf 0.56):

  1. **request_more_views** — `mode=targeted | Need evidence for what can be used to find where a particular book is located in`
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: The bookshelves/library shelving. That’s the only visible thing here you could use to locate a particular book; no computer catalog or directory sign is shown.
  **uncertainties**: A catalog computer or directory sign may exist elsewhere in the scene, but it is not visible in the provided frames and 

---

## Category: object localization

### `5b2c40fd-cee0-4a13-a9f1-d1ccf9b46521`  (v15=5 → chassis=1)

- **scene**: 158-scannet-scene0356_00
- **question**: What is the object under the phone at the corner?
- **GT answer**: a three-level cabinets

**v15** (kf 3 → 7, conf 0.66):

  1. **request_more_views** — `mode=targeted | Need views showing the phone mentioned in the query, especially a phone located `
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [5, 78]. Total keyframes now: 5.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "proxy_grounded",     "hypothesis_kind": "proxy",     "hypothesis_rank": 2,     "parse
  3. **retrieve_object_context** — `['Wi-Fi access point', 'portable speaker', 'phone']`
     → {   "portable speaker": "A small, black, rectangular object, consistent in shape and size with a portable speaker. It ha
  4. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['Wi-Fi access point', 'portable speaker', 'phone']. Each crop is 2.0x the object bbox

  **answer**: A wooden dresser/storage cabinet is under the phone at the corner.
  **uncertainties**: The visible phone appears to be a wall-mounted landline handset rather than a mobile phone. | Additional requested views/crops were used to support disambiguation, but the cited evidence comes from frames 1 and 2 w

**chassis** (kf 3 → 7, conf 0.45):

  1. **request_more_views** — `mode=targeted | Need views that clearly show the phone located at a corner and the object direct`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "proxy_grounded",     "hypothesis_kind": "proxy",     "hypothesis_rank": 2,     "parse
  3. **retrieve_object_context** — `['phone', 'portable speaker', 'corner', 'bed']`
     → {   "portable speaker": "A small, black, rectangular object, consistent in shape and size with a portable speaker. It ha
  4. **switch_or_expand_hypothesis** — `{'request_text': "The current proxy retrieval appears to have found a portable s`
     → No new_query specified for hypothesis switching.
  5. **switch_or_expand_hypothesis** — `{'request_text': 'Search for a phone or phone-like small rectangular handheld de`
     → No new_query specified for hypothesis switching.
  6. **switch_or_expand_hypothesis** — `{'request_text': 'Use a direct hypothesis for a phone or phone-like object at a `
     → No new_query specified for hypothesis switching.
  7. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['portable speaker']. Each crop is 2.0x the object bbox with red boxes marking all vis
  8. **request_more_views** — `mode=explore | Explore neighboring frames around the current keyframes to find any view showing`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [299, 298]. Total keyframes now: 6.
  9. **request_more_views** — `mode=targeted | Need targeted views of the small black rectangular object identified as a portab`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [5, 123]. Total keyframes now: 8.
  10. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['portable speaker', 'phone']. Each crop is 2.0x the object bbox with red boxes markin

  **answer**: It appears to be on the bed frame.
  **uncertainties**: A true phone is still not clearly visible in the images; the evidence points to a portable-speaker-like object used as t | The exact resting surface is at the bed corner and may be the bed frame/rail rather than a separate table or shelf.

---

### `9be4c0df-1754-4341-b1ac-afe3c503e8cc`  (v15=5 → chassis=1)

- **scene**: 145-scannet-scene0193_00
- **question**: Where is the two-tier shelf?
- **GT answer**: on top of the desk

**v15** (kf 3 → 3, conf 0.84):

(no tools)

  **answer**: The two-tier shelf is on top of the desk against the wall, near the door and beside the bed.

**chassis** (kf 3 → 5, conf 0.66):

  1. **request_more_views** — `mode=targeted | Need views that clearly show the suspected two-tier shelf/small shelving unit. C`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [62, 238]. Total keyframes now: 5.

  **answer**: The two-tier shelf is under the bed along the wall, near the back-left corner of the room.
  **uncertainties**: The shelf is dark and partly occluded by the bed frame, so only part of it is visible. | Its exact distance from the corner is approximate because the object is mostly seen in shadow.

---

### `b161deae-a42d-4fcb-b3b8-173016cf90d9`  (v15=5 → chassis=1)

- **scene**: 177-scannet-scene0608_02
- **question**: Where is the neck pillow?
- **GT answer**: On top of the piano keyboard

**v15** (kf 3 → 6, conf 0.92):

  1. **request_more_views** — `mode=targeted | Need additional views to verify the neck pillow's location in the room and groun`
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [224]. Total keyframes now: 4.
  2. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['pillow']. Each crop is 2.0x the object bbox with red boxes marking all visible objec

  **answer**: The neck pillow is on top of the keyboard, near the window.

**chassis** (kf 3 → 6, conf 0.81):

  1. **request_more_views** — `mode=targeted | Need clearer views of the pillow-like objects, especially any neck pillow on or `
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [1]. Total keyframes now: 4.
  2. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['pillow', 'throw pillow']. Each crop is 2.0x the object bbox with red boxes marking a

  **answer**: The neck pillow is on the couch, on the left side of the seat and leaning against the back cushions.
  **uncertainties**: The exact left/right wording depends on viewer perspective, but across the visible frames it is clearly on the couch sea

---

### `f8845772-d45a-48b0-84f9-659f34072c3c`  (v15=5 → chassis=1)

- **scene**: 166-scannet-scene0435_03
- **question**: Where is my room key?
- **GT answer**: on the desk

**v15** (kf 3 → 9, conf 0.62):

  1. **request_more_views** — `mode=targeted | Need additional views that clearly show any room key or keycard near the door an`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [91, 126]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['room key', 'keycard', 'key', 'desk', 'door']. Each crop is 2.0x the object bbox with

  **answer**: Your room key appears to be on the desk, near the lamp and close to the door.
  **uncertainties**: The object is somewhat blurry, so it appears to be a hotel keycard rather than a metal key, but the exact item text/deta | Additional requested views/crops are not directly citeable here because only the original visible frames are shown in th

**chassis** (kf 3 → 16, conf 0.61):

  1. **request_more_views** — `mode=targeted | Need additional views to locate a small room key/key card. Current frames show a`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [161, 266]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 11 annotated crop(s) for ['room key', 'key card', 'desk', 'chair', 'cabinet', 'bedside area', 'armchair']. Eac

  **answer**: Your room key looks to be on the armchair next to the bed, on top of the dark bag or cloth.
  **uncertainties**: The image is somewhat blurry, so the object cannot be confirmed in fine detail. | A blue card-like paper is also visible on the desk in frame 3, but the pink card on the chair is the stronger candidate 

---

## Category: object recognition

### `02a02496-904f-456a-9032-d88af21fc6fa`  (v15=5 → chassis=1)

- **scene**: 135-scannet-scene0131_00
- **question**: What leans on the wall on the right?
- **GT answer**: a poster

**v15** (kf 3 → 5, conf 0.64):

  1. **request_more_views** — `mode=targeted | Need views of the right side wall and any object leaning against it. Current key`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [28, 13]. Total keyframes now: 5.

  **answer**: A rolled-up poster is leaning on the wall on the right.
  **uncertainties**: The object is partially visible and somewhat blurry, but it appears cylindrical and poster-like rather than a flat board

**chassis** (kf 3 → 5, conf 0.67):

  1. **request_more_views** — `mode=targeted | Need clearer views of the right wall area to verify what object is leaning again`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [28, 36]. Total keyframes now: 5.

  **answer**: A chair.
  **uncertainties**: The exact posture is slightly occluded by overlapping chairs, but no other clear object appears to be leaning on the rig

---

### `5c9e876f-a4dd-4ce9-8e96-245ab1b2df94`  (v15=5 → chassis=1)

- **scene**: 186-scannet-scene0549_01
- **question**: What is the tall object on the corner?
- **GT answer**: A plant

**v15** (kf 3 → 5, conf 0.73):

  1. **request_more_views** — `mode=targeted | Need clearer views of the tall object located on a corner in the lounge area. Cu`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [72, 239]. Total keyframes now: 5.

  **answer**: It is a potted plant.
  **uncertainties**: The question is slightly ambiguous because another tall decorative object/arrangement is visible elsewhere in the room,  | The added targeted views were not displayed inline here, so the strongest directly visible support comes from frame 0.

**chassis** (kf 3 → 7, conf 0.72):

  1. **request_more_views** — `mode=targeted | Need additional views showing the room corners and any tall object/lamp. Current`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [98, 158]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['floor lamp', 'lamp', 'tall object']. Each crop is 2.0x the object bbox with red boxe

  **answer**: It is a floor lamp.
  **uncertainties**: The original three keyframes were blurry and did not clearly show the corner object; the answer relies on the additional

---

### `5d0efe22-db6d-4d6f-b9a8-3fcd5b29fd11`  (v15=5 → chassis=1)

- **scene**: 140-scannet-scene0077_00
- **question**: What is the large white object on the table next to the right wall of the room?
- **GT answer**: Printer

**v15** (kf 3 → 5, conf 0.73):

  1. **request_more_views** — `mode=targeted | Need clearer views of the table next to the right wall and the large white objec`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [80, 273]. Total keyframes now: 5.

  **answer**: It looks like a large white printer/copier, likely a multifunction printer.
  **uncertainties**: The scene is dim and slightly blurry, so the exact model/type is hard to read; it could be described as a copier or mult

**chassis** (kf 3 → 6, conf 0.67):

  1. **request_more_views** — `mode=targeted | Need clearer views of the large white object on the table near the right wall to`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [213, 288]. Total keyframes now: 5.
  2. **retrieve_object_context** — `['Binder', 'desk']`
     → {   "Binder": "A light-colored, rectangular binder lying flat on a surface. The images are dark and blurry, but it appea
  3. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['white object', 'binder', 'table']. Each crop is 2.0x the object bbox with red boxes 

  **answer**: It looks like a white binder lying on the table/desk next to the wall.
  **uncertainties**: The scene is dark, so the exact object type is somewhat ambiguous between a binder and a shallow document/storage box. | Text or branding on the object is not legible.

---

### `6a28c652-a68a-4716-91e6-578ab1224f8a`  (v15=5 → chassis=1)

- **scene**: 125-scannet-scene0426_00
- **question**: What is the pink object?
- **GT answer**: a jacket

**v15** (kf 3 → 4, conf 0.84):

  1. **request_crops** — ``
     → No scene objects found matching terms: ['pink jacket', 'coat', 'clothing']. Try different object descriptions.
  2. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['sweatpants']. Each crop is 2.0x the object bbox with red boxes marking all visible o

  **answer**: The pink object is a pink jacket or coat hanging on a stand.
  **uncertainties**: The crop request matched a garment annotation indirectly, so the exact clothing subtype (jacket vs. coat) is slightly un

**chassis** (kf 3 → 3, conf 0.52):

  1. **request_more_views** — `mode=targeted | Need additional views showing any pink-colored object in the scene. Current keyf`
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: It appears to be a pink pillow or cushion.
  **uncertainties**: The pink object is only partially visible at the far left edge of frames 1 and 2. | No additional views were available to confirm whether there are other pink objects elsewhere in the room.

---

### `6caf59bf-2792-44e6-8784-c65697c476e8`  (v15=5 → chassis=1)

- **scene**: 178-scannet-scene0685_02
- **question**: What is behind the garbage bin?
- **GT answer**: A circuit breaker box

**v15** (kf 3 → 5, conf 0.72):

  1. **request_more_views** — `mode=targeted | Need a view showing what is directly behind the large green garbage bin/trash ca`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 5.

  **answer**: A gray electrical panel is behind the garbage bin.
  **uncertainties**: The exact rear edge of the bin is partly occluded in some original frames, but the added view shows the area behind it c | A door is adjacent to the bin on one side, but the object directly behind it appears to be the wall-mounted electrical p

**chassis** (kf 3 → 3, conf 0.78):

  1. **request_more_views** — `mode=targeted | Need an additional view that clearly shows the area directly behind the trash ca`
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: A large framed picture/poster is behind the garbage bin.
  **uncertainties**: The exact content/type of the framed item is somewhat unclear due to image blur and low light; it appears to be a framed

---

### `846bca34-e9a3-4e6a-9edd-8358da559a64`  (v15=5 → chassis=1)

- **scene**: 183-scannet-scene0231_01
- **question**: What is the small object on the center of the table
- **GT answer**: A pen

**v15** (kf 3 → 9, conf 0.63):

  1. **request_crops** — ``
     → No scene objects found matching terms: ['small object on center of table', 'tabletop center object']. Try different obje
  2. **request_more_views** — `mode=targeted | Need clearer views of the desk/table surface, especially the center area, to ide`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [55, 298]. Total keyframes now: 5.
  3. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['desk', 'table']. Each crop is 2.0x the object bbox with red boxes marking all visibl

  **answer**: It looks like a pen or marker on the center of the table.
  **uncertainties**: The original wide keyframes are blurry, so the exact type is somewhat ambiguous. | The object is small and dark, making it hard to distinguish between a pen, marker, or similarly shaped desk item.

**chassis** (kf 3 → 9, conf 0.65):

  1. **request_more_views** — `mode=targeted | Need additional views centered on the table/desk surface, especially the middle `
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [55, 61]. Total keyframes now: 5.
  2. **retrieve_object_context** — `['Rubber duck', 'Toy vehicles', 'tissue box', 'desk', 'computer desk']`
     → {   "Toy vehicles": "A collection of small toy vehicles, possibly construction-themed, on a dark surface. The images are
  3. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['table', 'desk', 'small object']. Each crop is 2.0x the object bbox with red boxes ma

  **answer**: It appears to be a small rubber duck.
  **uncertainties**: The object is small and somewhat blurry in the wide view, so identification is based on the targeted tabletop views/crop

---

### `84833f1f-9c9b-4171-b948-88cfb0a3527a`  (v15=5 → chassis=1)

- **scene**: 155-scannet-scene0164_02
- **question**: How many pots and pans are there on the stove?
- **GT answer**: 3

**v15** (kf 3 → 7, conf 0.91):

  1. **request_more_views** — `mode=targeted | Need a clearer centered view of the full stove top to verify the exact number of`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['cooking pot', 'pan', 'electric stove']. Each crop is 2.0x the object bbox with red b

  **answer**: There are 3 pots and pans on the stove.

**chassis** (kf 3 → 3, conf 0.00):

  1. **submit_final** — `Visible in frames 0 and 2: one frying pan on the left burner area, one lidded co`
     → submitted; rationale='Visible in frames 0 and 2: one frying pan on the left burner area, one lidded cooking pot near cen

  **answer**: 4

---

### `9e900d39-f553-4ffe-937a-f81944c78634`  (v15=4 → chassis=1)

- **scene**: 189-scannet-scene0500_00
- **question**: What is the blue object in the ground?
- **GT answer**: A trash can

**v15** (kf 3 → 6, conf 0.74):

  1. **request_more_views** — `mode=targeted | Need additional views showing any blue object on the floor/ground. Current keyfr`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "proxy_grounded",     "hypothesis_kind": "proxy",     "hypothesis_rank": 2,     "parse
  3. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['blue object', 'floor', 'chairs']. Each crop is 2.0x the object bbox with red boxes m
  4. **request_crops** — ``
     → No scene objects found matching terms: ['blue object']. Try different object descriptions.
  5. **switch_or_expand_hypothesis** — `{'request_text': "Expand retrieval for the query 'What is the blue object in the`
     → No new_query specified for hypothesis switching.
  6. **switch_or_expand_hypothesis** — `{'request_text': 'Find additional evidence for a small blue item on the floor ne`
     → No new_query specified for hypothesis switching.
  7. **request_more_views** — `mode=explore | Explore additional unseen scene regions or neighboring views around frame 2 that`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [299, 298]. Total keyframes now: 6.
  8. **retrieve_object_context** — `['wire', 'rope', 'line', 'crayon', 'pen', 'tool', 'chairs', 'window with blinds'`
     → {   "window with blinds": "A rectangular, six-pane window with a dark, possibly black or dark brown, frame. It is fitted
  9. **request_crops** — ``
     → No scene objects found matching terms: ['chair legs', 'floor near chairs', 'small item']. Try different object descripti

  **answer**: It looks like a blue bag or bucket-like container on the floor next to the whiteboard easel.
  **uncertainties**: The exact object class is still slightly ambiguous because the visible portion is partial and blurry; it could be a soft

**chassis** (kf 3 → 5, conf 0.56):

  1. **request_more_views** — `mode=targeted | Need additional views specifically showing the ground/floor area and any blue ob`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "proxy_grounded",     "hypothesis_kind": "proxy",     "hypothesis_rank": 2,     "parse
  3. **retrieve_object_context** — `['carpet mark', 'floor', 'carpet', 'blue object']`
     → {   "carpet mark": "A dark, irregular, linear mark on the grey, low-pile carpet. It appears to be a scuff, a small tear,
  4. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['floor', 'carpet', 'blue object on ground']. Each crop is 2.0x the object bbox with r
  5. **switch_or_expand_hypothesis** — `{'request_text': "Need a different retrieval focused on any blue object on the f`
     → No new_query specified for hypothesis switching.
  6. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['carpet mark']. Each crop is 2.0x the object bbox with red boxes marking all visible 

  **answer**: It looks like the blue object on the ground is the blue-gray carpet.
  **uncertainties**: No distinct standalone blue item was found on the floor; the question may use 'object' loosely. | The only scene-level floor annotation available is a carpet mark, which is dark rather than blue.

---

### `a47013cb-309f-4494-9b32-bbc38edcc23b`  (v15=5 → chassis=1)

- **scene**: 180-scannet-scene0100_02
- **question**: What is the object in the floor near the door that exits the restroom?
- **GT answer**: A pair of slippers

**v15** (kf 3 → 3, conf 0.90):

(no tools)

  **answer**: A pair of slippers.

**chassis** (kf 3 → 6, conf 0.63):

  1. **request_more_views** — `mode=targeted | Need a lower/wider view showing the floor area directly in front of and near the`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  3. **retrieve_object_context** — `['interior door']`
     → {   "interior door": "A white, flat-panel interior door with a brass-colored, round doorknob. Several items, including a
  4. **switch_or_expand_hypothesis** — `{'request_text': 'Find views that best show the lower floor area near the interi`
     → No new_query specified for hypothesis switching.
  5. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['step stool', 'floor near interior door']. Each crop is 2.0x the object bbox with red
  6. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['step stool']. Each crop is 2.0x the object bbox with red boxes marking all visible o

  **answer**: It appears to be a small step stool.
  **uncertainties**: The object is only partially visible in the original frame and is clearer through targeted crops than the full image. | Its exact type could be described generically as a small stool/step stool rather than a fully detailed model name.

---

## Category: object state recognition

### `016fb5d2-3c6f-4aa3-8493-7768aa3762a9`  (v15=5 → chassis=1)

- **scene**: 002-scannet-scene0709_00
- **question**: Is the countertop clean?
- **GT answer**: No

**v15** (kf 3 → 7, conf 0.80):

  1. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['kitchen countertop', 'kitchen counter with sink', 'countertop']. Each crop is 2.0x t

  **answer**: No, the countertop does not appear clean overall.
  **uncertainties**: Not every inch of the countertop is visible at the same level of detail. | The image evidence shows clutter and dishes rather than obvious sticky spills or heavy grime.

**chassis** (kf 3 → 7, conf 0.72):

  1. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['kitchen countertop', 'kitchen counter with sink', 'countertop']. Each crop is 2.0x t

  **answer**: Yes, the countertop appears mostly clean.
  **uncertainties**: Some countertop areas are partially covered by appliances and kitchen items, so not every surface section is visible. | The crop request confirmed countertop-focused views were needed, but the inline evidence here is limited to the visible 

---

### `8f8966b6-7b62-4aa0-bcdf-b69312c153bf`  (v15=5 → chassis=1)

- **scene**: 145-scannet-scene0193_00
- **question**: Is the closet door fully closed?
- **GT answer**: No

**v15** (kf 3 → 8, conf 0.89):

  1. **request_more_views** — `mode=targeted | Need clearer views of the closet door to determine whether it is fully closed. P`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['closet door']. Each crop is 2.0x the object bbox with red boxes marking all visible 

  **answer**: No, the closet door is not fully closed; it appears slightly ajar.

**chassis** (kf 3 → 8, conf 0.81):

  1. **request_more_views** — `mode=targeted | Need a clearer, more centered view of the closet door to verify whether it is fu`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 2]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['closet door']. Each crop is 2.0x the object bbox with red boxes marking all visible 

  **answer**: Yes, the closet door appears to be fully closed.
  **uncertainties**: The handle and full perimeter of the closet door are not clearly visible, so a very small gap could be missed.

---

### `a3ebe222-885f-4069-b7ca-8441381333ec`  (v15=5 → chassis=1)

- **scene**: 149-scannet-scene0426_02
- **question**: Are the window blinds open or closed?
- **GT answer**: closed

**v15** (kf 3 → 4, conf 0.78):

  1. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['window blinds']. Each crop is 2.0x the object bbox with red boxes marking all visibl

  **answer**: The window blinds appear closed.
  **uncertainties**: Only part of the blinds is visible at the right edge of the frames, so the judgment is based on the visible section rath

**chassis** (kf 3 → 4, conf 0.80):

  1. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['window blinds']. Each crop is 2.0x the object bbox with red boxes marking all visibl

  **answer**: The window blinds are open.

---

### `f938ed69-466e-46b0-91e7-aee308d409db`  (v15=4 → chassis=1)

- **scene**: 102-scannet-scene0222_00
- **question**: If the fan turned on?
- **GT answer**: No

**v15** (kf 3 → 4, conf 0.63):

  1. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['box fan', 'fan']. Each crop is 2.0x the object bbox with red boxes marking all visib

  **answer**: It looks like the fan is off.
  **uncertainties**: A still image cannot prove power state with certainty; judgment is based only on visible lack of motion cues. | The crop output was summarized by the tool rather than shown inline here.

**chassis** (kf 3 → 4, conf 0.70):

  1. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['box fan']. Each crop is 2.0x the object bbox with red boxes marking all visible obje

  **answer**: Yes, the fan appears to be turned on.
  **uncertainties**: There is no direct view of a power indicator or switch setting; the conclusion is based on visual appearance of the blad

---

## Category: spatial understanding

### `093164ee-ba7e-4482-a8fa-b3bcb0b6f727`  (v15=5 → chassis=1)

- **scene**: 149-scannet-scene0426_02
- **question**: What large object would I see first when entering the room through the door?
- **GT answer**: Bed

**v15** (kf 3 → 5, conf 0.58):

  1. **request_more_views** — `mode=targeted | Need views that show the room entry door and what large object is directly visib`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [238, 70]. Total keyframes now: 5.

  **answer**: The bed.
  **uncertainties**: The exact camera pose relative to the doorway is not perfectly explicit in the provided frames. | Multiple large objects are near the door area, including a chest of drawers and tall bags/boxes, but the bed appears to 

**chassis** (kf 3 → 5, conf 0.62):

  1. **request_more_views** — `mode=targeted | Need views that better show the room entrance/doorway and what large object is d`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [12, 210]. Total keyframes now: 5.
  2. **retrieve_object_context** — `['interior door', 'bifold closet door', 'bed', 'chest of drawers', 'flatscreen t`
     → {   "flatscreen television": "A black, widescreen flatscreen television with a glossy bezel and a dark, reflective scree

  **answer**: The chest of drawers (dresser) is the large object you’d likely see first when entering through the door.
  **uncertainties**: The exact entering viewpoint is not fully shown, so 'first' is inferred from nearby doorway views rather than a perfect  | Several large objects are clustered near the door, including a golf bag and television, but the dresser appears most dir

---

### `0cf2b29c-af4c-4a50-9646-5aba1b3034ec`  (v15=5 → chassis=1)

- **scene**: 185-scannet-scene0435_00
- **question**: What is the object that is currently placed on the upper part of the desk, near the wall, on the left side?
- **GT answer**: A coffee machine

**v15** (kf 3 → 3, conf 0.88):

(no tools)

  **answer**: A single-serve coffee maker.
  **uncertainties**: The first frame is slightly blurry, but the third frame shows the upper-left desk area clearly enough to identify the ob

**chassis** (kf 3 → 3, conf 0.00):

  1. **submit_final** — `{"task_type":"qa","status":"completed","summary":"The object on the upper left p`
     → submitted; rationale='{"task_type":"qa","status":"completed","summary":"The object on the upper left part of the desk ne

  **answer**: A desk lamp.

---

### `199fc658-9cb8-431b-a087-211d797b8838`  (v15=4 → chassis=1)

- **scene**: 140-scannet-scene0077_00
- **question**: What is directly in between the large white printer and the room entrance closest to it?
- **GT answer**: paper recycling trash bin

**v15** (kf 3 → 21, conf 0.48):

  1. **request_more_views** — `mode=targeted | Need views that show the large white multifunction printer and the nearest room `
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 5.
  2. **request_more_views** — `mode=explore | Need wider or alternate views showing the doorway/room entrance closest to the l`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [214, 213]. Total keyframes now: 7.
  3. **request_more_views** — `mode=targeted | Need a view that includes both the large white multifunction printer and the nea`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [5, 2]. Total keyframes now: 9.
  4. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  5. **retrieve_object_context** — `['multifunction printer', 'desktop printer']`
     → {   "desktop printer": "A small, black desktop printer. The object is rectangular and made of dark plastic. It is mostly
  6. **request_more_views** — `mode=targeted | Need overlapping views that connect the large white multifunction printer to the`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [3, 4]. Total keyframes now: 11.
  7. **switch_or_expand_hypothesis** — `{'request_text': 'Need retrieval centered on the doorway/entrance nearest the la`
     → No new_query specified for hypothesis switching.
  8. **request_more_views** — `mode=targeted | Find views specifically showing the paper recycling bin or trash bin next to the`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [150, 146]. Total keyframes now: 13.
  9. **switch_or_expand_hypothesis** — `{'request_text': 'doorway nearest the large white multifunction printer; object `
     → No new_query specified for hypothesis switching.
  10. **retrieve_object_context** — `['paper recycling bin']`
     → No matching object context found for requested terms.
  11. **request_more_views** — `mode=explore | Need bridging views between frame 5 doorway and frames 3/4 multifunction printer`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [176, 178]. Total keyframes now: 15.
  12. **request_more_views** — `mode=targeted | Show the most direct view from the doorway toward the printer corner, or from th`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [186, 143]. Total keyframes now: 17.
  13. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['doorway', 'cabinet', 'trash bin', 'floor area']. Each crop is 2.0x the object bbox w

  **answer**: A small trash/recycling bin.
  **uncertainties**: The printer and doorway are still not visible together in one frame. | The conclusion relies on the doorway-area crop revealing the main candidate object near the entrance rather than a singl | There may also be cabinet edge/floor space in the path, but the only clear discrete object is the small bin.

**chassis** (kf 3 → 5, conf 0.66):

  1. **request_more_views** — `mode=targeted | Need a view that shows the large white multifunction printer and the nearest roo`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [80, 143]. Total keyframes now: 5.

  **answer**: A stack of cardboard boxes is directly in between them.
  **uncertainties**: The scene is dim, so small floor items are harder to distinguish. | The exact 'directly in between' interpretation is based on the visible path between the printer area and the nearest doo

---

### `1bb3eae5-36b9-4947-ae30-88de6f0ae85c`  (v15=4 → chassis=1)

- **scene**: 170-scannet-scene0378_01
- **question**: What is to the left of the blue paper on the desk with the monitor?
- **GT answer**: Pens

**v15** (kf 3 → 5, conf 0.91):

  1. **request_more_views** — `mode=targeted | Need a clearer view centered on the blue paper/file folder on the desk with the `
     → Added 1 new keyframe(s) (mode=targeted). New view IDs: [94]. Total keyframes now: 4.
  2. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['blue paper', 'file folder', 'items left of blue paper']. Each crop is 2.0x the objec

  **answer**: A small yellow note pad and some pens are to the left of the blue paper.

**chassis** (kf 3 → 6, conf 0.80):

  1. **request_more_views** — `mode=targeted | Need a clearer view of the blue paper on the desk with the monitor and the items`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [41, 132]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['blue paper', 'computer desk']. Each crop is 2.0x the object bbox with red boxes mark

  **answer**: A small stack of sticky notes is to the left of the blue paper on the desk with the monitor.

---

### `2c0e67a2-7625-40dc-ae37-00057910055a`  (v15=5 → chassis=1)

- **scene**: 046-scannet-scene0724_00
- **question**: What's on the wall above the top bunk
- **GT answer**: a wood shelf

**v15** (kf 3 → 3, conf 0.84):

(no tools)

  **answer**: A small wooden shelf is on the wall above the top bunk.

**chassis** (kf 3 → 5, conf 0.67):

  1. **request_more_views** — `mode=targeted | Need views that clearly show the wall area directly above the top bunk/upper bun`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [1, 2]. Total keyframes now: 5.

  **answer**: Nothing appears to be on the wall above the top bunk; it looks blank.
  **uncertainties**: The original three keyframes did not clearly show the wall area above the top bunk; the answer relies on the additional  | Only the visible portion of the wall can be assessed.

---

### `358927b1-dd2c-4f64-b754-05d15476a1dc`  (v15=5 → chassis=1)

- **scene**: 124-scannet-scene0131_02
- **question**: Is there a poster roll next to a white mini fridge?
- **GT answer**: yes

**v15** (kf 3 → 8, conf 0.83):

  1. **request_more_views** — `mode=targeted | Need clearer views of the white mini-fridge and any cylindrical poster roll next`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [194, 23]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['mini-fridge', 'poster roll']. Each crop is 2.0x the object bbox with red boxes marki

  **answer**: Yes, there appears to be a poster roll leaning next to the white mini-fridge.
  **uncertainties**: The object appears to be a rolled poster/document tube rather than an unambiguous labeled 'poster roll,' but it is clear

**chassis** (kf 3 → 6, conf 0.63):

  1. **request_more_views** — `mode=targeted | Need views showing any poster roll and any white mini fridge together or nearby.`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 2]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['poster roll', 'mini fridge', 'white mini fridge']. Each crop is 2.0x the object bbox

  **answer**: No, there is no visible poster roll next to a white mini fridge.
  **uncertainties**: The appliance at the lower right appears gray/silver rather than clearly white. | No clearly identifiable poster roll is visible in the cited views; a small bright object on the desk is not convincingly

---

### `50f4f7db-d50c-42f4-92fe-82387db91e0a`  (v15=4 → chassis=1)

- **scene**: 181-scannet-scene0314_00
- **question**: What is at the center of the table?
- **GT answer**: A lamp

**v15** (kf 3 → 10, conf 0.74):

  1. **request_more_views** — `mode=targeted | Need a clearer top-down or more centered view of the round conference table surf`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [289, 255]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 5 annotated crop(s) for ['conference table', 'round desk', 'workstation table']. Each crop is 2.0x the object 

  **answer**: A tall metal pole/fixture is at the center of the table.
  **uncertainties**: The exact tabletop center is partially occluded in some original wide views. | The vertical fixture may be part of a central table-mounted stand rather than a separate object.

**chassis** (kf 3 → 10, conf 0.72):

  1. **request_more_views** — `mode=targeted | Need views that clearly show the center area of the round conference/workstation`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [289, 100]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 5 annotated crop(s) for ['conference table', 'round desk', 'table', 'objects on table']. Each crop is 2.0x the

  **answer**: Computer equipment is at the center of the table, mainly computer monitors/screens with peripherals.
  **uncertainties**: The exact geometric center of the round table is partly occluded in the initial views. | I can identify computer equipment on the center area, but some individual devices are partially blocked by chairs or mon

---

### `58aba4ce-05b2-485c-a2de-6f313bbe641c`  (v15=5 → chassis=1)

- **scene**: 048-scannet-scene0745_00
- **question**: What is left of the trash can?
- **GT answer**: recycling

**v15** (kf 3 → 8, conf 0.72):

  1. **request_more_views** — `mode=targeted | Need more evidence because the question asks what is left of the trash can, and `
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['trash can', 'wastebasket', 'recycling bin']. Each crop is 2.0x the object bbox with 

  **answer**: The blue recycling bin is to the left of the trash can.
  **uncertainties**: The scene contains both a wastebasket and a recycling bin, so 'trash can' could be interpreted broadly; I treated the be

**chassis** (kf 3 → 3, conf 0.91):

(no tools)

  **answer**: A wooden chair is left of the trash can.

---

### `b0370ddd-3bf8-4a77-ab83-54a727d189a7`  (v15=5 → chassis=1)

- **scene**: 115-scannet-scene0046_02
- **question**: Where is my cellphone?
- **GT answer**: on the desk, to the left of the laptop

**v15** (kf 3 → 9, conf 0.76):

  1. **request_more_views** — `mode=targeted | Need additional views showing any cellphone/mobile phone/smartphone. Current key`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  3. **retrieve_object_context** — `['cellphone', 'mobile phone', 'smartphone', 'telephone handset', 'tablet compute`
     → {   "tablet computer": "A white tablet computer, possibly an older model iPad, with a dark, reflective screen that is cu
  4. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['cellphone', 'mobile phone', 'smartphone', 'small object on desk', 'object on chair',
  5. **switch_or_expand_hypothesis** — `{'request_text': 'Expand retrieval for cellphone by separating likely confusions`
     → No new_query specified for hypothesis switching.
  6. **request_crops** — ``
     → Generated 4 annotated crop(s) for ['tablet computer', 'telephone handset']. Each crop is 2.0x the object bbox with red b

  **answer**: Your cellphone is on the desk, just to the left of the laptop.
  **uncertainties**: The image is slightly blurry, but the black rectangular object on the desk is the strongest cellphone candidate.

**chassis** (kf 3 → 16, conf 0.63):

  1. **request_more_views** — `mode=targeted | Need additional views specifically showing any cellphone/mobile phone/smartphone`
     → No additional views available. Current keyframes already provide best coverage.
  2. **retrieve_object_context** — `['telephone handset', 'tablet computer', 'cellphone', 'phone']`
     → {   "tablet computer": "A white tablet computer, possibly an older model iPad, with a dark, reflective screen that is cu
  3. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  4. **request_crops** — ``
     → Generated 11 annotated crop(s) for ['tablet computer', 'telephone handset', 'bag', 'chair', 'desk']. Each crop is 2.0x t
  5. **switch_or_expand_hypothesis** — `{'request_text': "Expand retrieval for alternate interpretations of 'cellphone' `
     → No new_query specified for hypothesis switching.
  6. **request_more_views** — `mode=explore | Need exploratory views of unseen room regions and alternate angles around the de`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [102, 72]. Total keyframes now: 16.

  **answer**: I can’t see your cellphone. The visible device by the chair/window is a white tablet lying on top of red clothing/bags, and the other phone on the desk is a landline handset.
  **uncertainties**: A cellphone could still be hidden inside a bag or outside the visible areas. | The answer is limited to what is visible in the provided frames and crops.

---

### `df42c730-83e0-4f81-b6a8-7023356efea4`  (v15=5 → chassis=1)

- **scene**: 047-scannet-scene0747_00
- **question**: How many dining room chairs are not facing the table?
- **GT answer**: one

**v15** (kf 3 → 7, conf 0.61):

  1. **request_more_views** — `mode=targeted | Need clearer and possibly alternate views of the dining table and all surroundin`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [57, 140]. Total keyframes now: 5.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  3. **retrieve_object_context** — `['dining chair']`
     → {   "Queen Anne dining chair": "A traditional Queen Anne style dining chair with a dark, polished wood frame. It feature
  4. **request_more_views** — `mode=explore | Need a wider or alternate overview of the dining room to confirm the full set of`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [20, 19]. Total keyframes now: 7.

  **answer**: 1
  **uncertainties**: The original three keyframes are blurry, and not every chair is fully visible in those views. | Some far-side chair orientations are partially occluded by the table, so the count is based on the clearly visible out-o

**chassis** (kf 3 → 5, conf 0.71):

  1. **request_more_views** — `mode=targeted | Need additional views of the full dining table and all surrounding dining chairs`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [0, 1]. Total keyframes now: 5.

  **answer**: 0
  **uncertainties**: Some chairs are partially occluded or blurred in the original keyframes, but the added views provide enough coverage to 

---

### `ef569a31-9c3b-4361-aed3-be7e9467e5c8`  (v15=4 → chassis=1)

- **scene**: 138-scannet-scene0500_01
- **question**: What is on the floor in front of the blackboard?
- **GT answer**: the wire

**v15** (kf 3 → 3, conf 0.73):

  1. **request_more_views** — `mode=targeted | Need a clearer lower-angle view showing the floor area directly in front of the `
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: There are several wooden chairs on the floor in front of the blackboard, and a cable/cord is also visible there.
  **uncertainties**: No additional views were available, so the exact extent of all floor objects in front of the blackboard may be partially | Some of the floor is obscured by furniture, so smaller objects could be hidden.

**chassis** (kf 3 → 6, conf 0.46):

  1. **request_more_views** — `mode=targeted | Need views that clearly show the floor area directly in front of the blackboard/`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  3. **retrieve_object_context** — `['chalkboard', 'easel whiteboard', 'portable whiteboard']`
     → {   "chalkboard": "A large, two-tiered chalkboard mounted on a wall. The dark gray surface is covered in white chalk wri
  4. **switch_or_expand_hypothesis** — `{'request_text': 'Retrieve alternative views for board-related objects, especial`
     → No new_query specified for hypothesis switching.
  5. **switch_or_expand_hypothesis** — `{'request_text': 'Find views of the chalkboard and the floor area directly below`
     → No new_query specified for hypothesis switching.
  6. **request_more_views** — `mode=explore | Explore neighboring views around the chalkboard keyframe to reveal lower scene r`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [205, 204]. Total keyframes now: 5.
  7. **request_more_views** — `mode=targeted | Need any view that shows the chalkboard together with the floor area directly be`
     → No additional views available. Current keyframes already provide best coverage.
  8. **request_crops** — ``
     → Generated 1 annotated crop(s) for ['chalkboard', 'wooden wainscoting', 'floor']. Each crop is 2.0x the object bbox with 

  **answer**: A wooden chair is on the floor in front of the blackboard.
  **uncertainties**: Only a small portion of the floor area is visible; other objects on the floor may be out of frame. | The chair is partially visible at the bottom edge rather than fully shown.

---

### `f6ba1915-3f0d-4d2f-95e6-ce18f19093ed`  (v15=5 → chassis=1)

- **scene**: 142-scannet-scene0653_01
- **question**: Is the computer monitor on the middle work desk turned on or off?
- **GT answer**: It's turned off

**v15** (kf 3 → 11, conf 0.85):

  1. **request_crops** — ``
     → Generated 8 annotated crop(s) for ['computer monitor', 'dual monitor setup', 'office desk']. Each crop is 2.0x the objec

  **answer**: Off.
  **uncertainties**: The question says 'the computer monitor' but the middle work desk has two monitors; both appear off in the available evi | No power indicator light is clearly visible, so the judgment is based on the screens appearing blank/dark.

**chassis** (kf 3 → 8, conf 0.79):

  1. **request_more_views** — `mode=targeted | Need additional views of the middle work desk and its computer monitor to verify`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [64, 79]. Total keyframes now: 5.
  2. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['computer monitor']. Each crop is 2.0x the object bbox with red boxes marking all vis

  **answer**: It is turned on.
  **uncertainties**: The provided initial views show multiple desks, so identifying the 'middle work desk' relies on the added targeted views | The exact content on the screen is not fully legible, but the screen is visibly illuminated rather than black.

---

## Category: world knowledge

### `0b727ffa-414d-4495-8c75-8e2688c8204f`  (v15=4 → chassis=1)

- **scene**: 170-scannet-scene0378_01
- **question**: I need to get rid of the stack of papers on my desk. Where should I put it?
- **GT answer**: Place it in the recycling bin to the right of the desk with the monitor

**v15** (kf 3 → 3, conf 0.88):

  1. **retrieve_object_context** — `['recycling bin']`
     → {   "recycling bin": "A small, blue plastic recycling bin with a white recycling symbol (three chasing arrows) printed o

  **answer**: Put the papers in the blue recycling bin next to/under the desk.

**chassis** (kf 3 → 14, conf 0.72):

  1. **request_more_views** — `mode=explore | Need more views of the office around the desks to identify a suitable place to p`
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [209, 208]. Total keyframes now: 5.
  2. **retrieve_object_context** — `['computer desk', 'desk pedestal', 'small desk']`
     → {   "computer desk": "A two-tiered computer desk with light gray laminate surfaces and a dark gray metal frame. The lowe
  3. **request_crops** — ``
     → Generated 9 annotated crop(s) for ['papers', 'desk', 'filing cabinet', 'printer', 'trash can']. Each crop is 2.0x the ob

  **answer**: Put the stack on the larger brown desk, next to the existing pile of papers. It has visible open surface space and already appears to be used for paper storage, unlike the crowded computer desk or the blue bin.
  **uncertainties**: The drawers in the desk pedestal are closed, so I cannot verify whether inside-drawer storage is available. | No dedicated paper tray or shelf is clearly visible in the provided views.

---

### `aee9e1df-66b1-45a4-b1ab-e52248239e93`  (v15=5 → chassis=1)

- **scene**: 162-scannet-scene0535_00
- **question**: How many people are intended to use this space for work?
- **GT answer**: 2

**v15** (kf 3 → 5, conf 0.66):

  1. **request_more_views** — `mode=explore | Need wider scene coverage to estimate how many people this space is intended to `
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [254, 253]. Total keyframes now: 5.

  **answer**: 2 people
  **uncertainties**: The full room is not exhaustively shown, so there could be additional work seating out of frame. | One visible tall stool may be multipurpose rather than a primary desk chair.

**chassis** (kf 3 → 5, conf 0.77):

  1. **request_more_views** — `mode=targeted | Need additional views showing the full workspace and all seating intended for wo`
     → Added 2 new keyframe(s) (mode=targeted). New view IDs: [1, 240]. Total keyframes now: 5.

  **answer**: 3
  **uncertainties**: The scene is somewhat cluttered and narrow, so a hidden extra seat cannot be ruled out completely. | The answer infers intended work capacity from visible seating at work surfaces.

---

### `e1b47f95-7ae0-417f-8f3e-d096feae96f6`  (v15=5 → chassis=1)

- **scene**: 154-scannet-scene0193_01
- **question**: Is the drawer (under the table) open?
- **GT answer**: no

**v15** (kf 3 → 6, conf 0.86):

  1. **request_crops** — ``
     → Generated 3 annotated crop(s) for ['drawer', 'computer desk']. Each crop is 2.0x the object bbox with red boxes marking 

  **answer**: No, the drawer under the table/desk appears to be closed.
  **uncertainties**: The images are somewhat blurry, so very small partial openings could be hard to detect. | The question says 'the drawer' but the desk has multiple drawers; none of the visible under-desk drawers appears open.

**chassis** (kf 3 → 5, conf 0.87):

  1. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['drawer', 'desk drawer unit']. Each crop is 2.0x the object bbox with red boxes marki

  **answer**: Yes, the drawer under the table appears to be open.
  **uncertainties**: The phrase 'drawer under the table' could refer either to the long shallow desk drawer or one of the drawers in the pede

---

### `e621e982-d424-431a-acb8-4afa7a032d24`  (v15=5 → chassis=1)

- **scene**: 014-scannet-scene0714_00
- **question**: Can I use the stairs to go up?
- **GT answer**: No

**v15** (kf 3 → 7, conf 0.66):

  1. **request_more_views** — `mode=targeted | Need clearer views showing whether the staircase is accessible for going up from`
     → No additional views available. Current keyframes already provide best coverage.
  2. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "direct_grounded",     "hypothesis_kind": "direct",     "hypothesis_rank": 1,     "par
  3. **retrieve_object_context** — `['staircase', 'step']`
     → {   "staircase": "A staircase with a wooden handrail and white vertical balusters. The handrail has a light brown, natur
  4. **switch_or_expand_hypothesis** — `{'request_text': 'Find alternate views or context focused on staircase usability`
     → No new_query specified for hypothesis switching.
  5. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['staircase', 'step']. Each crop is 2.0x the object bbox with red boxes marking all vi
  6. **request_crops** — ``
     → Generated 2 annotated crop(s) for ['staircase', 'step']. Each crop is 2.0x the object bbox with red boxes marking all vi

  **answer**: No — the stairs shown here appear to go down, not up.
  **uncertainties**: The actual stair treads are only partially visible through the railing, so the exact run of the stairs is not fully show | No alternate full-scene views were available beyond the current retrieval.

**chassis** (kf 3 → 3, conf 0.64):

  1. **request_more_views** — `mode=targeted | Need views that clearly show the staircase path and whether it is accessible for`
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: Yes, it looks like you can use the stairs to go up.
  **uncertainties**: The actual upper run of the stairs is not fully visible; the judgment is based on the visible stairwell opening, railing | No additional views were available to confirm the full staircase path.

---

### `fd5f875a-c63c-4042-af45-32aa5a5bc450`  (v15=5 → chassis=1)

- **scene**: 156-scannet-scene0461_00
- **question**: Can I mop the floor of this room if it is dirty?
- **GT answer**: no

**v15** (kf 3 → 3, conf 0.84):

  1. **request_more_views** — `mode=targeted | Need more views of the floor surface to determine if this room has carpet/rug or`
     → No additional views available. Current keyframes already provide best coverage.

  **answer**: No. The visible floor looks carpeted/patterned rather than bare hard flooring, so if it gets dirty you should not mop it like a hard floor.
  **uncertainties**: The images do not show the entire room, but the visible floor area is carpeted/patterned rug-like rather than exposed ti | The question is hypothetical ('if it is dirty'); dirt itself is not directly visible.

**chassis** (kf 3 → 5, conf 0.72):

  1. **request_more_views** — `mode=explore | Need wider views of the room and more of the floor surface to judge whether the `
     → Added 2 new keyframe(s) (mode=explore). New view IDs: [207, 206]. Total keyframes now: 5.
  2. **request_crops** — ``
     → No scene objects found matching terms: ['floor']. Try different object descriptions.
  3. **inspect_stage1_metadata** — `{}`
     → {   "hypothesis": {     "status": "proxy_grounded",     "hypothesis_kind": "proxy",     "hypothesis_rank": 2,     "parse
  4. **retrieve_object_context** — `['scan artifact']`
     → {   "scan artifact": "A long, thin, rectangular shape on the floor with a glowing, bluish-white appearance. The object i

  **answer**: Yes. The room appears to have a hard floor that could be mopped if it gets dirty, although you would need to mop around the two chairs and the small table.
  **uncertainties**: The new wider views are not shown inline here, so exact floor extent is not fully verifiable from the displayed images a | I cannot confirm actual dirtiness; the answer is conditional on the floor being dirty.

---
