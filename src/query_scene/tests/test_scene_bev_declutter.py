from query_scene.scene_bev_builder import _stack_overlapping_anchors


def test_stack_overlapping_anchors_offsets_overlapping_labels():
    anchors = [(100, 100), (100, 100), (200, 100)]
    text_w, text_h = 60, 18
    out = _stack_overlapping_anchors(anchors, ref_w=text_w, ref_h=text_h)
    assert out[0] == (100, 100)
    assert out[1] == (100, 100 + text_h + 6)
    assert out[2] == (200, 100)


def test_stack_overlapping_anchors_stacks_three():
    anchors = [(100, 100), (105, 100), (110, 100)]
    text_w, text_h = 60, 18
    out = _stack_overlapping_anchors(anchors, ref_w=text_w, ref_h=text_h)
    assert out[1] == (105, 100 + text_h + 6)
    assert out[2] == (110, 100 + 2 * (text_h + 6))
