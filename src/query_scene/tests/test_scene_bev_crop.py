import numpy as np

from query_scene.scene_bev_builder import _crop_to_non_white


def test_crop_keeps_eight_pixel_margin():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    img[100:110, 200:220] = (50, 50, 50)
    cropped, (ox, oy) = _crop_to_non_white(img, margin=8)
    assert cropped.shape[0] == 26
    assert cropped.shape[1] == 36
    assert ox == 192
    assert oy == 92


def test_crop_all_white_returns_original():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cropped, (ox, oy) = _crop_to_non_white(img, margin=8)
    assert cropped.shape == img.shape
    assert (ox, oy) == (0, 0)
