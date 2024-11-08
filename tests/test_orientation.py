# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from rapid_orientation import RapidOrientation

test_file_dir = cur_dir / "test_files"
text_orientation = RapidOrientation()

img_path = test_file_dir / "img_rot0_demo.jpg"
img_tmp = cv2.imread(str(img_path))


@pytest.mark.parametrize(
    "img_path, result",
    [
        (f"{test_file_dir}/img_rot0_demo.jpg", "0"),
        (f"{test_file_dir}/img_rot180_demo.jpg", "180"),
    ],
)
def test_img(img_path, result):
    img = cv2.imread(str(img_path))
    pred_result, _ = text_orientation(img)

    assert pred_result == result


@pytest.mark.parametrize(
    "img_content", [img_path, str(img_path), open(img_path, "rb").read(), img_tmp]
)
def test_multi_input(img_content):
    pred_result, _ = text_orientation(img_content)

    assert pred_result == "0"
