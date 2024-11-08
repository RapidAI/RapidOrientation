# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapid_orientation import RapidOrientation


def scale_resize(img, resize_value=(280, 32)):
    """
    @params:
    img: ndarray
    resize_value: (width, height)
    """
    # padding
    ratio = resize_value[0] / resize_value[1]  # w / h
    h, w = img.shape[:2]
    if w / h < ratio:
        # 补宽
        t = int(h * ratio)
        w_padding = (t - w) // 2
        img = cv2.copyMakeBorder(
            img, 0, 0, w_padding, w_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        # 补高  (top, bottom, left, right)
        t = int(w / ratio)
        h_padding = (t - h) // 2
        color = tuple([int(i) for i in img[0, 0, :]])
        img = cv2.copyMakeBorder(
            img, h_padding, h_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    img = cv2.resize(img, resize_value, interpolation=cv2.INTER_LANCZOS4)
    return img


orientation_engine = RapidOrientation()
img = cv2.imread("tests/test_files/1.png")
cls_result, _ = orientation_engine(img)
print(cls_result)
