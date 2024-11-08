# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapid_orientation import RapidOrientation

orientation_engine = RapidOrientation()
img = cv2.imread("tests/test_files/img_rot0_demo.jpg")
cls_result, elapse = orientation_engine(img)
print(cls_result)
print(elapse)
