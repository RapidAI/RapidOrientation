# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import random

import cv2
import numpy as np


class Preprocess:
    def __init__(self, batch_size: int = 3):
        self.resize_img = ResizeImage(resize_short=256)
        self.crop_img = CropImage(size=224)
        self.rand_crop = RandCropImageV2(size=224)
        self.normal_img = NormalizeImage()
        self.cvt_channel = ToCHWImage()

        self.batch_size = batch_size

    def __call__(self, img: np.ndarray):
        ori_img = self.resize_img(img)

        norm_img_batch = []
        for _ in range(self.batch_size):
            img = self.crop_img(copy.deepcopy(ori_img))
            img = self.normal_img(img)
            img = self.cvt_channel(img)
            img = img[None, ...]
            norm_img_batch.append(img)
        norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
        return norm_img_batch


class ResizeImage:
    def __init__(self, size=None, resize_short=None):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w, self.h = None, None
        elif size is not None:
            self.resize_short = None
            self.w = size if isinstance(size, int) else size[0]
            self.h = size if isinstance(size, int) else size[1]
        else:
            raise ValueError(
                "invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None"
            )

    def __call__(self, img: np.ndarray):
        img_h, img_w = img.shape[:2]

        w, h = self.w, self.h
        if self.resize_short:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))

        return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)


class CropImage:
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]

        if img_h < h or img_w < w:
            raise ValueError(
                f"The size({h}, {w}) of CropImage must be greater than "
                f"size({img_h}, {img_w}) of image."
            )

        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class RandCropImageV2:
    """RandCropImageV2 is different from RandCropImage,
        it will Select a cutting position randomly in a uniform distribution way,
        and cut according to the given size without resize at last.

    Modified from https://github.com/PaddlePaddle/PaddleClas/blob/177e4be74639c0960efeae2c5166d3226c9a02eb/ppcls/data/preprocess/ops/operators.py#L448C1-L479C62

    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)  # (h, w)

    def __call__(self, img: np.ndarray):
        img_h, img_w = img.shape[0], img.shape[1]

        tw, th = self.size
        if img_h + 1 < th or img_w + 1 < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger then input image size {(img_h, img_w)}"
            )

        if img_w == tw and img_h == th:
            return img

        top = random.randint(0, img_h - th + 1)
        left = random.randint(0, img_w - tw + 1)
        return img[top : top + th, left : left + tw, :]


class NormalizeImage:
    def __init__(self):
        self.scale = np.float32(1.0 / 255.0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        shape = 1, 1, 3
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img = (img * self.scale - self.mean) / self.std
        return img.astype(np.float32)


class ToCHWImage:
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        return img.transpose((2, 0, 1))
