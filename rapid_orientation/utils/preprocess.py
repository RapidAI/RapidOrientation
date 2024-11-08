# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
import numpy as np


class Preprocess:
    def __init__(self):
        self.resize_img = ResizeImage(resize_short=256)
        self.crop_img = CropImage(size=224)
        self.normal_img = NormalizeImage()
        self.cvt_channel = ToCHWImage()

    def __call__(self, img: np.ndarray):
        img = self.resize_img(img)
        img = self.crop_img(img)
        img = self.normal_img(img)
        img = self.cvt_channel(img)
        return img


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
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)


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


class NormalizeImage:
    def __init__(
        self,
    ):
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
