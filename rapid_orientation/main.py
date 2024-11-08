# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .utils.infer_engine import OrtInferSession
from .utils.load_image import LoadImage
from .utils.preprocess import Preprocess
from .utils.utils import read_yaml

root_dir = Path(__file__).resolve().parent
DEFAULT_PATH = root_dir / "models" / "rapid_orientation.onnx"
DEFAULT_CFG = root_dir / "config.yaml"


class RapidOrientation:
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_PATH,
        cfg_path: Union[str, Path] = DEFAULT_CFG,
    ):
        config = read_yaml(cfg_path)
        config["model_path"] = model_path

        self.session = OrtInferSession(config)
        self.labels = self.session.get_character_list()

        self.preprocess = Preprocess()
        self.load_img = LoadImage()

    def __call__(self, img_content: Union[str, np.ndarray, bytes, Path]):
        image = self.load_img(img_content)

        s = time.perf_counter()

        image = self.preprocess(image)
        image = image[None, ...]

        pred_output = self.session(image)[0]

        pred_output = pred_output.squeeze()
        pred_idx = np.argmax(pred_output)
        pred_txt = self.labels[pred_idx]

        elapse = time.perf_counter() - s
        return pred_txt, elapse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-img", "--img_path", type=str, required=True, help="Path to image for layout."
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=str(root_dir / "models" / "rapid_orientation.onnx"),
        help="The model path used for inference.",
    )
    args = parser.parse_args()

    orientation_engine = RapidOrientation(args.model_path)

    img = cv2.imread(args.img_path)
    orientaion_result, _ = orientation_engine(img)
    print(orientaion_result)


if __name__ == "__main__":
    main()
