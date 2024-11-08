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

        self.preprocess = Preprocess(batch_size=3)
        self.load_img = LoadImage()

    def __call__(self, img_content: Union[str, np.ndarray, bytes, Path]):
        image = self.load_img(img_content)

        s = time.perf_counter()

        image = self.preprocess(image)

        pred_output = self.session(image)[0]
        pred_idxs = list(np.argmax(pred_output, axis=1))
        final_idx = max(set(pred_idxs), key=pred_idxs.count)
        pred_txt = self.labels[final_idx]

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
