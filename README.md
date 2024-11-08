
> ## 📣 原有RapidStructure仓库已经移到[RapidDoc](https://github.com/RapidAI/RapidDoc)下了，RapidStructure也将以RapidDoc方式重生

<div align="center">
  <div align="center">
    <h1><b>Rapid Orientation</b></h1>
  </div>

<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://pypi.org/project/rapid-orientation/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-orientation"></a>
<a href="https://pepy.tech/project/rapid-orientation"><img src="https://static.pepy.tech/personalized-badge/rapid-orientation?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://github.com/RapidAI/RapidOrientation/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOrientation?color=ccf"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

### 简介和说明

该部分主要是做含文字图像方向分类模型。模型来源：[PaddleClas 含文字图像方向分类模型](https://github.com/PaddlePaddle/PaddleClas/blob/177e4be74639c0960efeae2c5166d3226c9a02eb/docs/zh_CN/models/PULC/PULC_text_image_orientation.md)

| 模型类型  |        模型名称         | 模型大小 |                           支持种类                           |
|:---:|:---:|:---:|:---:|
|   四方向分类   |   `rapid_orientation.onnx`   |  6.5M | `0 90 180 270`|

### 安装

由于模型较小，已经将分类模型(`rapid_orientation.onnx`)打包进了whl包内：

  ```bash
  pip install rapid-orientation
  ```

### 脚本运行

```python
import cv2

from rapid_orientation import RapidOrientation

orientation_engine = RapidOrientation()
img = cv2.imread("tests/test_files/img_rot180_demo.jpg")
cls_result, _ = orientation_engine(img)
print(cls_result)
```

### 终端运行

用法:

```bash
$ rapid_orientation -h
usage: rapid_orientation [-h] -img IMG_PATH [-m MODEL_PATH]

optional arguments:
-h, --help            show this help message and exit
-img IMG_PATH, --img_path IMG_PATH
                      Path to image for layout.
-m MODEL_PATH, --model_path MODEL_PATH
                      The model path used for inference.
```

示例:

```bash
rapid_orientation -img test_images/layout.png
```

结果

```python
# 返回结果为str类型，有四类：0 | 90 | 180 | 270
```
