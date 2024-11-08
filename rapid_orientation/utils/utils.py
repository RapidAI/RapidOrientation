# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import yaml


def read_yaml(yaml_path):
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
