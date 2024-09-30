import re
from typing import Tuple, List

import yaml
from torch import nn
import numpy as np

from layer.general import Conv, FC, AvgPool, Softmax, MaxPool, ConvBN, Dropout


class ModelBuilder:
    names2layers_nonparam = {
        'Softmax': Softmax,
        # 'Flatten': Flatten,
    }

    names2layers_param = {
        'Conv': Conv,
        'ConvBN': ConvBN,
        'FC': FC,
        'AvgPool': AvgPool,
        'MaxPool': MaxPool,
        'Dropout': Dropout,
    }

    names2activations = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'LReLU': nn.LeakyReLU,
    }

    def __init__(self, config_path: str):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.vars = config.get('vars', [])
        self.act = config.get('act', 'ReLU')

        self.backbone = self._replace_vars(config.get('backbone', None), self.vars)
        self.neck = self._replace_vars(config.get('neck', None), self.vars)
        self.head = self._replace_vars(config.get('head', None), self.vars)
        self.input_shape = self._replace_vars(config.get('input_shape', None), self.vars)

        if self.act not in self.names2activations.keys():
            raise ValueError(f'Unknown activation {self.act}')
        self.act = self.names2activations[self.act]

    def _eval(self, exp: str, vars):
        return eval(exp, {'__builtins__': {}}, vars)

    def _replace_vars(self, obj: str, variables):
        if isinstance(obj, str):
            matches = re.findall(r'\$(\w+)', obj)
            for match in matches:
                if match in variables:
                    obj = obj.replace(f'${match}', str(variables[match]))
            if re.search(r'[+\-*/]', obj):
                obj = self._eval(obj, variables)
        elif isinstance(obj, list):
            return [self._replace_vars(item, variables) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._replace_vars(val, variables) for key, val in obj.items()}

        return obj

    def _build(self, layers: list, input_shape: List[int], idx=None, from_idx=-1) -> Tuple[nn.Module, List[int]]:
        modules = nn.Sequential()
        current_shape = input_shape

        for i, layer in enumerate(layers):
            from_layer, quantity, layer_type, params = layer

            if layer_type == 'Block':
                for _ in range(quantity):
                    block, current_shape = self._build(params, current_shape, idx=i, from_idx=from_layer)
                    modules.append(block)
            else:
                layer_class = self.names2layers_param.get(layer_type, None)

                if layer_class is None:
                    raise ValueError(f"Unknown layer type {layer_type}")

                for _ in range(quantity):
                    if layer_type in ('Conv', 'ConvBN'):
                        if len(current_shape) != 3:
                            raise ValueError('Conv layer operates with only 3-dimensional inputs')
                        in_ch = current_shape[-1]
                        current_shape = self._update_shape_conv(current_shape, *params)
                        params = [in_ch,] + params
                    elif layer_type == 'MaxPool' or layer_type == 'AvgPool':
                        if len(current_shape) != 3:
                            raise ValueError('Pool layers operates with only 3-dimensional inputs')
                        current_shape = self._update_shape_pool(current_shape, *params)
                    elif layer_type == 'FC':
                        if len(current_shape) != 1:
                            flatten_module = nn.Flatten()
                            modules.add_module(f'Flatten_{i}', flatten_module)
                        in_ch = current_shape[-1]
                        params = [in_ch,] + params
                        current_shape = (params[0],)

                    layer_module = layer_class(*params, idx=i, from_idx=from_layer, act=self.act)
                    modules.append(layer_module)

        return modules, current_shape

    @staticmethod
    def _update_shape_conv(input_shape, out_channels, kernel_size, stride=1, padding=0):
        h, w, ch = input_shape
        new_h = (h + 2 * padding - kernel_size) // stride + 1
        new_w = (w + 2 * padding - kernel_size) // stride + 1
        return new_h, new_w, out_channels

    @staticmethod
    def _update_shape_pool(input_shape, kernel_size, stride=1):
        h, w, ch = input_shape
        new_h = (h - kernel_size) // stride + 1
        new_w = (w - kernel_size) // stride + 1
        return new_h, new_w, ch

    def build(self):
        backbone_model, backbone_out_shape = self._build(
            self.backbone,
            self.input_shape
        )

        neck_model, neck_out_shape = self._build(
            self.neck,
            backbone_out_shape
        ) if self.neck else (None, backbone_out_shape)

        head_model, head_out_shape = self._build(
            self.head,
            neck_out_shape
        ) if self.head else (None, neck_out_shape)

        model = nn.Sequential()
        model.add_module('backbone', backbone_model)
        if neck_model:
            model.add_module('neck', neck_model)
        if head_model:
            model.add_module('head', head_model)

        return model
