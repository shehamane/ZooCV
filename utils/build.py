import re
import ast

from torch import nn
import numpy as np

from layer.general import Conv, FC, AvgPool, Softmax, MaxPool, ConvBN


class ModelBuilder:
    VAR_REGEX = re.compile(r'\$\{(\w+)\}')

    PARAMETRIZED_LAYERS = ('Conv', 'ConvBN', 'FC', 'MaxPool', 'AvgPool', 'Dropout')
    SIMPLE_LAYERS = ('Flatten', 'Softmax')

    def __init__(self, config: dict):
        self.backbone = config['backbone']
        self.activation = config.get('activation', None)
        self.vars = config['vars']
        self.input_shape = (config['input']['w'], config['input']['h'], config['input']['ch'])

    def _replace_vars(self, seq):
        for var_name in self.vars:
            seq = seq.replace('$' + var_name, str(self.vars[var_name]))
        return seq

    def _parse_exp(self, exp):
        if not isinstance(exp, str):
            return exp
        exp = self._replace_vars(exp)
        parse = ast.parse(exp, mode='eval')
        return eval(compile(parse, '<string>', 'eval'))

    def _calc_conv_out_shape(self, w, h, n, k, s=1, p=0):
        return int((w - k + 2 * p) / s + 1), int((h - k + 2 * p) / s + 1), n

    def _build(self, layers: list, input_shape, idx=None, from_idx=-1) -> (nn.Module, tuple):
        build_layers = []
        cur_shape = input_shape

        for layer_idx, layer_desc in enumerate(layers):
            if not isinstance(layer_desc, list):
                raise "Layer should be a list"

            from_, quantity, type_ = layer_desc[:3]

            for _ in range(quantity):
                layer = None
                if type_ in self.PARAMETRIZED_LAYERS:
                    if not isinstance(layer_desc[3], list):
                        params = [self._parse_exp(layer_desc[3]), ]
                    else:
                        params = [self._parse_exp(exp) for exp in layer_desc[3]]

                    if type_ == 'Conv':
                        layer = Conv(cur_shape[-1], *params, idx=layer_idx, from_idx=from_)
                        cur_shape = self._calc_conv_out_shape(*cur_shape[:2], *params)
                    elif type_ == 'ConvBN':
                        layer = ConvBN(cur_shape[-1], *params, idx=layer_idx, from_idx=from_)
                        cur_shape = self._calc_conv_out_shape(*cur_shape[:2], *params)
                    elif type_ == 'FC':
                        if len(cur_shape) > 1:
                            flatten = nn.Flatten()
                            cur_shape = (np.prod(cur_shape),)
                            build_layers.append(flatten)
                        layer = FC(cur_shape[-1], *params, idx=layer_idx, from_idx=from_)
                        cur_shape = (params[0],)
                    elif type_ == 'MaxPool':
                        layer = MaxPool(*params, idx=layer_idx, from_idx=from_)
                        cur_shape = self._calc_conv_out_shape(*cur_shape[:3], *params)
                    elif type_ == 'AvgPool':
                        layer = AvgPool(*params, idx=layer_idx, from_idx=from_)
                        cur_shape = self._calc_conv_out_shape(*cur_shape[:3], *params)
                    elif type_ == 'Dropout':
                        layer = nn.Dropout(*params)
                elif type_ in self.SIMPLE_LAYERS:
                    if type_ == 'Softmax':
                        layer = Softmax(dim=1, idx=layer_idx, from_idx=from_)
                elif type_ == 'Block':
                    block_layers = layer_desc[3]
                    layer, cur_shape = self._build(block_layers, cur_shape, layer_idx, from_)
                else:
                    raise Exception(f'Unknown layer: {type_}')
                build_layers.append(layer)

        return nn.Sequential(*build_layers), cur_shape

    def build(self):
        model, out_shape = self._build(self.backbone, self.input_shape)
        return model
