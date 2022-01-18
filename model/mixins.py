# -*- encoding: utf-8 -*-
'''
@File    :   mixins.py
@Time    :   2021/10/01 17:52:40
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
from nets.embeddings import ColumnParallelLinear, RowParallelLinear
from mpu.func_utils import unscaled_init_method
from .base_model import BaseMixin
from .cached_autoregressive_model import CachedAutoregressiveMixin
from .base_model import BaseModel, BaseMixin, non_conflict
from nets.attentions import standard_attention

class PrefixTuningMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size_per_attention_head, num_attention_heads, prefix_len):
        super().__init__()
        self.prefix = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(2, num_attention_heads, prefix_len, hidden_size_per_attention_head)*0.01)
            for layer_id in range(num_layers)
        ])
        self.prefix_len = prefix_len

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
        prefix_k, prefix_v = self.prefix[kw_args['layer_id']]

        b, nh, seq_len, hidden_size = k.shape
        prefix_k = prefix_k.unsqueeze(0).expand(b, nh, -1, hidden_size)
        prefix_v = prefix_v.unsqueeze(0).expand(b, nh, -1, hidden_size)

        k = torch.cat((k, prefix_k), dim=2)
        v = torch.cat((v, prefix_v), dim=2)
        if mask.numel() > 1:
            mask_prefixed = torch.ones(self.prefix_len, device=mask.device, dtype=mask.dtype)
            mask_prefixed = mask_prefixed.expand(*(mask.size()[:-1]), -1)
            mask = torch.cat((mask, mask_prefixed), dim=-1)
        return old_impl(q, k, v, mask, dropout_fn, **kw_args)

PTuningV2Mixin = PrefixTuningMixin

class MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005):
        super().__init__()
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for sz in output_sizes:
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            self.layers.append(this_layer)

    def final_forward(self, logits, **kw_args):
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class PositionEmbeddingMixin(BaseMixin):
    def __init__(self, additional_sequence_length, hidden_size, 
                init_method_std=0.02, reinit_slice=slice(-1024, None)
        ):
        super(PositionEmbeddingMixin, self).__init__()
        self.reinit_slice = reinit_slice
        self.position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)
    def reinit(self, *pre_mixins):
        old_weights = self.transformer.position_embeddings.weight.data[self.reinit_slice]
        old_len, hidden_size = old_weights.shape
        assert hidden_size == self.position_embeddings.weight.shape[-1]
        self.position_embeddings.weight.data.view(-1, old_len, hidden_size).copy_(old_weights)

class AttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02)
        ):
        super(AttentionMixin, self).__init__()
        self.num_layers = num_layers # replace attention in the LAST n layers
        self.query_key_value = torch.nn.ModuleList(
            [ColumnParallelLinear(hidden_size, 3*hidden_size,stride=3,
                gather_output=False,init_method=init_method)
                for layer_id in range(num_layers)
            ])
        self.dense = torch.nn.ModuleList(
            [RowParallelLinear(hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
                for layer_id in range(num_layers)
            ])
    def reinit(self, *pre_mixins):
        start_layer = len(self.transformer.layers) - self.num_layers
        assert start_layer >= 0
        for layer_id in range(self.num_layers):
            old_attention = self.transformer.layers[start_layer + layer_id].attention
            self.query_key_value[layer_id].weight.data.copy_(old_attention.query_key_value.weight.data)
            self.query_key_value[layer_id].bias.data.copy_(old_attention.query_key_value.bias.data)
            self.dense[layer_id].weight.data.copy_(old_attention.dense.weight.data)
            self.dense[layer_id].bias.data.copy_(old_attention.dense.bias.data)
