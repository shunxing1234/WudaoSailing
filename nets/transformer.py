# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""


import copy
import torch
import torch.nn.init as init
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from mpu.initialize import get_model_parallel_world_size
from nets.embeddings import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from mpu.mappings import gather_from_model_parallel_region, copy_to_model_parallel_region
import deepspeed
import torch.nn.functional as F
from mpu.random import checkpoint
from mpu.random import get_cuda_rng_tracker

from mpu.utils import divide
from mpu.utils import split_tensor_along_last_dim
from nets.embeddings import PositionalEmbedding

from nets.attentions import ParallelCrossAttention, ParallelSelfAttention, BertParallelSelfAttention

from nets.decoders import ParallelMLP, ParallelDecoderLayer
from nets.layer_norm import CogLayerNorm
from mpu.func_utils import sqrt, scaled_init_method, unscaled_init_method, gelu
from nets.attentions import standard_attention, SelfAttention
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

class MLP(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method,
                output_layer_init_method=None, hooks={}):
        super(MLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            4*hidden_size,
            gather_output=False,
            init_method=init_method
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method
        )
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args, layer_id=self.layer_id)
        else:
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = gelu(intermediate_parallel)
            output = self.dense_4h_to_h(intermediate_parallel)
            
        if self.training:
            output = self.dropout(output)
        return output


class BaseTransformerLayer(torch.nn.Module):
    """
    Do pre-LN structure if Sandwich-LN not specified
    """
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        layernorm_epsilon,
        init_method,
        layer_id,
        output_layer_init_method=None,
        sandwich_ln=True,
        hooks={}
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.hooks = hooks

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            output_layer_init_method=output_layer_init_method,
            hooks=hooks
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.sandwich_ln = sandwich_ln
        if sandwich_ln:
            self.third_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            hooks=hooks
        )
    
    def forward(self, hidden_states, mask, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''

        # Layer norm at the begining of the transformer layer.
        layernorm_output1 = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, output_this_layer = self.attention(layernorm_output1, mask, **kw_args)

        # Third LayerNorm
        if self.sandwich_ln:
            attention_output = self.third_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output, **kw_args)

        # Fourth LayerNorm
        if self.sandwich_ln:
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output, output_this_layer  # temporally, output_this_layer is only from attention


class BaseTransformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 sandwich_ln=True,
                 parallel_output=True,
                 hooks={}
                 ):
        super(BaseTransformer, self).__init__()
        
        # recording parameters
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.hooks = copy.copy(hooks)  # hooks will be updated each forward
        
        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=unscaled_init_method(0.02))
        
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        # create all layers
        self.output_layer_init_method = scaled_init_method(init_method_std, num_layers)
        self.init_method = unscaled_init_method(init_method_std)
        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                output_layer_init_method=self.output_layer_init_method,
                sandwich_ln=sandwich_ln,
                hooks=self.hooks
                )
        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, input_ids, position_ids, attention_mask, *, branch_input=None, **kw_args):
        # sanity check 
        assert len(input_ids.shape) == 2 
        batch_size, query_length = input_ids.shape
        assert len(attention_mask.shape) == 2 or \
            len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1
        assert branch_input is None or 'layer_forward' in self.hooks and isinstance(branch_input, torch.Tensor)
        # branch_input is a new part of input need layer-by-layer update,
        #   but with different hidden_dim and computational routine.
        #   In most cases, you can just ignore it.

        # embedding part
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](input_ids, **kw_args)
        else: # default
            hidden_states = self.word_embeddings(input_ids)
            
        if 'position_embedding_forward' in self.hooks:
            position_embeddings = self.hooks['position_embedding_forward'](position_ids, **kw_args)
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == query_length
            position_embeddings = self.position_embeddings(position_ids)    
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Branch related embedding
        if branch_input is None and 'branch_embedding_forward' in self.hooks:
            branch_input = self.hooks['branch_embedding_forward'](branch_input, **kw_args)

        # Define custom_forward for checkpointing
        output_per_layers = []
        if self.checkpoint_activations:
            def custom(start, end):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask = inputs[0], inputs[1]    
                    if len(inputs) > 2:  # have branch_input
                        branch_ = inputs[2]     
                    output_per_layers_part = []               
                    for i, layer in enumerate(layers_):
                        if len(inputs) > 2:
                            x_, branch_, output_this_layer = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id, branch_input=branch_, **kw_args
                            )
                        elif 'layer_forward' in self.hooks:
                            x_, output_this_layer = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id, **kw_args
                            )
                        else:
                            x_, output_this_layer = layer(x_, mask, **kw_args)
                        output_per_layers_part.append(output_this_layer)
                    return x_, output_per_layers_part
                return custom_forward
        
            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask]
                if branch_input is not None:
                    hidden_states, branch_input, output_per_layers_part = checkpoint(custom(l, l + chunk_length),
                                                                                     *args, branch_input)
                else:
                    hidden_states, output_per_layers_part = checkpoint(custom(l, l + chunk_length), *args)
                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]
                if branch_input is not None:  # customized layer_forward with branch_input
                    hidden_states, branch_input, output_this_layer = self.hooks['layer_forward'](*args, layer_id=torch.tensor(i), branch_input=branch_input, **kw_args)
                elif 'layer_forward' in self.hooks:  # customized layer_forward
                    hidden_states, output_this_layer = self.hooks['layer_forward'](*args, layer_id=torch.tensor(i), **kw_args)
                else:
                    hidden_states, output_this_layer = layer(*args, **kw_args)
                output_per_layers.append(output_this_layer) 

        # Final layer norm.
        logits = self.final_layernorm(hidden_states)
        
        if 'final_forward' in self.hooks:
            logits_parallel = self.hooks['final_forward'](logits, **kw_args)
        else:
            logits_parallel = copy_to_model_parallel_region(logits)
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)
            
        # branch related embedding
        if branch_input is None and 'branch_final_forward' in self.hooks:
            branch_input = self.hooks['branch_final_forward'](branch_input, **kw_args)

        if not self.parallel_output:
            logits_parallel = gather_from_model_parallel_region(logits_parallel)
            
        if branch_input is not None:
            return (logits_parallel, branch_input, *output_per_layers)
        
        return (logits_parallel, *output_per_layers)
 


class ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False,
                 performer=False,
                 attention_scale=1.0):
        super(ParallelTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            relative_encoding=relative_encoding,
            performer=performer,
            attention_scale=attention_scale)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output





class GPT2ParallelTransformer(torch.nn.Module):
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 performer=False,
                 use_decoder_layer=False,
                 attention_scale=1.0,
                 ):
        super(GPT2ParallelTransformer, self).__init__()
        self.hidden_size = hidden_size
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.performer = performer
        self.use_decoder_layer = use_decoder_layer
        assert not (performer and relative_encoding)

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                          num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        if relative_encoding:
            # Relative position embedding
            self.position_embeddings = PositionalEmbedding(hidden_size)
            # Per attention head and per partition values.
            world_size = get_model_parallel_world_size()
            self.hidden_size_per_attention_head = divide(hidden_size,
                                                         num_attention_heads)
            self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                            world_size)
            self.r_w_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_w_bias.model_parallel = True
            self.r_r_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_r_bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.r_w_bias.zero_()
                self.r_r_bias.zero_()
        else:
            # Position embedding (serial).
            if block_position_encoding:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                self.block_position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
            else:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
            # Initialize the position embeddings.
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer():
            if use_decoder_layer:
                return ParallelDecoderLayer(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    layernorm_epsilon,
                    unscaled_init_method(init_method_std),
                    output_layer_init_method=output_layer_init_method
                )
            else:
                return ParallelTransformerLayer(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    layernorm_epsilon,
                    unscaled_init_method(init_method_std),
                    output_layer_init_method=output_layer_init_method,
                    relative_encoding=relative_encoding,
                    performer=performer,
                    attention_scale=attention_scale)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, position_ids, attention_mask, memory_states=None, encoder_states=None,
                return_memory=False, detach_memory=True):
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = memory_states[0].size(1) if memory_states else 0
        key_length = query_length + memory_length
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if self.performer:
            assert is_scalar, 'attention_mask should be a scalar to indicate the seperation position.'
            assert memory_length == 0, 'Do not support transformer-xl.'
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones((batch_size, seq_length, memory_length)), m), dim=2)
                m = m.unsqueeze(1)
                return m

            if not self.performer:
                attention_mask = build_mask_matrix(query_length, sep, memory_length=memory_length)
        else:
            attention_mask = attention_mask[:, :, :, -query_length - memory_length:]

        if self.relative_encoding:
            position_sequence = torch.arange(key_length - 1, -1, -1.0, device=hidden_states.device,
                                             dtype=hidden_states.dtype)
            position_embeddings = self.position_embeddings(position_sequence)
            # Apply dropout
            position_embeddings = self.embedding_dropout(position_embeddings)
        else:
            if self.block_position_encoding:
                position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(block_position_ids)
                hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            if detach_memory:
                return _hidden_states.detach()
            return _hidden_states

        if self.max_memory_length > 0 or return_memory:
            mem_layers = [check_detach(hidden_states)]
        else:
            mem_layers = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                if self.relative_encoding:
                    inputs, mems_ = inputs[:4], inputs[4:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0 or return_memory:
                        mem_layers.append(check_detach(x_))
                return x_

            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask] if not self.use_decoder_layer else [hidden_states,
                                                                                           encoder_states,
                                                                                           attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                if memory_states:
                    args += memory_states[l: l + chunk_length]
                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask] if not self.use_decoder_layer else [hidden_states,
                                                                                           encoder_states,
                                                                                           attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                mem_i = memory_states[i] if memory_states else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0 or return_memory:
                    mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0 or return_memory:
            mem_layers = self.update_mems(mem_layers, memory_states, return_memory=return_memory)

        return (output, mem_layers)

    def update_mems(self, hiddens, mems, return_memory=False):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i]), dim=1))
        return new_mems





class BertParallelTransformerOutput(torch.nn.Module):
    """The output layer used after self attention and intermediate
    parts of transformer layer."""

    def __init__(self, input_size, output_size, dropout_prob,
                 layernorm_epsilon=1.0e-12, input_is_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerOutput, self).__init__()
        # Components.
        self.dense = RowParallelLinear(input_size,
                                       output_size,
                                       input_is_parallel=input_is_parallel,
                                       init_method=init_method)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.layernorm = LayerNorm(output_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        layernorm_input = hidden_states + input_tensor
        hidden_states = self.layernorm(layernorm_input)
        return hidden_states


class BertParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for Bert.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        intermediate_size: size of the intermediate state after
                           self attention. In both BERT and GPT
                           this is set to be 4 times the hidden
                           size.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        intermediate_activation_fn: activation function for output
                                    of intermediate.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
    """

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 intermediate_activation_fn,
                 layernorm_epsilon,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerLayer, self).__init__()

        # Self attention.
        self.attention = BertParallelSelfAttention(hidden_size,
                                                   num_attention_heads,
                                                   attention_dropout_prob,
                                                   output_parallel=True,
                                                   init_method=init_method)
        # Self attention output.
        self.self_output = BertParallelTransformerOutput(
            hidden_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)
        # Intermediate.
        self.intermediate = ColumnParallelLinear(hidden_size, intermediate_size,
                                                 gather_output=False,
                                                 init_method=init_method)
        self.intermediate_activation_fn = intermediate_activation_fn
        # Output.
        self.output = BertParallelTransformerOutput(
            intermediate_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)

    def forward(self, hidden_states, attention_mask):
        # [b, s, hp]
        attention_output_parallel = self.attention(hidden_states,
                                                   attention_mask)
        # [b, s, h]
        attention_self_output = self.self_output(attention_output_parallel,
                                                 hidden_states)
        # [b, s, ip]
        intermediate_output_parallel = self.intermediate(attention_self_output)
        intermediate_output_parallel = self.intermediate_activation_fn(
            intermediate_output_parallel)
        # [b, s, h]
        layer_output = self.output(intermediate_output_parallel,
                                   attention_self_output)

        return layer_output
