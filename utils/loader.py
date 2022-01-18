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

"""Utilities for logging and serialization"""

import os
import random
import time
import numpy as np
import torch
import json
import subprocess

from fp16 import FP16_Optimizer
import mpu

from utils.utils import get_checkpoint_name, ensure_directory_exists,get_checkpoint_tracker_filename,get_checkpoint_iteration
from utils.prints import print_rank_0
 


def load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=False, no_load_optim=False):
    """Load a model checkpoint."""

    load_dir, tag, release, success = get_checkpoint_iteration(args.load)

    if not success:
        return 0

    if args.deepspeed and not no_deepspeed:
        checkpoint_name, sd = model.load_checkpoint(load_dir, tag,
                                                    load_optimizer_states=not args.no_load_optim and not no_load_optim,
                                                    load_lr_scheduler_states=not args.no_load_lr_scheduler)
        if not args.no_load_lr_scheduler and "client_lr_scheduler" in sd:
            lr_scheduler.load_state_dict(sd["client_lr_scheduler"])
            print_rank_0("Load lr scheduler state")
        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return tag

    else:

        # Checkpoint.
        checkpoint_name = get_checkpoint_name(load_dir, tag, release)

        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        # Model.
        if args.deepspeed:
            model = model.module
        missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
        if missing_keys or unexpected_keys:
            print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim and not no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                             'Specify --no-load-optim or --finetune to prevent '
                             'attempting to load the optimizer '
                             'state.'.format(checkpoint_name))

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, starting from 0 iteration'.format(checkpoint_name))
                iteration = 0

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load random state from checkpoint {}, exiting. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the random '
                         'state.'.format(checkpoint_name))

    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def load_weights(src, dst, dst2src=False):
    """
    Loads weights from src to dst via in place copy.
    src is a huggingface gpt2model, while dst is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src is still untested
    """
    conv_layer = 'Conv1D' in str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)


#        dst._parameters[n].data.copy_(data)

def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)


def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)


def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)
    
    
def load_pretrained(model, checkpoint_path, args, task_tokens=None):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading pretrained model {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')
    if args.deepspeed:
        model = model.module
    if isinstance(model, TorchDDP):
        model = model.module
    if isinstance(model, FP16_Module):
        model = model.module
    if hasattr(model, "model"):
        model = model.model

    # Model.
    def extend_embedding_weights(state_weights, model_weights):
        original_length = state_weights.shape[0]
        assert original_length <= args.max_position_embeddings + 1
        new_weights = model_weights.clone()
        new_weights[:original_length] = state_weights
        return new_weights

    if args.block_lm:
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd['module']["transformer.position_embeddings.weight"]
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
                    position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
                print_rank_0(f"Extend position embedding to {args.max_position_embeddings + 1}")
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
            if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
                sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
                    block_position_weights,
                    model.state_dict()["transformer.block_position_embeddings.weight"].data)
                print_rank_0(f"Extend block position embedding to {args.max_position_embeddings + 1}")
    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    if args.continuous_prompt and args.prompt_init:
        model.prompt_spell.init_embedding(model.word_embeddings.weight.data, task_tokens)


 
