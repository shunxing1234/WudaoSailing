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

"""Pretrain GPT2"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.

import os
import random

import numpy as np
import torch
import mpu
import nets
from dataprocessing.train_data import get_batch,get_two_batch




def forward_step(data_iterator, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    timers('data loader').start()
    rand = random.Random(args.iteration * mpu.get_data_parallel_world_size() + mpu.get_data_parallel_rank())
    if data_iterator[1] and rand.random() < args.multi_task_ratio:
        data = next(data_iterator[1]) if data_iterator[1] else None
        data["mode"] = "multi-task"
    else:
        data = next(data_iterator[0]) if data_iterator[0] else None
    # print_rank_0("data iterator")
    timers('data loader').stop()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    timers('batch generator').stop()

    # print_rank_0("get batch")

    def print_masked_text(batch_id):
        block_position_ids = position_ids[:, 1]
        position_ids_ = position_ids[:, 0]
        sep = attention_mask.item() if torch.numel(attention_mask) == 1 else attention_mask[batch_id].item()
        text, last_segment = "", []
        for i, token_id in enumerate(tokens[batch_id, :sep].tolist()):
            token = tokenizer.IdToToken(token_id)
            if token.startswith('[MASK') or token.endswith('MASK]'):
                if last_segment:
                    text += tokenizer.DecodeIds(last_segment)
                    last_segment = []
                text += f" [{position_ids_[batch_id, i].item()}, {token}]"
            else:
                last_segment.append(token_id)
        if last_segment:
            text += tokenizer.DecodeIds(last_segment)
        print(text.encode('utf-8'))
        last_index = None
        for i in range(sep, tokens.size(1)):
            if tokenizer.IdToToken(tokens[batch_id, i].item()).startswith("<|startofpiece"):
                if last_index is not None:
                    print(tokenizer.DecodeIds(tokens[batch_id, last_index: i].tolist()).encode('utf-8'), "|",
                          tokenizer.DecodeIds(labels[batch_id, last_index: i].tolist()).encode('utf-8'),
                          position_ids_[batch_id, last_index: i].tolist(),
                          block_position_ids[batch_id, last_index:i].tolist())
                last_index = i
        if last_index is not None:
            print(tokenizer.DecodeIds(tokens[batch_id, last_index:].tolist()).encode('utf-8'), "|",
                  tokenizer.DecodeIds(labels[batch_id, last_index:].tolist()).encode('utf-8'),
                  position_ids_[batch_id, last_index:].tolist(), block_position_ids[batch_id, last_index:].tolist())

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'

    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    losses = nets.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                              labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()

    return loss, mems, mode


