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

 
import random

import numpy as np
import torch
import json


from fp16 import FP16_Optimizer
import mpu
from utils.utils import get_checkpoint_name, ensure_directory_exists,get_checkpoint_tracker_filename



def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))
    
    
def save_ds_checkpoint(iteration, model, lr_scheduler, args, tag):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    if lr_scheduler is not None:
        sd['client_lr_scheduler'] = lr_scheduler.state_dict()
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
    model.save_checkpoint(args.save, tag, client_state=sd)



def save_checkpoint(iteration, model, optimizer, lr_scheduler, args, tag=None, barrier=True,
                    only_changed_parameters=False, no_deepspeed=False, no_save_optim=False):
    """Save a model checkpoint."""
    if tag is None:
        tag = str(iteration)
    if args.deepspeed and not no_deepspeed:
        save_ds_checkpoint(iteration, model, lr_scheduler, args, tag=tag)
    else:
        # Only rank zer0 of the data parallel writes to the disk.

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, tag)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                  format(torch.distributed.get_rank(), iteration, checkpoint_name))
            sd = {'iteration': iteration}
            if args.deepspeed:
                model = model.module
            state_dict = model.state_dict()
            if only_changed_parameters:
                requires_grad_dict = {}
                for name, parameter in model.named_parameters():
                    requires_grad_dict[name] = parameter.requires_grad
                state_dict = {key: value for key, value in state_dict.items() if requires_grad_dict[key]}
            sd['module'] = state_dict

            # Optimizer stuff.
            if not args.no_save_optim and not no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    if barrier:
        torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(tag)





