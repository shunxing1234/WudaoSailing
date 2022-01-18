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
import sys
root_dir='/data/wang/models/wudao'
sys.path.append(root_dir)
# Flag to use Pytorch ddp which uses overlapping communication and computation.
from datetime import datetime
import os
from filelock import FileLock
import numpy as np
import torch
from contextlib import ExitStack
from arguments import get_args
from data_utils.configure_data import   prepare_tokenizer, build_multi_task_dataset
from setup_model import setup_model_and_optimizer
from utils.reports import Timers
from utils.saver import save_checkpoint
from utils.loader import load_checkpoint
from utils.prints import print_and_save_args, print_rank_0
from utils.utils import get_sample_writer, get_log_dir
from utils.prints import print_rank_0
from train_utils.forward import forward_step
from data_utils.train_data import get_train_val_test_data 
from train_utils.evaluate import evaluate_and_print_results 
from train_utils.train_init import initialize_distributed,set_random_seed
import pathlib
from train_utils.trainers import train

def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Timer.
    timers = Timers()
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # Arguments.
    args = get_args()
    args.mem_length = args.mem_length if args.transformer_xl else 0
    if args.load and not args.new_save_directory:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    global tokenizer
    tokenizer = prepare_tokenizer(args)
    train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)
    multi_train_data, multi_val_data = None, None
    if args.multi_task_ratio > 0.0:
        multi_train_data, multi_val_data = build_multi_task_dataset(args, tokenizer)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    if args.load is not None:
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0
    torch.distributed.barrier()
    if args.switch_linear:
        lr_scheduler.switch_linear(args)

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        args.log_dir = None
        if args.train_iters > 0:
            args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
            summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        print_rank_0("Resume dataloader")
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % len(train_data)
        if val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters
            val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
        if multi_train_data is not None:
            multi_train_data.batch_sampler.start_iter = int(args.iteration * args.multi_task_ratio) % len(
                multi_train_data)
        if multi_val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters * args.multi_task_ratio
            multi_val_data.batch_sampler.start_iter = start_iter_val % len(multi_val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if multi_train_data is not None:
        multi_train_iterator = iter(multi_train_data)
    else:
        multi_train_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None
    if multi_val_data is not None:
        multi_val_iterator = iter(multi_val_data)
    else:
        multi_val_iterator = None

    # TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)

                # stack.callback(save_on_exit, args, model, optimizer, lr_scheduler)
                iteration, skipped = train(model, optimizer,
                                           lr_scheduler,
                                           (train_data_iterator, multi_train_iterator),
                                           (val_data_iterator, multi_val_iterator),
                                           timers, args, summary_writer=summary_writer)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, timers, verbose=False, forward_step_func=forward_step)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, (test_data_iterator, None),
                                   model, args, timers, verbose=True, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
