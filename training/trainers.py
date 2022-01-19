import torch
import mpu
from fp16 import   DynamicLossScaler
from utils.prints import print_rank_0 
from training.backward import backward_step
from training.forward import forward_step
from utils.reports import  report_memory

from utils.saver import save_checkpoint
 
from utils.reports import report_iteration_metrics,report_evaluate_metrics
 
from training.evaluate import evaluate_and_print_results,evaluate

def train_step(data_iterator, model, optimizer, lr_scheduler, args, timers, forward_step_func, mems=None,
               single_step=False):
    """Single training step."""
    lm_loss_total, count = 0.0, 0
    mems = [] if mems is None else mems
    if not args.deepspeed:
        optimizer.zero_grad()
    while True:
        skipped_iter, complete = 0, False
        # Forward model for one step.
        timers('forward').start()
        lm_loss, mems, _ = forward_step_func(data_iterator, model, args, timers, mems)
        timers('forward').stop()
        # print_rank_0("Forward step")
        if not args.deepspeed:
            lm_loss /= args.gradient_accumulation_steps

        reduced_loss = lm_loss.detach().clone().view(1)
        torch.distributed.all_reduce(reduced_loss.data, group=mpu.get_data_parallel_group())
        reduced_loss.data = reduced_loss.data / (args.world_size / args.model_parallel_size)

        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            lm_loss_total += reduced_loss
            count += 1

            # Calculate gradients, reduce across processes, and clip.
            timers('backward').start()
            backward_step(optimizer, model, lm_loss, args, timers)
            timers('backward').stop()
            # print_rank_0("Backward step")
            # Update parameters.
            timers('optimizer').start()
            if args.deepspeed:
                if model.is_gradient_accumulation_boundary():
                    model.step()
                    complete = True
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
                else:
                    model.step()
            else:
                if count == args.gradient_accumulation_steps:
                    optimizer.step()
                    complete = True
                    # Update learning rate.
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
            # print_rank_0("Optimizer step")
            timers('optimizer').stop()
            if complete:
                break
        else:
            print_rank_0("Found NaN loss, skip backward")
            del lm_loss, reduced_loss
            mems = []
        if single_step:
            break
    if args.deepspeed:
        lm_loss_total = lm_loss_total / count
    return lm_loss_total, skipped_iter, mems


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, summary_writer=None):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    mems = []
    while args.iteration < args.train_iters:

        lm_loss, skipped_iter, mems = train_step(train_data_iterator,
                                                 model,
                                                 optimizer,
                                                 lr_scheduler,
                                                 args, timers, mems=mems, forward_step_func=forward_step)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False
            # for i in range(torch.distributed.get_world_size()):
            #     if i == torch.distributed.get_rank():
            #         print(get_hostname())
            #         timers.log(['forward', 'backward', 'optimizer',
            #                     'batch generator', 'data loader'],
            #                    normalizer=args.log_interval, reset=False)
            #     torch.distributed.barrier()
            if args.deepspeed or args.DDP_impl == 'torch':
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, args, timers, verbose=False, step=args.iteration,
                summary_writer=summary_writer, forward_step_func=forward_step)

    return args.iteration, skipped_iters
