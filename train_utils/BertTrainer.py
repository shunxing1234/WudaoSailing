import time
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from utils.model_loader import load_model
from utils.model_saver import save_model
from utils.struct_nav import str2dataloader, str2dataset
from utils.setting_nav import str2scheduler, str2optimizer
from utils.seed import set_seed
from data_utils.BertDataset import BertDataset

from data_utils.dataloader import BertDataLoader


def train_and_validate(args, model):
    set_seed(args.seed)

    # Load pretrained model if exists
    if args.load_pretrained:
        print("加载预训练模型："+args.load_pretrained)
        model = load_model(model, args.load_pretrained)
    else:
        # Initialize with normal distribution.
        if args.deep_init:
            scaled_factor = 1 / math.sqrt(2.0 * args.layers_num)
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    if "linear_2.weight" in n or "final_linear.weight" in n:
                        p.data.normal_(0, 0.02 * scaled_factor)
                    elif "linear_2.bias" in n or "final_linear.bias" in n:
                        p.data.zero_()
                    else:
                        p.data.normal_(0, 0.02)
        else:
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)

    if args.deepspeed:
        worker(args.local_rank, None, args, model)
    elif args.dist_train:
        # Multiple GPU mpde
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU Mode
        worker(args.gpu_id, None, args, model)
    else:
        # CPU Mode
        worker(None, None, args, model)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.report_steps = args.report_steps
        self.save_interval = args.save_interval
        self.vocab = args.tokenizer.vocab
        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.tokenizer = args.tokenizer

    def forward_propagation(self, batch, model):
        raise NotImplementedError

    def predict(self, batch, model):
        raise NotImplementedError

    def report_and_reset_stats(self):
        raise NotImplementedError

    def validate(self,args,gpu_id,loader,model):
        loader_iter = iter(loader)
        batch = list(next(loader_iter))
        batch = batch[0,:]
        info = self.forward_propagation(batch, model)
        return info

    def train(self, args, gpu_id, rank, loader, demo_loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)
        cum_samp = 0
        print('1')
        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))          # load batch data
            cum_samp = cum_samp+len(batch)
            if cum_samp >= len(loader.buffer)-args.batch_size:
                if args.dynamic_masking:
                    print('procssing data')
                    dataset = str2dataset[args.dataset](args, self.tokenizer.vocab, self.tokenizer)
                    dataset.build_and_save(8)
                if args.dist_train:
                    loader = str2dataloader[args.dataloader](args,  args.batch_size, rank, args.world_size, True)
                else:
                    loader = str2dataloader[args.dataloader](args,   args.batch_size, 0, 1, True)
                    demo_loader = str2dataloader[args.dataloader](args,   args.batch_size, 0, 1, True, demo_mode=True)
                loader_iter = iter(loader)
                cum_samp = 0
                
            self.seq_length = batch[0].size(1)
           
            if gpu_id is not None:                   # Put data to GPU is GPU available
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            if args.deepspeed:
                model.backward(loss)
            else:
                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            if self.current_step % self.gradient_accumulation_steps == 0:
                if args.deepspeed:
                    model.step()
                else:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()
                loader_iter = iter(demo_loader)
                batch = list(next(loader_iter))
                if gpu_id is not None:
                    for i in range(len(batch)):
                        batch[i] = batch[i].cuda(gpu_id)
                src, tgt_mlm, tgt_sp, seg = batch
                info = self.predict(batch, model)
                output_mlm, tgt_mlm = info
                output_mlm_list = torch.argmax(output_mlm, dim=1).tolist()
                tgt_mlm_list = tgt_mlm.tolist()[0]

                text_input = self.tokenizer.convert_ids_to_tokens(src.tolist()[0])
                masked_positions = []
                # Add highlights for masked positions
                for i in range(len(tgt_mlm_list)):
                    if tgt_mlm_list[i] > 0:
                        masked_positions.append(i)
                        # Assume length of masked position is 5, append space if less than that
                        if len(text_input[i]) < 5:
                            text_input[i] = text_input[i].center(5, " ")
                        text_input[i] = "\033[4;31m" + text_input[i] + "\033[0m"

                text_output = "".join(text_input).replace('[PAD]','')

                def cut(obj, sec):
                    return [obj[i:i + sec] for i in range(0, len(obj), sec)]
                # text_input = cut(text_input, 100)
                # for e in text_input:
                #     print(e)
                tgt_mlm_list = [ind for ind in tgt_mlm_list if ind > 0]
                print("输入的处理后文本：")
                print(text_output)
                # print("模型预测的被遮挡的字:"+str(self.tokenizer.convert_ids_to_tokens(output_mlm_list[:len(tgt_mlm_list)])))
                # print("被遮挡的字的正确答案:"+str(self.tokenizer.convert_ids_to_tokens(tgt_mlm_list)))

                print("\n填入预测词后的文本：")
                fills = self.tokenizer.convert_ids_to_tokens(output_mlm_list[:len(tgt_mlm_list)])
                for i in masked_positions:
                    fill = fills.pop(0)
                    if len(fill) < 5:
                        fill = fill.center(5, " ")
                    text_input[i] = "\033[4;32m" + fill + "\033[0m"
                text_output = "".join(text_input).replace('[PAD]','')
                print(text_output)

                print("\n原始文本(目标文本)")
                fills = self.tokenizer.convert_ids_to_tokens(tgt_mlm_list)
                for i in masked_positions:
                    fill = fills.pop(0)
                    if len(fill) < 5:
                        fill = fill.center(5, " ")
                    text_input[i] = "\033[4;34m" + fill + "\033[0m"
                text_output = "".join(text_input).replace('[PAD]','')
                print(text_output)
                print("\n")

            if args.deepspeed:
                if self.current_step % self.save_interval == 0:
                    model.save_checkpoint(self.output_model_path, str(self.current_step))
            else:
                if self.current_step % self.save_interval == 0 and \
                        (not self.dist_train or (self.dist_train and rank == 0)):
                    save_model(model, self.output_model_path + "-" + str(self.current_step))

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.gradient_accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def predict(self, batch, model):
        src, tgt_mlm, tgt_sp, seg = batch
        loss_info = model(src, (tgt_mlm, tgt_sp), seg)
        # print(loss_info)
        # print(len(loss_info))
        output_mlm, tgt_sp = loss_info[0], loss_info[6]
        return output_mlm, tgt_mlm

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_sp, seg = batch
        loss_info = model(src, (tgt_mlm, tgt_sp), seg)
        output_mlm, loss_mlm, loss_sp, correct_mlm, correct_sp, denominator, tgt_mlm = loss_info
        loss = loss_mlm + loss_sp
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.gradient_accumulation_steps
        
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_sp: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_sp: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_mlm / self.report_steps,
                  self.total_loss_sp / self.report_steps,
                  self.total_correct_mlm / self.total_denominator,
                  self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


str2trainer = {"bert": BertTrainer, "mlm": MlmTrainer}


def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend=args.backend)
        rank = dist.get_rank()
        gpu_id = proc_id
    elif args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = str2dataloader[args.dataloader](args,  args.batch_size, rank, args.world_size, True)
    else:
        train_loader = str2dataloader[args.dataloader](args,  args.batch_size, 0, 1, True)
        demo_loader = str2dataloader[args.dataloader](args,   args.batch_size, 0, 1, True, demo_mode=True)

    # Construct optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    if args.optimizer in ["adamw"]:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    else:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.lr, scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup)
    else:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup, args.total_steps)

    if args.deepspeed:
        optimizer = None
        scheduler = None

        # IF User NOT defined optimezer in deepspeed config,
        # Then use Self Defined Optimizer
        if "optimizer" not in args.deepspeed_config_param:
            optimizer = custom_optimizer
            if args.local_rank == 0:
                print("Use Custum Optimizer", optimizer)
        if "scheduler" not in args.deepspeed_config_param:
            scheduler = custom_scheduler
            if args.local_rank == 0:
                print("Use Custom LR Schedule", scheduler)
        model, optimizer, _, scheduler = deepspeed.initialize(
                                                    model=model,
                                                    model_parameters=optimizer_grouped_parameters,
                                                    args=args,
                                                    optimizer=optimizer,
                                                    lr_scheduler=scheduler,
                                                    mpu=None,
                                                    dist_init_required=False)
    else:
        if gpu_id is not None:
            model.cuda(gpu_id)
        optimizer = custom_optimizer
        scheduler = custom_scheduler
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            args.amp = amp

        if args.dist_train:
            # Initialize multiprocessing distributed training environment.
            dist.init_process_group(backend=args.backend,
                                    init_method=args.master_ip,
                                    world_size=args.world_size,
                                    rank=rank)
            model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
            print("Worker %d is training ... " % rank)
        else:
            print("Worker is training ...")
    
    trainer = BertTrainer(args)
    
    trainer.train(args, gpu_id, rank, train_loader, demo_loader, model, optimizer, scheduler)

