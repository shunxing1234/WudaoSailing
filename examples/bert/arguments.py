import configparser
import argparse
import torch
import os

import json
import sys
from argparse import Namespace
from utils.io_utils import *



def build_args(**kargs):
    parser,init_args = init_agrs(**kargs)
    add_args_path(parser,init_args)
    add_args_mask(parser)
    add_args_preprocess(parser)
    add_args_model_config(parser)
    add_args_training(parser)
    add_args_fp16_config(parser)
    add_args_opts(parser)
    add_args_deepspeed(parser)
    args = parser.parse_args(args=[])
    update_args_dist(args)
    return args



def init_agrs(**kargs):
    init_args=dict()
    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    if 'root_dir' not in kargs.keys():
        init_args['root_dir']=os.getcwd().split('/test')[0]
        parser.add_argument('--root_dir', type=str,
                            default=os.getcwd().split('/test')[0],
                            help='root directory of the packages')
    for k in kargs.keys():
        parser.add_argument('--'+k, type=type(kargs[k]),default=kargs[k])
        if 'root_dir' in k:
            init_args['root_dir']=kargs[k]
    return parser,init_args



def add_args_path(parser,init_args):
    """path settings"""
    group = parser.add_argument_group('path', 'some basic path setting')
    root_dir=init_args['root_dir']
    group.add_argument('--corpus_dir', type=str, default="./data/train_data/corpus",
                       help='Path to a directory containing corpsus.')
    group.add_argument('--pt_dir', type=str, default= root_dir+"/data/train_data/pre_data/"  ,
                       help='Path to a directory containing the training data.')
    group.add_argument('--vocab_path', type=str, default="./data/vocab_data/google_zh_vocab.txt",
                       help='Path to the vocabs.')
    group.add_argument('--output_model_path', type=str, default="./data/model_save/book_review_model.bin",
                       help='Path to the output model.')

    group.add_argument('--load_pretrained', type=str, default="",
                       help='load model path.')
    group.add_argument('--deepspeed_config', type=str, default=root_dir+"/data/model_save/deepspeed_config.json",
                       help='Path to a config file for deepspeed.')
    group.add_argument('--temp_dir', type=str, default="./data/temp/",
                       help='directory of temporary files.')
    return parser
    
    
    
def add_args_mask(parser):
    """msk"""
    group = parser.add_argument_group('mask', 'mask setting')
    group.add_argument('--mask', type=str, default='fully_visible',
                       choices=["fully_visible", "causal", "causal_with_prefix"],
                       help='mask type')
    group.add_argument('--dynamic_masking', type=bool, default=False,
                       help='dynamic masking')
    group.add_argument('--whole_word_masking', type=bool, default=False,
                       help='word masking')
    group.add_argument('--span_masking', type=bool, default=False,
                       help='word masking')
    group.add_argument('--span_max_length', type=int, default=10,
                       help='the max length of the span masking')
    group.add_argument('--span_geo_prob', type=float, default=0.2,
                       help='')
    return parser
    

def add_args_preprocess(parser):
    """setting for processing datasets"""
    group = parser.add_argument_group('datasets', 'dataset-preprocessing configuration')
    group.add_argument('--tokenizer', type=str, default="bert",
                       help='process tokenizer type')
    group.add_argument('--pre_dataset', type=str, default="bert",
                       help='process  tokenizer type')
    group.add_argument('--preprocess_num', type=int, default=8,
                       help='the num of processers for preprocessing ')
    group.add_argument('--process_size', type=int, default=1000,
                       help='the number of samples for per work in one block')
    group.add_argument('--max_input_chars_per_word', type=int, default=200,
                       help='the max number of chars in one word')
    group.add_argument('--dup_factor', type=int, default=1,
                       help='duplication times')
    group.add_argument('--short_seq_prob', type=float, default=0.1,
                       help='probability for generating short sequence')
    group.add_argument('--docs_buffer_size', type=int, default=10000,
                       help='buffer size')
    group.add_argument('--demo_ratio', type=int, default=0.0001,
                       help='show probability')
    group.add_argument('--sentence_selection_strategy', type=str, default="lead" ,
                       help='type of position masking')
    group.add_argument('--file_preproces_dist', type=bool, default=True ,
                       help='the type for preprocessing the training files')
    
    group.add_argument('--file_types', type=list, default=["txt",'json','csv'],choices=["txt",'json','csv'] ,
                       help='type of files of the corpus')
    group.add_argument('--readers', type=dict, default={"json": read_json, "txt": read_txt, "csv": read_csv},
                       help='type of the supported files ')
    
    
    
    
    return parser
    

def add_args_model_config(parser):
    """modle configure"""
    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--attention_dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--num_attention_heads', type=int, default=12,
                       help='num of transformer attention heads')
    group.add_argument('--emb_size', type=int, default=128,
                       help='embedding size')
    group.add_argument('--hidden_size', type=int, default=128,
                       help='tansformer hidden size')
    group.add_argument('--intermediate_size', type=int, default=512,
                       help='transformer embedding dimension for FFN'
                            'set to 4*`--hidden-size` if it is None')
    group.add_argument('--num-layers', type=int, default=12,
                       help='num decoder layers')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='layer norm epsilon')
    group.add_argument('--hidden_dropout', type=float, default=0.1,
                       help='dropout probability for hidden state transformer')
    group.add_argument('--output_dropout', type=float, default=0.1,
                       help='dropout probability for pooled output')
    group.add_argument('--seq_length', type=int, default=128,
                       help='sequence length')
    group.add_argument('--max_seq_length', type=int, default=512,
                       help='max sequence length')
    group.add_argument('--vocab_size', type=int, default=30522,
                       help='vocab size to use for non-character-level '
                            'tokenization. This value will only be used when '
                            'creating a tokenizer')
    return parser
    
    

def add_args_training(parser):
    """Training arguments."""
    group = parser.add_argument_group('train', 'training configurations')
    group.add_argument('--experiment-name', type=str, default="gpt-345M",
                       help="The experiment name for summary and checkpoint")
    group.add_argument('--world_size', type=int, default=8,
                       help="The number of the gpus in total")
    
    group.add_argument('--gpu_ranks', type=list, default=[1],
                       help="The gpus")
    group.add_argument('--batch-size', type=int, default=64,
                       help='Data Loader batch size')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Data Loader batch size')
    group.add_argument('--epochs', type=int, default=30,
                       help='Number of  epochs in total.')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='report interval')
    group.add_argument('--summary-dir', type=str, default="", help="The directory to store the summary")
    group.add_argument('--seed', type=int, default=1234, help='random seed')
    group.add_argument('--lr', type=float, default=2e-5,
                       help='initial learning rate')
    group.add_argument('--scheduler', type=str, default='linear',
                       choices=['linear','consine','cosine_with_restarts','polynomial',
                                 'constant', 'constant_with_warmup'],
                       help='scheduler for learning rate')
    group.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw','adafactor'],
                       help='optimzier')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.1')
    group.add_argument('--save_epoch', type=int, default=1,
                       help='number of epochs between saves')
    group.add_argument('--save_interval', type=int, default=1000000,
                       help='number of iterations between saves')
    group.add_argument('--instances_buffer_size', type=int, default=25600,
                       help='size of instance buffer in training')
    return parser    
    
    
def add_args_fp16_config(parser):
    """Mixed precision arguments."""
    group = parser.add_argument_group('fp16', 'fp16 configurations')
    group.add_argument('--fp16', type=bool, default=False,
                       help='Run model in fp16 mode')
    group.add_argument('--fp16_opt_level', type=str, default='01',
                       choices=["01", "02", "03", "04"],
                       help='apex fp16 level, see https://nvidia.github.io/apex/amp.html')
    return parser



def add_args_opts(parser):
    """some opitional settings"""
    group = parser.add_argument_group('opitional', 'opitional configurations')
    group.add_argument('--remove_embedding_layernorm', type=bool, default=False,
                       help='whether include the layernorm')
    group.add_argument('--layernorm_positioning', type=str, default="pre",choices=["post","pre"],
                       help="the type of the layernorm positioning")
    group.add_argument('--parameter_sharing', type=bool, default=False,
                       help="parameter sharing between")
    group.add_argument('--factorized_embedding_parameterization', type=bool, default=False,
                       help="")
    group.add_argument('--has_residual_attention', type=bool, default=False,
                       help="")
    group.add_argument('--relative_position_embedding', type=bool, default=False,
                       help="")
    group.add_argument('--remove_transformer_bias', type=bool, default=False,
                       help="")
    group.add_argument('--remove_attention_scale', type=bool, default=False,
                       help="")
    group.add_argument('--tie_weights', type=bool, default=False,
                       help="")
    group.add_argument('--share_embedding', type=bool, default=False,
                       help="")
    group.add_argument('--deep_init', type=bool, default=False,
                       help="")
    group.add_argument('--feed_forward', type=str, default='dense',
                       help="")
    return parser


def add_args_deepspeed(parser):
    
    group = parser.add_argument_group('deepspeed', 'deepspeed configurations')
    group.add_argument('--deepspeed', type=bool, default=False,
                       help="")
    group.add_argument('--local_rank', type=int, default=0,
                       help="")
    group.add_argument('--backend', type=str, default="nccl", choices=["nccl","gloo"],
                       help="")
    group.add_argument('--master_ip', type=str, default="tcp://localhost:12345",
                       help="")
    return parser


def update_args_dist(args):
    """
    根据已有的信息更新分布式训练的参数，包括是否分布式训练(args.dist_train)以及是否用一个gpu(args.single_gpu)
    """
    if args.dynamic_masking:
        args.dup_factor = 1

    gpu_ranks = args.gpu_ranks
    world_size = args.world_size

    if args.deepspeed:
        if world_size > 1:
            args.dist_train = True
        else:
            args.dist_train = False
    else:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print("当前可用gpu个数: %d" % num_gpus)
            assert len(gpu_ranks) <= num_gpus, "设置的gpu数量超过可用的gpu数量"

            if num_gpus == 1:
                gpu_ranks = [0]

            if len(gpu_ranks) == 1:
                # 单gpu模式
                gpu_ranks = [0]  # 只可能用到一个gpu
                args.gpu_id = gpu_ranks[0]
                args.dist_train = False
                args.single_gpu = True
            else:
                # 多gpu模式
                args.dist_train = True
                args.ranks_num = len(gpu_ranks)
                print("采用分布式的方法进行训练")
        else:
            if world_size >= 1:
                print("无可用的GPU, 已自动转成CPU模式")
            # CPU模式
            args.dist_train = False
            args.single_gpu = False
    return

def load_hyperparam(default_args, config_path=None):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    if config_path is None:
        return default_args
    with open(config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)

    # if "deepspeed" in default_args.__dict__:
    #     with open(default_args.deepspeed_config, mode="r", encoding="utf-8") as f:
    #         default_args.deepspeed_config_param = json.load(f)

    default_args_dict = vars(default_args)

    command_line_args_dict = {k: default_args_dict[k] for k in [a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)]}
    default_args_dict.update(config_args_dict)
    default_args_dict.update(command_line_args_dict)
    args = Namespace(**default_args_dict)

    return args

def get_total_params_num(args):
    """
    提供基础模型里参数量的计算，只是为了方便理解而写的，得到真实的参数量请用get_total_params_num_from_model()
    """
    # 从原始的one-hot encoding输入到embedding layer的参数量
    emb_layer = (len(args.tokenizer.vocab) + args.max_seq_length + 3) * args.emb_size
    # 一个transformer里从embedding layer到Q,K,V连接层的参数量
    qkv_layer = args.emb_size * (args.hidden_size/args.heads_num) * 3 * args.heads_num
    # 一个transformer里layernorm层的参数量, 这里的2代表layernorm里gamma和beta
    layernorm = args.hidden_size * 2 + args.hidden_size * 2
    # 得到feedforward层的参数量
    feedforward_layer = args.hidden_size * args.feedforward_size + args.feedforward_size * args.hidden_size
    # 得到所有layer的transformer的参数量
    transformer_layer = (qkv_layer + layernorm + feedforward_layer) * args.layers_num
    # MLM的参数量
    mlm = args.hidden_size * args.hidden_size + args.hidden_size * len(args.tokenizer.vocab) + args.hidden_size * 2
    # NSP的参数量
    nsp = args.max_seq_length * args.hidden_size + args.hidden_size * 2
    return int(emb_layer + transformer_layer + mlm + nsp)


def get_total_params_num_from_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return

