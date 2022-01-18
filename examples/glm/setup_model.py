import deepspeed
import torch
import mpu
from modeling_glm import GLMModel 
from downstream import GLMForMultiTokenCloze, GLMForMultiTokenClozeFast, GLMForSingleTokenCloze, GLMForSequenceClassification
from utils.prints import print_rank_0 
from model import PyTorchDistributedDataParallel as TorchDDP, DistributedDataParallel as LocalDDP
from train_utils.optimizers import get_optimizer_param_groups,get_optimizer
from fp16 import FP16_Module
from train_utils.schedulers import get_learning_rate_scheduler

def build_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Build the model."""
    print_rank_0('building GPT2 model ...')
    if args.pretrained_bert:
        if model_type == "multiple_choice":
            model = BertForMultipleChoice.from_pretrained(args.tokenizer_model_type,
                                                          cache_dir=args.cache_dir,
                                                          fp32_layernorm=args.fp32_layernorm,
                                                          fp32_embedding=args.fp32_embedding,
                                                          layernorm_epsilon=args.layernorm_epsilon)
        elif model_type == "classification":
            model = BertForSequenceClassification.from_pretrained(args.tokenizer_model_type,
                                                                  cache_dir=args.cache_dir,
                                                                  fp32_layernorm=args.fp32_layernorm,
                                                                  fp32_embedding=args.fp32_embedding,
                                                                  layernorm_epsilon=args.layernorm_epsilon,
                                                                  num_labels=num_labels)
        else:
            raise NotImplementedError
    else:
        output_predict, paralle_output = True, True
        if (model_type == "multiple_choice" or model_type == "classification") and not args.cloze_eval:
            output_predict = False
        if model_type is not None:
            paralle_output = False
        if spell_length is not None:
            print_rank_0(f"Continuous spell length {spell_length}")
        model = GLMModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale)
        if args.freeze_transformer:
            model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
        if model_type is not None:
            if model_type == 'multiple_choice':
                if args.cloze_eval:
                    if multi_token:
                        if args.fast_decode:
                            model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                        else:
                            model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
                    else:
                        model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
                else:
                    model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                         num_class=num_labels)
            elif model_type == 'classification':
                model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                     num_class=num_labels)
            elif model_type == 'generation':
                pass
            else:
                raise NotImplementedError(model_type)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if not args.deepspeed and (args.train_iters or args.epochs):
        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = TorchDDP(model, device_ids=[i], output_device=i,
                             process_group=mpu.get_data_parallel_group())
        elif args.DDP_impl == 'local':
            model = LocalDDP(model)
        else:
            print_rank_0("Skip DDP model")
    return model


def get_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Build the model."""
    print_rank_0('building GPT2 model ...')
    if args.pretrained_bert:
        if model_type == "multiple_choice":
            model = BertForMultipleChoice.from_pretrained(args.tokenizer_model_type,
                                                          cache_dir=args.cache_dir,
                                                          fp32_layernorm=args.fp32_layernorm,
                                                          fp32_embedding=args.fp32_embedding,
                                                          layernorm_epsilon=args.layernorm_epsilon)
        elif model_type == "classification":
            model = BertForSequenceClassification.from_pretrained(args.tokenizer_model_type,
                                                                  cache_dir=args.cache_dir,
                                                                  fp32_layernorm=args.fp32_layernorm,
                                                                  fp32_embedding=args.fp32_embedding,
                                                                  layernorm_epsilon=args.layernorm_epsilon,
                                                                  num_labels=num_labels)
        else:
            raise NotImplementedError
    else:
        output_predict, paralle_output = True, True
        if (model_type == "multiple_choice" or model_type == "classification") and not args.cloze_eval:
            output_predict = False
        if model_type is not None:
            paralle_output = False
        if spell_length is not None:
            print_rank_0(f"Continuous spell length {spell_length}")
        model = GLMModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale)
        if args.freeze_transformer:
            model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
        if model_type is not None:
            if model_type == 'multiple_choice':
                if args.cloze_eval:
                    if multi_token:
                        if args.fast_decode:
                            model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                        else:
                            model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
                    else:
                        model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
                else:
                    model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                         num_class=num_labels)
            elif model_type == 'classification':
                model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                     num_class=num_labels)
            elif model_type == 'generation':
                pass
            else:
                raise NotImplementedError(model_type)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if not args.deepspeed and (args.train_iters or args.epochs):
        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = TorchDDP(model, device_ids=[i], output_device=i,
                             process_group=mpu.get_data_parallel_group())
        elif args.DDP_impl == 'local':
            model = LocalDDP(model)
        else:
            print_rank_0("Skip DDP model")
    return model


def setup_model_and_optimizer(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Setup model and optimizer."""

    model = build_model(args, model_type=model_type, multi_token=multi_token, num_labels=num_labels,
                      spell_length=spell_length)
    param_groups = get_optimizer_param_groups(model)

    if args.train_data is not None or args.data_dir is not None and (args.epochs > 0 or args.train_iters > 0):
        if args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")

            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                args=args,
                mpu=mpu,
                dist_init_required=False
            )
        else:
            optimizer = get_optimizer(param_groups, args)
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    else:
        optimizer, lr_scheduler = None, None

    return model, optimizer, lr_scheduler
