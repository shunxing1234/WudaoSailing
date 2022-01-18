#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/mapping-data/GLM

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
#source $main_dir/config/model_glm_roberta_large.sh
source $main_dir/config/model_glm_large_chinese.sh

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

#en_data="/mapping-data/SST2/dev.tsv"
#eval_data="/mapping-data/SST2/train.tsv"
en_data="/mapping-data/SAT/dataset/dev.tsv"
eval_data="/mapping-data/SAT/dataset/train.tsv"
test_data="/mapping-data/SAT/dataset/test.tsv"

visible_devices="0"
num_categories=3
tuning_mode="ptuning"

#INDS="0 1 2 3"
INDS="0 1 2 3 4 5 6 7 8 9"
config_json="$script_dir/ds_config_ft.json"

for IND in $INDS
do
  echo ${IND}
  en_data="/mapping-data/SAT/dataset/full/train${IND}.tsv"
  eval_data="/mapping-data/SAT/dataset/full/val${IND}.tsv"
  test_data="/mapping-data/SAT/dataset/test.tsv"
  #en_data="/mapping-data/SAT/dataset/fewshot/train_fewshot${IND}.tsv"
  #eval_data="/mapping-data/SAT/dataset/fewshot/val_fewshot${IND}.tsv"
  #test_data="/mapping-data/SAT/dataset/test_fewshot.tsv"
  gpt_options=" \
         --experiment-name finetune-glm-zy \
         --model-parallel-size ${MP_SIZE} \
         --mode finetune \
         --train-iters 2000\
         --resume-dataloader \
         $MODEL_ARGS \
         --train-data ${en_data} \
         --valid-data ${eval_data} \
         --test-data ${test_data} \
         --distributed-backend nccl \
         --lr-decay-style cosine \
         --warmup .02 \
         --checkpoint-activations \
         --fp16 \
         --save-interval 2000 \
         --eval-interval 100 \
         --save /mapping-data/SAT/model \
         --split 1 \
         --strict-eval \
         --eval-batch-size 8
         --visible_devices ${visible_devices}
         --num_categories ${num_categories}
         --tuning_mode ${tuning_mode}
  "
         # --load  /root/checkpoints/pretrain-bert-mid-std-fulltrain12-02-06-10
         #  \       --sandwich-ln
         # --split 949,50,1 \
         # --load /root/checkpoints/pretrain-bert-mid11-28-15-38 \



  gpt_options="${gpt_options}
         --deepspeed \
         --deepspeed_config ${config_json} \
  "


  run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_glm_zy.py $@ ${gpt_options}"
  echo ${run_cmd}
  eval ${run_cmd}

  set +x
  wait
done
