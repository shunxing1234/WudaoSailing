EXPERIMENT_NAME=${MODEL_TYPE}-AFQMC
TASK_NAME=afqmc
DATA_PATH="${DATA_ROOT}/lcqmc"
MAX_SEQ_LEN=256

LR_SINGLE=1e-5
EPOCH_SINGLE=10
XXLARGE_EPOCH=10

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 100 \
             --log-interval 100 \
             --eval-interval 100 \
             --eval-iters 100"

PATTERN_IDS=(0 1)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16
