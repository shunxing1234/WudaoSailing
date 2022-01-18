EXPERIMENT_NAME=${MODEL_TYPE}-TNews
TASK_NAME=tnews
DATA_PATH="${DATA_ROOT}/toutiaonews"
MAX_SEQ_LEN=256

LR_SINGLE=1e-5
EPOCH_SINGLE=1
XXLARGE_EPOCH=1

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 200 \
             --log-interval 100 \
             --eval-interval 100 \
             --eval-iters 100"

PATTERN_IDS=(0 1)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16
