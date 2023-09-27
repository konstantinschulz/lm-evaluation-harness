#!/bin/bash
#SBATCH --job-name=eval-harness
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=48           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=opengptx-elm
#SBATCH --partition=booster
#SBATCH --output=logs/%x-%A_%a.out  # output file name
#SBATCH --error=logs/%x-%A_%a.err   # error file name

set -eo pipefail
set -x

export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=8

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export UCX_RC_TIMEOUT=4s
export MAX_JOBS=$SLURM_JOB_CPUS_PER_NODE

export TORCH_DISTRIBUTED_DETAIL=DEBUG
export NCCL_DEBUG=INFO


#export MEGATRON_OUTPUT_FIRST_LOGITS=1

module load Stages/2023 GCC CMake Ninja OpenMPI git Python CUDA PyTorch torchvision cuDNN TensorFlow pybind11/.2.9.2

# adjust these paths, ROOT_DIR should point to the install dir containing lm-evaluation-harness and Megatron-LM
export PATH_TO_FQUAD="/path/to/fquad1.0"
export ROOT_DIR="/path/to/installdir"

export MEGATRON_LM_REPO="$ROOT_DIR/Megatron-LM"
export VENV_DIR="$ROOT_DIR/pyenv_opengptx"
export PYTHONPATH="$VENV_DIR/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="$MEGATRON_LM_REPO:$PYTHONPATH"

source $VENV_DIR/bin/activate

MODEL_CHECKPOINTS=$1
TASKS=$2
OUTPUT_PATH=$3
BATCH_SIZE=$4
CHECKPOINT_ITER=$5

readarray -t models < $MODEL_CHECKPOINTS
MODEL_PATH=${models[$SLURM_ARRAY_TASK_ID]}
IFS="/" read -ra PTH <<< "$MODEL_PATH"
IFS="." read -ra NAME <<< "${PTH[-2]}"
model_name="${NAME[0]}"

if [ -z "$5" ]
  then
    read -r CHECKPOINT_ITER < "$MODEL_PATH/latest_checkpointed_iteration.txt"
fi

CHECKPOINT_ITER_STR="iter_$(printf "%07d" "$CHECKPOINT_ITER")"

SRUN_ARGS=" \
    --overlap \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

hostname

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun $SRUN_ARGS python run_server_no_opt.py --checkpoint-path ${models[$SLURM_ARRAY_TASK_ID]} --checkpoint-iter-step $CHECKPOINT_ITER_STR --megatron-path $MEGATRON_LM_REPO &
server_proc_id="$!"

api_scheme=http
api_host=127.0.0.1
api_port=5000
api_addr="$api_scheme://$api_host:$api_port"
while ! nc -z "$api_host" "$api_port"; do
  sleep 3
done

# adjust these paths if necessary
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE='/path/to/.cache/huggingface/datasets'
export HF_METRICS_CACHE='/path/to/.cache/huggingface/metrics'


JOB_DATE=$(date '+%Y%m%d-%H:%M%S')
JOB_IDENTIFIER=$SLURM_JOB_NAME-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID-$JOB_DATE

EVAL_CMD="$ROOT_DIR/lm-evaluation-harness/main.py \
    --model megatronlm \
    --model_args server_url=$api_addr,model_name=$model_name \
    --tasks "$TASKS" \
    --output_path "$OUTPUT_PATH"/$model_name-$JOB_IDENTIFIER.json \
    --no_cache \
    --batch_size "$BATCH_SIZE" \
    "

print_task_data() {
   echo Model: $model_name ;
   echo Tasks: $TASKS ;
   echo Batch Size: $BATCH_SIZE ;
   echo Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ;
   echo Date/Time: $JOB_DATE
}

FAIL_PATH=$OUTPUT_PATH/failures/$JOB_IDENTIFIER
SUCCESS_PATH=$OUTPUT_PATH/successes/$JOB_IDENTIFIER

srun python $EVAL_CMD && print_task_data > $SUCCESS_PATH || print_task_data > $FAIL_PATH

kill "$server_proc_id"
wait
