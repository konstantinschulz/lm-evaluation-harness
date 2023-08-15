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
#SBATCH --array=1-1

# This script launches an array of server-client pair jobs for evaluation of Megatron-LM model checkpoints.

set -eo pipefail
set -x

# misc setup
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export MAX_JOBS=$SLURM_JOB_CPUS_PER_NODE

# (python) environment setup
module load Stages/2023 GCC CMake Ninja OpenMPI git Python CUDA PyTorch torchvision cuDNN TensorFlow pybind11/.2.9.2

export ROOT_DIR=/p/project/opengptx-elm/thellmann1/opengpt_2023/megatron-lm
export MEGATRON_LM_REPO=$ROOT_DIR/Megatron-LM
export VENV_DIR=$ROOT_DIR/env
export PYTHONPATH=$VENV_DIR/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$MEGATRON_LM_REPO:$PYTHONPATH

source $VENV_DIR/bin/activate

# get cl args
while getopts "l:p:t:" opt; do
  case $opt in
    l) MODEL_CHECKPOINTS="$OPTARG";;
    p) RESULTS_PATH="$OPTARG";;
    t) TASK_LIST="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1;;
  esac
done

if [ -z "$MODEL_CHECKPOINTS" ]; then MODEL_CHECKPOINTS=checkpoint_list.txt; fi

# load model checkpoint paths and extract model name from path
readarray -t models < $MODEL_CHECKPOINTS
MODEL_PATH=${models[$SLURM_ARRAY_TASK_ID]}
IFS="/" read -ra PTH <<< "$MODEL_PATH"
IFS="." read -ra NAME <<< "${PTH[-2]}"
model_name="${NAME[0]}"

# start server
SRUN_ARGS=" \
    --overlap \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID python run_server.py --checkpoint-path ${models[$SLURM_ARRAY_TASK_ID]} --checkpoint-iter-step iter_0053100 --megatron-path $MEGATRON_LM_REPO &
server_proc_id="$!"

ERROR_PATH="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# detect server readiness
api_addr=http://127.0.0.1:5000
while ! nc -z 127.0.0.1 5000; do   
  sleep 3
done

# prepare evaluation client
export HF_DATASETS_OFFLINE=1

EVAL_CMD="/p/project/opengptx-elm/thellmann1/workdir/lm_eval_setup/opengptx/lm-evaluation-harness/main.py \
    --model megatronlm \
    --model_args server_url=$api_addr,model_name=$model_name \
    --tasks "$TASK_LIST" \
    --output_path "$RESULTS_PATH/$model_name-out.json" \
    --no_cache \
    --batch_size 100 \
    "

# launch evaluation
srun --jobid $SLURM_JOB_ID python $EVAL_CMD

kill "$server_proc_id"
wait
