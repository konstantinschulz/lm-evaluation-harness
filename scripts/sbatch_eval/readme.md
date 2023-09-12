
# Basic Info

The sbatch-script `run_megatron_server.sbatch` is designed to perform evaluation of all current model checkpoints for tokenizer ablation, launching Megatron-LM inference server / LM-eval-harness client pairs.

# Setup

Make sure `run_server_no_opt.py` is present in the same directory as `run_megatron_server.sbatch`.\
Obtain the current list of model checkpoint locations (`checkpoints.txt`) and place it in this directory.\
Adjust the working directory and cache paths in `run_megatron_server.sbatch`.

# Usage

Run the script using

```
sbatch [sbatch_options] run_megatron_server.sbatch /path/to/checkpoints.txt "list,of,tasks" /path/to/output/directory batch_size
```
By default, the list of checkpoint paths to be evaluated is checkpoint_list.txt in the same directory as `run_checkpoints.sbatch`. \
The results will be stored as a single .json per model checkpoint containing results for all selected tasks, in the directory provided by  `/path/to/output/directory`.\
Standard output is written to `./logs/eval-harness-%A_%a.out` and `./logs/eval-harness-%A_%a.err` by default.

# Remarks

Script timeout is set at 10 hours by default. Note that this timer applies _per checkpoint_. Further note that depending on the current slurm configuration, only 4 checkpoints will be evaluated at a time.