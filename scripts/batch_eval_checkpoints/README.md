# Basic Info

The sbatch-script `run_checkpoints.sbatch` is designed to perform evaluation of all current model checkpoints for tokenizer ablation, launching Megatron-LM inference server / LM-eval-harness client pairs.

# Setup

Make sure `run_server.py` is present in the same directory as `run_checkpoints.sbatch`.

# Usage

Run the script using

```
sbatch [sbatch_options] run_checkpoints.sbatch -p /path/to/results/ [-t "list,of,tasks"] [-l /path/to/checkpoints.txt]
```
If -t is not provided, all tasks will be run.\
By default, the list of checkpoint paths to be evaluated is checkpoint_list.txt in the same directory as `run_checkpoints.sbatch`. \
The results will be stored as a single .json per model checkpoint containing results for all selected tasks, in the directory provided with -p.\
Standard output is written to `./logs/eval-harness-%j.out` and `./logs/eval-harness-%j.err` by default.

# Remarks

Script timeout is set at 10 hours by default. Note that this timer applies _per checkpoint_. Further note that depending on the current slurm configuration, only 4 checkpoints will be evaluated at a time.