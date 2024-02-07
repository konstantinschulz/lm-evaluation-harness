# Basic Info

The sbatch-script `run_megatron_server_client.sbatch` is designed to perform evaluation of a set of \
model checkpoints for ablation purposed, launching Megatron-LM inference server / LM-eval-harness client pairs.

# Setup

Make sure `run_server_no_opt.py` is present in the same directory as `run_megatron_server_client.sbatch`.\
Obtain a list of model checkpoint locations (`checkpoints.txt`) and specify its directory (see below).\
Adjust the working directory and cache paths in `run_megatron_server_clients.sbatch`.

# Usage

It is recommended to call the script through a wrapper such as `run_checkpoints.sh`.\
In there, specify the location of `checkpoints.txt`, the task list, desired output location and batch size as well as the checkpoint iteration.\
If the latter is not set, it defaults to the last iteration as per `./latest_checkpointed_iteration.txt` in the checkpoint directory.

One may also run the script directly using

```
sbatch [--array=0-N] [other_sbatch_options] run_megatron_server_client.sbatch /path/to/checkpoints.txt "list,of,tasks" /path/to/output/directory batch_size [checkpoint_iter]
```
The results will be stored as a single .json per model checkpoint containing results for all selected tasks, in the directory provided by  `/path/to/output/directory`.\
Stout and stderr are written to `./logs/eval-harness-%A_%a.out` and `./logs/eval-harness-%A_%a.err` by default.

# Remarks

Script timeout is set at 10 hours by default. Note that this timer applies _per checkpoint_.\
Further note that depending on the current slurm configuration, only 4 checkpoints will be evaluated at a time (e.g. on develbooster).
