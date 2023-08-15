import runpy
import argparse
import torch
import sys
from pathlib import Path

"""
Helper script to load model config from checkpoint file.
Loads model config and then calls Mgetron-LM server start script.
"""

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True) # Checkpoint path
    parser.add_argument("--checkpoint-iter-step", required=True)
    parser.add_argument("--megatron-path", required=True)

    cargs = parser.parse_args()
    
    data = torch.load(f"{cargs.checkpoint_path}/{cargs.checkpoint_iter_step}/mp_rank_00/model_optim_rng.pt")
    args = data["args"]

    sys.argv = ['',
                f"--nproc_per_node={args.pipeline_model_parallel_size * args.tensor_model_parallel_size}",
                "--nnodes=1",
                "--node_rank=0",
                "--master_addr=localhost",
                "--master_port=63543",
                f"{cargs.megatron_path}/tools/run_text_generation_server.py",
                f"--load={cargs.checkpoint_path}",
                f"--tokenizer-model={args.tokenizer_model}",
                f"--tokenizer-type={args.tokenizer_type}",
                f"--pipeline-model-parallel-size={args.pipeline_model_parallel_size}",
                f"--tensor-model-parallel-size={args.tensor_model_parallel_size}",
                f"--num-layers={args.num_layers}",
                f"--hidden-size={args.hidden_size}",
                f"--num-attention-heads={args.num_attention_heads}",
                f"--max-position-embeddings={args.max_position_embeddings}",
                f"--micro-batch-size={args.micro_batch_size}",
                f"--global-batch-size={args.micro_batch_size}",
                f"--seq-length={args.seq_length}",
                f"--out-seq-length={args.seq_length}",
                f"--temperature=0.8",
                f"--top_p=0.5",
                f"--seed={args.seed}",
                f"--position-embedding-type={args.position_embedding_type.name}",
                f"--max-tokens-to-oom=300000"
                ]

    if args.bf16:
        sys.argv.append("--bf16")

    if not args.add_position_embedding:
        sys.argv.append("--no-position-embedding")

    # Not enabling --reset-attention-mask and --reset-position-ids as advised due to bug
    # if args.reset_attention_mask:
    #     sys.argv.append("--reset-attention-mask")
    #
    # if args.reset_position_ids:
    #     sys.argv.append("--reset-position-ids")

    # Not enabling --use-flash-attn during inference as advised
    # if args.use_flash_attn:
    #     sys.argv.append("--use-flash-attn")

    # Other ignored parameters: --init-method-std, --recompute-activations, --train-samples

    runpy.run_module('torch.distributed.run', run_name='__main__')