import json

import torch.cuda

from lm_eval import evaluator
from lm_eval.models.gpt2 import HFLM


def evaluate_germanquad():
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained="yongzx/gpt2-finetuned-oscar-de"),  # dbmdz/german-gpt2
        # model_args=args.model_args,
        tasks=["germanquad"],
        num_fewshot=10,
        batch_size=1,
        device="cuda:0",  # torch.device("cuda")
        no_cache=True,
        limit=10,
        description_dict={},
        # check_integrity=args.check_integrity
    )
    dumped = json.dumps(results, indent=2)
    print(dumped)
    with open("output.json", "w") as f:
        f.write(dumped)
    # print(
    #     f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
    #     f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    # )
    print(evaluator.make_table(results))


print("CUDA: ", torch.cuda.is_available())
evaluate_germanquad()
