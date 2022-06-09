import json
import torch.cuda
from lm_eval import evaluator
from lm_eval.models.gpt2 import HFLM


def evaluate_germanquad():
    device = "cuda:0"
    hflm: HFLM = HFLM(pretrained="malteos/gpt2-wechsel-german-ds-meg", device=device)
    # device = "cpu"
    # hflm: HFLM = HFLM(pretrained="malteos/gpt2-xl-wechsel-german", device=device)
    # dbmdz/german-gpt2 yongzx/gpt2-finetuned-oscar-de benjamin/gpt2-wechsel-german facebook/xglm-1.7B
    # facebook/xglm-564M facebook/xglm-1.7B
    results = evaluator.simple_evaluate(
        model=hflm,
        # model_args=args.model_args,
        tasks=["germanquad"],
        num_fewshot=10,  # 10 25
        batch_size=1,
        device=device,
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


print("CUDA available: ", torch.cuda.is_available())
evaluate_germanquad()
