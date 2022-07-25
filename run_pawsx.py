import json
import torch.cuda
from lm_eval import evaluator
from lm_eval.models.gpt2 import HFLM


def evaluate_pawsx():
    device = "cuda:0"
    hflm: HFLM = HFLM(pretrained="malteos/gpt2-wechsel-german-ds-meg", device=device)
 
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=["pawsx"],
        num_fewshot=10,
        batch_size=1,
        device=device,
        no_cache=True,
        limit=10,
        description_dict={},
    )
    dumped = json.dumps(results, indent=2, default=lambda x: None)
    print(dumped)
    with open("output.json", "w") as f:
        f.write(dumped)
    print(evaluator.make_table(results))


print("CUDA available: ", torch.cuda.is_available())
evaluate_pawsx()