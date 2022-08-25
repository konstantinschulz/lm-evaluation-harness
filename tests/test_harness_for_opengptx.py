
'''
Test performance of implemented tasks

You can also run
python main.py --model gptx --model_args pretrained=dbmdz/german-gpt2  --tasks mlsum_de
 --output_path ./output_mlsum.json
to run outside the test mode

'''
from lm_eval import tasks
from lm_eval import models
from lm_eval import evaluator
import random
import pytest
import torch
import json
import time
from datetime import datetime


@pytest.mark.parametrize("eval_task", [("cnndm")])
def test_downstream_task_evaluator(
        eval_task,
        provide_description=False,
        num_fewshot=0,
        limit=4,
        bootstrap_iters=1000
    ):

    model_name = "gpt2"
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.get_model(model_name).create_from_arg_string("", {
    'batch_size': batch_size, 'device': device, 'pretrained':'gpt2'
    })

    task_dict = tasks.get_task_dict([eval_task])
    start_time = time.time()
    results = evaluator.evaluate(
        lm=model,
        task_dict=task_dict,
        provide_description=provide_description,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters
    ).get("results")
    end_time = time.time()

    output_path = f'output/results_{str(eval_task)}_{str(num_fewshot)}shots_{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json'
    configuration = {
        'model': model.gpt2.config._name_or_path,
        'batch_size': batch_size,
        'num_fewshot': num_fewshot,
        'running_time': end_time-start_time
    }
    outputs = {'Config':configuration, 'results':results}
    dumped = json.dumps(outputs, indent=4)
    print(dumped)
    if output_path:
        with open(output_path, "w") as f:
            f.write(dumped)