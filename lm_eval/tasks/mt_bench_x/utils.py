import json
import os

import datasets
import numpy as np

from lm_eval.utils import eval_logger


def process_docs(dataset: datasets.Dataset):
    eval_logger.warning(
        "This task is meant for chat-finetuned models, and may not give meaningful results for models other than `openai` or `anthropic` if `doc_to_text` in its YAML is not wrapped in the appropriate chat template string. This warning will be removed when chat templating support is added natively to local models"
    )
    def _helper(doc):
      # first turn only. Second turn needs generation of first turn
      doc["prompt"] = doc["turns"][0]
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object


def process_results(doc, results):
    eval_logger.warning(
        "This task is meant for chat-finetuned models, and may not give meaningful results for models other than `openai` or `anthropic` if `doc_to_text` in its YAML is not wrapped in the appropriate chat template string. This warning will be removed when chat templating support is added natively to local models"
    )
    doc["response"] = results
    question_id = doc["question_id"]
    filename = f"first_turn_response_{question_id}_{model_id}.json"
    with open(filename, "w") as f:
        json.dump(doc, f)



def gpt_4_as_a_judge(references, predictions, **kwargs):
    """Toxicity Perspective API is a metric which uses Perspective API to score the toxicity of a generated sentence.
    Scores above and including 0.5 are considered toxic based on the current practice in existing literature.
    Source: https://perspectiveapi.com/

    Lower is better
    """

    try:
        API_KEY = os.environ["OPENAI_API_KEY"]
    except KeyError:
        eval_logger.warning(
            "OPENAI_API_KEY is not set. If you are running the `mt_bench` task, please set this environment variable."
        )
        raise
    scores = []
    return np.mean(scores)
