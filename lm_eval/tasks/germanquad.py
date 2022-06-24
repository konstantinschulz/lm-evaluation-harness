"""
GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval
https://arxiv.org/abs/2104.12741

In order to raise the bar for non-English QA, we are releasing a high-quality, human-labeled German QA dataset consisting of 13 722 questions, incl. a three-way annotated test set. The creation of GermanQuAD is inspired by insights from existing datasets as well as our labeling experience from several industry projects. We combine the strengths of SQuAD, such as high out-of-domain performance, with self-sufficient questions that contain all relevant information for open-domain QA as in the NaturalQuestions dataset. Our training and test datasets do not overlap like other popular datasets and include complex questions that cannot be answered with a single entity or only a few words.

Homepage: https://www.deepset.ai/germanquad
"""
from typing import Any

import datasets
from math import exp

from datasets import Dataset

from lm_eval.base import rf
from .common import HFTask
from functools import partial
from packaging import version

_CITATION = """
@misc{möller2021germanquad,
      title={GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval}, 
      author={Timo Möller and Julian Risch and Malte Pietsch},
      year={2021},
      eprint={2104.12741},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references)[key]


class GermanQuAD(HFTask):
    VERSION = 1
    DATASET_PATH = "deepset/germanquad"
    DATASET_NAME = None
    TEST_DATASET: list[dict[str, Any]] = None
    TRAIN_DATASET: list[dict[str, Any]] = None

    # HF changed squad on us so we have to make sure we aren't running the old one
    assert version.parse(datasets.__version__) >= version.parse(
        "1.11.0"), "datasets v1.11.0 or later required for GermanQuAD"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.data["train"]

    def test_docs(self):
        return self.data["test"]

    def doc_to_text(self, doc):
        def get_qa_string():
            if not self.TRAIN_DATASET:
                self.TRAIN_DATASET = list(self.data["train"])
            if not self.TEST_DATASET:
                self.TEST_DATASET = list(self.data["test"])
            target_property: str = "context"
            context: str = doc[target_property]
            relevant_items = [x for x in self.TRAIN_DATASET if x[target_property] == context]
            if not relevant_items:
                relevant_items = [x for x in self.TEST_DATASET if x[target_property] == context]
            relevant_items.remove(doc)
            qa_strings: list[str] = [f"Question: {x['question']}\n\nAnswer: {x['answers']['text'][0]}" for x in
                                     relevant_items]  # [:5]
            return '\n\n'.join(qa_strings) + ''
        # 'Background: ' + doc['context'] + '\n\n' + 'Question: ' + doc['question'] + '\n\n' + 'Answer:'
        # return 'How long is the answer to the following question: ' + doc['question'] + '\n\n' + 'Answer:'
        # return 'Question: ' + doc['question'] + '\n\n' + 'Answer:'
        # return 'Background: ' + doc['context'] + '\n\n' + 'Question: ' + doc['question'] + '\n\n' + 'Answer:'
        # example_qa_string: str = get_qa_string()
        # return f"Background: {context}\n\n{example_qa_string}\n\nQuestion: {doc['question']}\n\nAnswer:"
        return 'Question: ' + doc['question'] + '\n\n' + 'Answer:'

    def doc_to_target(self, doc):
        answer_list = doc['answers']['text']
        if len(answer_list) > 0:
            answer = answer_list[0]  # len(answer_list[0])
        else:
            answer = 0  # 'unanswerable'
        return " " + str(answer)  # answer

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        continuation = rf.greedy_until(ctx, ['\n'])
        # there are no unanswerable questions in GermanQuAD
        # is_unanswerable = rf.loglikelihood(ctx, " " + "unanswerable")
        return continuation  # , is_unanswerable

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # continuation, (logprob_unanswerable, _) = results
        continuation = results
        # no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            'id': doc['id'],
            'prediction_text': continuation,
            # all questions have an answer in GermanQuAD
            'no_answer_probability': 0,  # no_answer_probability,
        }

        references = {
            'id': doc['id'],
            'answers': doc['answers'],  # str(len(doc['answers']['text'][0]))
        }

        return {
            'exact': (predictions, references),  # Exact match (the normalized answer exactly match the gold answer)
            'f1': (predictions, references),  # The F-score of predicted tokens versus the gold answer
            'HasAns_exact': (predictions, references),
            # Exact match (the normalized answer exactly match the gold answer)
            'HasAns_f1': (predictions, references),  # The F-score of predicted tokens versus the gold answer
            # 'NoAns_exact': (predictions, references),
            # Exact match (the normalized answer exactly match the gold answer)
            # 'NoAns_f1': (predictions, references),  # The F-score of predicted tokens versus the gold answer
            'best_exact': (predictions, references),  # Best exact match (with varying threshold)
            'best_f1': (predictions, references),  # Best F1 (with varying threshold)
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            'exact': partial(_squad_agg, 'exact'),  # Exact match (the normalized answer exactly match the gold answer)
            'f1': partial(_squad_agg, 'f1'),  # The F-score of predicted tokens versus the gold answer
            'HasAns_exact': partial(_squad_agg, 'HasAns_exact'),
            # Exact match (the normalized answer exactly match the gold answer)
            'HasAns_f1': partial(_squad_agg, 'HasAns_f1'),  # The F-score of predicted tokens versus the gold answer
            'NoAns_exact': partial(_squad_agg, 'NoAns_exact'),
            # Exact match (the normalized answer exactly match the gold answer)
            'NoAns_f1': partial(_squad_agg, 'NoAns_f1'),  # The F-score of predicted tokens versus the gold answer
            'best_exact': partial(_squad_agg, 'best_exact'),  # Best exact match (with varying threshold)
            'best_f1': partial(_squad_agg, 'best_f1'),  # Best F1 (with varying threshold)
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            'exact': True,  # Exact match (the normalized answer exactly match the gold answer)
            'f1': True,  # The F-score of predicted tokens versus the gold answer
            'HasAns_exact': True,  # Exact match (the normalized answer exactly match the gold answer)
            'HasAns_f1': True,  # The F-score of predicted tokens versus the gold answer
            'NoAns_exact': True,  # Exact match (the normalized answer exactly match the gold answer)
            'NoAns_f1': True,  # The F-score of predicted tokens versus the gold answer
            'best_exact': True,  # Best exact match (with varying threshold)
            'best_f1': True,  # Best F1 (with varying threshold)
        }
