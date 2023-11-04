"""
PubMedQA: A Dataset for Biomedical Research Question Answering
https://arxiv.org/pdf/1909.06146.pdf

PubMedQA is a novel biomedical question answering (QA) dataset collected from
PubMed abstracts. The task of PubMedQA is to answer research questions with
yes/no/maybe (e.g.: Do preoperative statins reduce atrial fibrillation after
coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA
has 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA
instances. Each PubMedQA instance is composed of (1) a question which is either
an existing research article title or derived from one, (2) a context which is
the corresponding abstract without its conclusion, (3) a long answer, which is
the conclusion of the abstract and, presumably, answers the research question,
and (4) a yes/no/maybe answer which summarizes the conclusion.

Homepage: https://pubmedqa.github.io/
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, fertility


_CITATION = """
@inproceedings{jin2019pubmedqa,
    title={PubMedQA: A Dataset for Biomedical Research Question Answering},
    author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
    booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
    pages={2567--2577},
    year={2019}
}
"""


class Pubmed_QA(Task):
    VERSION = 0
    DATASET_PATH = "pubmed_qa"
    DATASET_NAME = "pqa_labeled"

    CHOICES = ["yes", "no", "maybe"]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            # HF is labelled as train but its really just for testing
            return self.dataset["train"]

    def doc_to_text(self, doc):
        ctxs = "\n".join(doc["context"]["contexts"])
        return "Abstract: {}\nQuestion: {}\nAnswer:".format(ctxs, doc["question"])

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"] + " " + "\n".join(doc["context"]["contexts"])

    def doc_to_target(self, doc):
        return " {}".format(doc["final_decision"])

    # TODO: vgl. TwoChoiceTask
    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns
        an iterable of Requests which will be sent to the LM.
        """
        ll_yes, _, req_stats = rf.loglikelihood_reqstats(ctx, " yes")
        ll_no, _, _ = rf.loglikelihood_reqstats(ctx, " no")
        ll_maybe, _, _ = rf.loglikelihood_reqstats(ctx, " maybe")

        return ll_yes, ll_no, ll_maybe, req_stats

    def process_results(self, doc, results):
        gold = doc["final_decision"]
        ll_yes, ll_no, ll_maybe, req_stats = results

        # TODO: pred umbenennen da es nicht direkt mit gold vergleichbar ist
        pred = np.argmax([ll_yes, ll_no, ll_maybe])

        token_ctx_count =  req_stats["tokens_ctx"]
        word_ctx_count =  req_stats["words_ctx"]

        return {
            "acc": self.CHOICES[pred] == gold,
            "fertility_ctx": {"tokens": token_ctx_count, "words": word_ctx_count, "include": True},
            "fertility_ctx_pos": {"tokens": token_ctx_count, "words": word_ctx_count, "include": self.CHOICES[pred] == gold},
            "fertility_ctx_neg": {"tokens": token_ctx_count, "words": word_ctx_count, "include": self.CHOICES[pred] != gold},
        }

    def aggregation(self):
        return {"acc": mean, "fertility_ctx": False, "fertility_ctx_pos": False, "fertility_ctx_neg": False,}

    def higher_is_better(self):
        return {
            "acc": True,
            "fertility_ctx": fertility,
            "fertility_ctx_pos": fertility,
            "fertility_ctx_neg": fertility
    }
