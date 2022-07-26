"""
XNLI: Evaluating Cross-lingual Sentence Representations
https://arxiv.org/abs/1809.05053


XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification in 15 languages.
""" """

Homepage: https://github.com/facebookresearch/XNLI
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """@misc{conneau2018xnli,
      title={XNLI: Evaluating Cross-lingual Sentence Representations},
      author={Alexis Conneau and Guillaume Lample and Ruty Rinott and Adina Williams and Samuel R. Bowman and Holger Schwenk and Veselin Stoyanov},
      year={2018},
      eprint={1809.05053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class XNLIBase(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "xnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])


class XNLIDe(XNLIBase):
    DATASET_NAME = "de"  # German part of xnli

    def _process_doc(self, doc):
        return {
            "premise": doc["premise"],
            "hypothesis": doc["hypothesis"],
            "choices": ["Wahr", "Neutral", "Falsch"],  # The list of choices.
            "gold": doc[
                "label"
            ],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        print(doc)
        return "{}\nDaraus folgt: {} Wahr, Falsch oder Neutral?\nAntwort:".format(
            doc["premise"],
            doc["hypothesis"].strip()
            + ("" if doc["hypothesis"].strip().endswith(".") else "."),
        )
