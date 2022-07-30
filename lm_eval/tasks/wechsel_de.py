"""
Validation set used by WECHSEL (German) based on a 0.1 * 4GB subset of `oscar/unshuffled_deduplicated_de`.

See
- https://arxiv.org/abs/2112.06598
- https://github.com/cpjku/wechsel
- https://huggingface.co/datasets/malteos/wechsel_de

"""
import re
import datasets

from lm_eval.base import PerplexityTask


class WechselDE(PerplexityTask):
    VERSION = 1
    DATASET_PATH = 'valid.random_1636.json.gz'  # use random 1% instead of full set
    DATASET_NAME = 'malteos/wechsel_de'
    dataset = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            self.DATASET_NAME,
            data_files={"test": self.DATASET_PATH},
            split="test",
            cache_dir=cache_dir,
            download_mode=download_mode
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _process_doc(self, doc):
        print(f'process doc: {doc}')
        return doc

    def test_docs(self):
        return self.dataset['text']

    def doc_to_target(self, doc):
        return doc

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))
