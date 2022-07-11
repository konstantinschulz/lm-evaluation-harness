"""
Ten Thousand German News Articles Dataset
Paper of the original One Million Posts Corpus: https://dl.acm.org/doi/10.1145/3077136.3080711

Homepage: https://tblock.github.io/10kGNAD/
Git: https://github.com/tblock/10kGNAD

The 10k German News Article Dataset consists of 10273 German language news articles from the online Austrian newspaper website DER Standard. 
Each news article has been classified into one of 9 categories by professional forum moderators employed by the newspaper. 
This dataset is extended from the original One Million Posts Corpus. The dataset was created to support topic classification in German 
because a classifier effective on a English dataset may not be as effective on a German dataset due to higher inflections and longer compound words. 
Additionally, this dataset can be used as a benchmark dataset for German topic classification.
"""
from lm_eval.base import Task

_CITATION = """
@InProceedings{Schabus2017,
  Author    = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
  Title     = {One Million Posts: A Data Set of German Online Discussions},
  Booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  Pages     = {1241--1244},
  Year      = {2017},
  Address   = {Tokyo, Japan},
  Doi       = {10.1145/3077136.3080711},
  Month     = aug
}"""


class GNAD10(Task):
    VERSION = 0
    DATASET_PATH = "gnad10"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        # No validation set
        return False

    def has_test_docs(self):
        return True


    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:

                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.


                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():

            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.


            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():

            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]





    """def _process_doc(self, doc):
>>>>>>> 31d14889a7ca0fb305d592d17f7bbafbb620cc1e
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
<<<<<<< HEAD
        return doc

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return ""

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = ""
=======
        return doc"""

    def doc_to_text(self, doc):
        return "text: "+ doc["text"]+ "\n\n"+ "label: "

    def doc_to_target(self, doc):
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = doc["label"]

        return " " + str(target)

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.
<<<<<<< HEAD

=======
>>>>>>> 31d14889a7ca0fb305d592d17f7bbafbb620cc1e
        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        return []

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document
<<<<<<< HEAD

=======
>>>>>>> 31d14889a7ca0fb305d592d17f7bbafbb620cc1e
        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        return {}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {}
