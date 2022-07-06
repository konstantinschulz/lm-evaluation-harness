# TODO: Remove all TODO comments once the implementation is complete.
"""
X-Stance: A Multilingual Multi-Target Dataset for Stance Detection
http://ceur-ws.org/Vol-2624/paper9.pdf

The x-stance dataset consists of more than 150 political questions and 67,000 comments by candidates.
It can be used to train and evaluate stance detection systems.
Comments are in German, French and Italian (for this task only the German subset has been selected). Questions are given in all three languages and English.
The data have been extracted from the Swiss voting advice platform Smartvote.

https://github.com/ZurichNLP/xstance
"""
import datasets
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from functools import partial
import numpy as np

_CITATION = """
@inproceedings{vamvas2020xstance,
    author    = "Vamvas, Jannis and Sennrich, Rico",
    title     = "{X-Stance}: A Multilingual Multi-Target Dataset for Stance Detection",
    booktitle = "Proceedings of the 5th Swiss Text Analytics Conference (SwissText)  16th Conference on Natural Language Processing (KONVENS)",
    address   = "Zurich, Switzerland",
    year      = "2020",
    month     = "jun",
    url       = "http://ceur-ws.org/Vol-2624/paper9.pdf"
}
"""

# Helper functions for aggregation (separate function for each metric)
def _xstance_agg_precision(key, items):
    references, predictions = zip(*items)
    precision_metric = datasets.load_metric("precision")
    return precision_metric.compute(references=references, predictions=predictions, average='macro', labels= np.unique(predictions))[key]

def _xstance_agg_recall(key, items):
    references, predictions = zip(*items)
    recall_metric = datasets.load_metric("recall")
    return recall_metric.compute(references=references, predictions=predictions, average='macro', labels= np.unique(predictions))[key]

def _xstance_agg_f1(key, items):
    references, predictions = zip(*items)
    f1_metric = datasets.load_metric("f1")
    return f1_metric.compute(references=references, predictions=predictions, average='macro', labels= np.unique(predictions))[key]

class XStance(Task):
    VERSION = 0
    DATASET_PATH = "x_stance"
    DATASET_NAME = None
    
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
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(map(self._process_doc, self.dataset["train"]))

            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])
        
    def _process_doc(self, doc):
        # Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # Returns only docs with German comments
        #Not needed if test should pass

        if doc["language"]=="de":
            return {"question":doc["question"], "comment":doc["comment"], "label":doc["label"]}
        #else:
        #    pass

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        # Query part consists of the question and comment part only (no label)
        if type(doc) is dict:
            return "QUESTION: "+ doc["question"]+ "\n\n"+ "COMMENT: "+ doc["comment"]+ "\n\n"+ "LABEL: "
        #else:
        #    pass

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        # Target is the label (i.e.'Favor' or 'Against'), which is appended to the string returned by doc_to_text
        if type(doc) is dict:
            target = doc["label"]
            return " " + target
        #else:
        #    pass

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # rf.loglikelihood as the task is a classification problem. For each document the model predicts loglikelihood for the correct label
        # ctx is the fully formatted fewshot example, i.e. K examples + comment to rate

        
        ll_favor = rf.loglikelihood(ctx, " "+"FAVOR")
        ll_against = rf.loglikelihood(ctx, " "+"AGAINST")

        return ll_favor, ll_against

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        #if doc is not None:
        pred = 0
        favor, against = results
        
        # Evaluation metrics will only work with numerical labels
        if favor[0] > against[0]:
            pred = 1
        else:
            pred = 0       
        true_label = doc["numerical_label"]
        
        # Save prediction and true label for evaluation
        predictions = {"id":doc["id"], "prediction":pred}

        y_true = {"id":doc["id"], "true label":true_label}

        return {"acc": pred==true_label, "precision":(true_label, pred), "recall":(true_label, pred), "f1":(true_label, pred)}
    
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

        return {"acc":mean, "precision": partial(_xstance_agg_precision, "precision"), 
                "recall" : partial(_xstance_agg_recall, "recall"), 
                "f1" : partial(_xstance_agg_f1, "f1")}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {"acc":True, "precision":True, "recall":True, "f1":True}
