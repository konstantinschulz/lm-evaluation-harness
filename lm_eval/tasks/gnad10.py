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
import datasets
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from functools import partial
import numpy as np

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

# Helper functions for aggregation (separate function for each metric)
def _gnad10_agg_precision(key, items):
    references, predictions = zip(*items)
    precision_metric = datasets.load_metric("precision")
    return precision_metric.compute(references=references, predictions=predictions, average='macro', labels= np.unique(predictions))[key]

def _gnad10_agg_recall(key, items):
    references, predictions = zip(*items)
    recall_metric = datasets.load_metric("recall")
    return recall_metric.compute(references=references, predictions=predictions, average='macro', labels= np.unique(predictions))[key]

def _gnad10_agg_f1(key, items):
    references, predictions = zip(*items)
    f1_metric = datasets.load_metric("f1")
    return f1_metric.compute(references=references, predictions=predictions, average='macro', labels= np.unique(predictions))[key]
  
  
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
                self._training_docs = list(map(self._process_doc, self.dataset["train"]))
                #self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])
            #return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])
            #return self.dataset["test"]
          
    def _process_doc(self, doc):
      # Truncate examples which exceed the maximum token length for the model (1024)      
      text = doc["text"]
      
      if len(text.split(' ')) > 1023:
        c = 0
        text = ""
        for t in text.split(' '):
          while c < 1023:
            text += t + " "
            c += 2
        print(len(text))
        return {
            'text': text,
            'label': doc["label"],
        }
      print(len(text))
      return {
            'text': text,
            'label': doc["label"],
      }
      
    def doc_to_text(self, doc):
      # Truncate examples which exceed the maximum token length for the model (1024)
      """text = doc["text"]
      
      if len(text.split(' ')) > 1023:
        c = 0
        text = ""
        for t in text.split(' '):
          while c < 1024:
            text += t + " "
            c += 2"""
            
      return "text: "+ doc["text"]+ "\n\n"+ "label: "

    def doc_to_target(self, doc):
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = doc["label"]

        return " " + str(target)

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
        """ll_web = rf.loglikelihood(ctx, " "+str(0))
        ll_panorama = rf.loglikelihood(ctx, " "+str(1))
        ll_international = rf.loglikelihood(ctx, " "+str(2))
        ll_wirtschaft = rf.loglikelihood(ctx, " "+str(3))
        ll_sport = rf.loglikelihood(ctx, " "+str(4))
        ll_inland = rf.loglikelihood(ctx, " "+str(5))
        ll_etat = rf.loglikelihood(ctx, " "+str(6))
        ll_wissenschaft = rf.loglikelihood(ctx, " "+str(7))
        ll_kultur = rf.loglikelihood(ctx, " "+str(8))"""
        
        ll_web = rf.loglikelihood(ctx, " "+"Web")
        ll_panorama = rf.loglikelihood(ctx, " "+"Panorama")
        ll_international = rf.loglikelihood(ctx, " "+"International")
        ll_wirtschaft = rf.loglikelihood(ctx, " "+"Wirtschaft")
        ll_sport = rf.loglikelihood(ctx, " "+"Sport")
        ll_inland = rf.loglikelihood(ctx, " "+"Inland")
        ll_etat = rf.loglikelihood(ctx, " "+"Etat")
        ll_wissenschaft = rf.loglikelihood(ctx, " "+"Wissenschaft")
        ll_kultur = rf.loglikelihood(ctx, " "+"Kultur")
        
        return ll_web, ll_panorama, ll_international, ll_wirtschaft, ll_sport, ll_inland, ll_etat, ll_wissenschaft, ll_kultur

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        ll_web, ll_panorama, ll_international, ll_wirtschaft, ll_sport, ll_inland, ll_etat, ll_wissenschaft, ll_kultur = results
        
        #pred = max(ll_web[0], ll_panorama[0], ll_international[0], ll_wirtschaft[0], ll_sport[0], ll_inland[0], ll_etat[0], ll_wissenschaft[0], ll_kultur[0])
        
        pred = float('-inf')
        
        for i in results:
          if i[0] > pred:
            pred = results.index(i)
        
        # Evaluation metrics will only work with numerical labels
        """if pred == ll_web[0]:
          #pred = "Web"
          pred = 0
        elif pred == ll_panorama[0]:
          #pred = "Panorama"
          pred = 1
        elif pred == ll_international[0]:
          #pred = "International"
          pred = 2
        elif pred == ll_wirtschaft[0]:
          #pred == "Wirtschaft"
          pred = 3
        elif pred == ll_sport[0]:
          #pred == "Sport"
          pred = 4
        elif pred == ll_inland[0]:
          #pred == "Inland"
          pred = 5
        elif pred == ll_etat[0]:
          #pred == "Etat"
          pred = 6
        elif pred == ll_wissenschaft[0]:
          #pred == "Wissenschaft"
          pred = 7
        else:
          #pred = "Kultur"
          pred = 8"""
              
        true_label = doc["label"]
        
        return {"acc": pred==true_label, "precision":(true_label, pred), "recall":(true_label, pred), "f1":(true_label, pred)}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {"acc":mean, "precision": partial(_gnad10_agg_precision, "precision"), 
                "recall" : partial(_gnad10_agg_recall, "recall"), 
                "f1" : partial(_gnad10_agg_f1, "f1")}

    def higher_is_better(self):
        return {"acc":True, "precision":True, "recall":True, "f1":True}
