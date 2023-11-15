# TODO: Remove all TODO comments once the implementation is complete.
"""
The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants
https://arxiv.org/abs/2308.16884

'We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants.
Significantly expanding the language coverage of natural language understanding (NLU) benchmarks,
this dataset enables the evaluation of text models in high-, medium-, and low-resource languages.
Each question is based on a short passage from the Flores-200 dataset and has four multiple-choice answers.
The questions were carefully curated to discriminate between models with different levels of general language comprehension.
The English dataset on its own proves difficult enough to challenge state-of-the-art language models.
Being fully parallel, this dataset enables direct comparison of model performance across all languages.
We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs).'

https://github.com/facebookresearch/belebele
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """@article{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants}, 
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      journal={arXiv preprint arXiv:2308.16884}
}
"""

_PROMPT_PATTERN = {
    "Passage":  "Passage",
    "Question": "Question",
    "Answer":   "Answer",
    "Choices":  "Choices"
}

class belebeleBase(MultipleChoiceTask):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "facebook/belebele"

    def __init__(self, lang: str = None):
        self.DATASET_NAME = self.lang_code = lang
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc,
                       self.dataset["train"].filter(
                           lambda x: x["dialect"]==self.lang_code))

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Passage: <passage>
            Question: <prompt>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """

            #TODO: decie whether choices should be included in prommpt (on bloom-1b5-clp: worse performance when included, 0.38 vs 0.27)
            """
                                 _PROMPT_PATTERN["Choices"] + "\n" +\
                     "".join([f"{keys[i-1]}: {doc[f'mc_answer{i}']}\n" for i in range(1,5)]) +\
            """
            prompt = _PROMPT_PATTERN["Passage"] + ": " + doc["flores_passage"] + "\n" +\
                     _PROMPT_PATTERN["Question"] + ": " + doc["question"] + "\n" +\
                     _PROMPT_PATTERN["Answer"] + ":"
            return prompt
        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),  # The query prompt.
            "choices": [doc[f"mc_answer{i}"] for i in range(1,5)],  # The list of choices.
            "gold": int(doc["correct_answer_num"])-1,  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        return doc["query"]

def create_translation_task(language, version=0):
    class belebele(belebeleBase):
        VERSION = version

        def __init__(self):
            super().__init__(language)

    return belebele

def construct_tasks():
    return {f"belebele_{lang}":create_translation_task(lang) for lang in _LANGUAGES}
        
_LANGUAGES = [
    # "ace_Arab",
    # "ace_Latn",
    # "acm_Arab",
    # "acq_Arab",
    # "aeb_Arab",
    # "afr_Latn",
    # "ajp_Arab",
    # "aka_Latn",
    # "als_Latn",
    # "amh_Ethi",
    # "apc_Arab",
    # "arb_Arab",
    # "arb_Latn",
    # "ars_Arab",
    # "ary_Arab",
    # "arz_Arab",
    # "asm_Beng",
    # "ast_Latn",
    # "awa_Deva",
    # "ayr_Latn",
    # "azb_Arab",
    # "azj_Latn",
    # "bak_Cyrl",
    # "bam_Latn",
    # "ban_Latn",
    # "bel_Cyrl",
    # "bem_Latn",
    # "ben_Beng",
    # "bho_Deva",
    # "bjn_Arab",
    # "bjn_Latn",
    # "bod_Tibt",
    # "bos_Latn",
    # "bug_Latn",
    # "bul_Cyrl",
    # "cat_Latn",
    # "ceb_Latn",
    # "ces_Latn",
    # "cjk_Latn",
    # "ckb_Arab",
    # "crh_Latn",
    # "cym_Latn",
    # "dan_Latn",
    "deu_Latn",
    # "dik_Latn",
    # "dyu_Latn",
    # "dzo_Tibt",
    # "ell_Grek",
    "eng_Latn",
    # "epo_Latn",
    # "est_Latn",
    # "eus_Latn",
    # "ewe_Latn",
    # "fao_Latn",
    # "fij_Latn",
    # "fin_Latn",
    # "fon_Latn",
    "fra_Latn",
    # "fur_Latn",
    # "fuv_Latn",
    # "gaz_Latn",
    # "gla_Latn",
    # "gle_Latn",
    # "glg_Latn",
    # "grn_Latn",
    # "guj_Gujr",
    # "hat_Latn",
    # "hau_Latn",
    # "heb_Hebr",
    # "hin_Deva",
    # "hne_Deva",
    # "hrv_Latn",
    # "hun_Latn",
    # "hye_Armn",
    # "ibo_Latn",
    # "ilo_Latn",
    # "ind_Latn",
    # "isl_Latn",
    "ita_Latn",
    # "jav_Latn",
    # "jpn_Jpan",
    # "kab_Latn",
    # "kac_Latn",
    # "kam_Latn",
    # "kan_Knda",
    # "kas_Arab",
    # "kas_Deva",
    # "kat_Geor",
    # "kaz_Cyrl",
    # "kbp_Latn",
    # "kea_Latn",
    # "khk_Cyrl",
    # "khm_Khmr",
    # "kik_Latn",
    # "kin_Latn",
    # "kir_Cyrl",
    # "kmb_Latn",
    # "kmr_Latn",
    # "knc_Arab",
    # "knc_Latn",
    # "kon_Latn",
    # "kor_Hang",
    # "lao_Laoo",
    # "lij_Latn",
    # "lim_Latn",
    # "lin_Latn",
    # "lit_Latn",
    # "lmo_Latn",
    # "ltg_Latn",
    # "ltz_Latn",
    # "lua_Latn",
    # "lug_Latn",
    # "luo_Latn",
    # "lus_Latn",
    # "lvs_Latn",
    # "mag_Deva",
    # "mai_Deva",
    # "mal_Mlym",
    # "mar_Deva",
    # "min_Arab",
    # "min_Latn",
    # "mkd_Cyrl",
    # "mlt_Latn",
    # "mni_Beng",
    # "mos_Latn",
    # "mri_Latn",
    # "mya_Mymr",
    # "nld_Latn",
    # "nno_Latn",
    # "nob_Latn",
    # "npi_Deva",
    # "nso_Latn",
    # "nus_Latn",
    # "nya_Latn",
    # "oci_Latn",
    # "ory_Orya",
    # "pag_Latn",
    # "pan_Guru",
    # "pap_Latn",
    # "pbt_Arab",
    # "pes_Arab",
    # "plt_Latn",
    # "pol_Latn",
    # "por_Latn",
    # "prs_Arab",
    # "quy_Latn",
    # "ron_Latn",
    # "run_Latn",
    # "rus_Cyrl",
    # "sag_Latn",
    # "san_Deva",
    # "sat_Olck",
    # "scn_Latn",
    # "shn_Mymr",
    # "sin_Sinh",
    # "slk_Latn",
    # "slv_Latn",
    # "smo_Latn",
    # "sna_Latn",
    # "snd_Arab",
    # "som_Latn",
    # "sot_Latn",
    "spa_Latn",
    # "srd_Latn",
    # "srp_Cyrl",
    # "ssw_Latn",
    # "sun_Latn",
    # "swe_Latn",
    # "swh_Latn",
    # "szl_Latn",
    # "tam_Taml",
    # "taq_Latn",
    # "taq_Tfng",
    # "tat_Cyrl",
    # "tel_Telu",
    # "tgk_Cyrl",
    # "tgl_Latn",
    # "tha_Thai",
    # "tir_Ethi",
    # "tpi_Latn",
    # "tsn_Latn",
    # "tso_Latn",
    # "tuk_Latn",
    # "tum_Latn"
    # "tur_Latn",
    # "twi_Latn",
    # "tzm_Tfng",
    # "uig_Arab",
    # "ukr_Cyrl",
    # "umb_Latn",
    # "urd_Arab",
    # "uzn_Latn",
    # "vec_Latn",
    # "vie_Latn",
    # "war_Latn",
    # "wol_Latn",
    # "xho_Latn",
    # "ydd_Hebr",
    # "yor_Latn",
    # "yue_Hant",
    # "zho_Hans",
    # "zho_Hant",
    # "zsm_Latn",
    # "zul_Latn",
]
