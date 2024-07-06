"""
Take in a YAML, and output all other splits with this YAML
"""

import yaml
import argparse
from itertools import permutations
import pycountry

API_URL = "https://datasets-server.huggingface.co/splits?dataset=facebook/belebele"

LANGS = [
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
    "bul_Cyrl",
    # "cat_Latn",
    # "ceb_Latn",
    "ces_Latn",
    # "cjk_Latn",
    # "ckb_Arab",
    # "crh_Latn",
    # "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    # "dik_Latn",
    # "dyu_Latn",
    # "dzo_Tibt",
    "ell_Grek",
    "eng_Latn",
    # "epo_Latn",
    "est_Latn",
    # "eus_Latn",
    # "ewe_Latn",
    # "fao_Latn",
    # "fij_Latn",
    "fin_Latn",
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
    "hun_Latn",
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
    "lit_Latn",
    # "lmo_Latn",
    # "ltg_Latn",
    # "ltz_Latn",
    # "lua_Latn",
    # "lug_Latn",
    # "luo_Latn",
    # "lus_Latn",
    "lvs_Latn",
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
    "nld_Latn",
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
    "pol_Latn",
    "por_Latn",
    # "prs_Arab",
    # "quy_Latn",
    "ron_Latn",
    # "run_Latn",
    # "rus_Cyrl",
    # "sag_Latn",
    # "san_Deva",
    # "sat_Olck",
    # "scn_Latn",
    # "shn_Mymr",
    # "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
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
    "swe_Latn",
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_name", required=True)
    parser.add_argument("--save_dir", required=True)
    return parser.parse_args()


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    kwargs = {f"alpha_{len(code)}": code}
    language_tuple = pycountry.languages.get(**kwargs)

    return language_tuple.name


if __name__ == "__main__":
    args = parse_args()

    for src_lang_code, tgt_lang_code in permutations(LANGS, 2):
        src_lang = code_to_language(src_lang_code[:3])
        tgt_lang = code_to_language(tgt_lang_code[:3])
        task_name = f"ogx_flores200-trans-{src_lang_code}-{tgt_lang_code}"

        yaml_dict = {
            "include": args.base_yaml_name,
            "task": task_name,
            "dataset_name": f"{src_lang_code}-{tgt_lang_code}",
            "doc_to_text": f"{src_lang} phrase: {{{{sentence_{src_lang_code}}}}}\n{tgt_lang} phrase:",
            "doc_to_decontamination_query": f"{{{{sentence_{src_lang_code}}}}}",
            "doc_to_target": f"{{{{sentence_{tgt_lang_code}}}}}",
        }

        file_save_path = f"{args.save_dir}/{task_name}.yaml"

        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
                sort_keys=False,
            )
