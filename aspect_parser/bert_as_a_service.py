import os
import pandas as pd
import pysolr
import time

from extract_trial_data import store_as_segmented_pickle, combine_pickle

from bert_serving.client import BertClient

if __name__ == "__main__":
    YEARS = [2018]
    for YEAR in YEARS:
    #     qf = "id^1"
    #     bm25f_params={}
    #     TASK = "clinical"
    #     NUM_PROCS = 1
    #     STEP=1000
    # #     NUM_T_IDS = len(t_ids) # calculated inside of function
    #     use_solr = True
    #     return_fields = [
    #         "score",
    #         "id", "brief_summary", "brief_title",
    #         "minimum_age", "gender",
    #         "primary_outcome", "detailed_description",
    #         "keywords", "official_title",
    #         "intervention_type",
    #         "intervention_name",
    #         "intervention_browse",
    #         "condition_browse",
    #         "inclusion", "exclusion",
    #     ]

    #     if use_solr:
    #         if YEAR != 2019:
    #             index_path = 'http://localhost:8983/solr/ct2017'
    #         else:
    #             index_path = 'http://localhost:8983/solr/ct2019'
    #         index = pysolr.Solr(index_path, timeout=1200)
    #     else:
    #         if YEAR != 2019:
    #             ct_path = "../A2A4UMA/indices/ct17_whoosh"
    #         else:
    #             ct_path = "../A2A4UMA/indices/ct19_whoosh"
    #         index = open_dir(ct_path)

    #     CT_JUDGEMENT_PATH = f"./data/pm_labels_{YEAR}/clinical_trials_judgments_{YEAR}.csv"
    #     df_judge = pd.read_csv(CT_JUDGEMENT_PATH)
    #     df_judge = df_judge[df_judge["pm_rel_desc"] == "Human PM"]
    #     print(f"df_judge.shape: {df_judge.shape}")
    #     time.sleep(5)

    #     PICKLE_PATH = f"./data/aspect_data/judgements_pickle_{YEAR}/"
    #     if not os.path.exists(PICKLE_PATH):
    #         os.mkdir(PICKLE_PATH)

    #     t_ids = list(df_judge["trec_doc_id"])
    #     store_as_segmented_pickle(
    #         t_ids=t_ids, step=STEP,
    #         index=index, qf=qf,
    #         return_fields=return_fields,
    #         task=TASK, bm25f_params=bm25f_params,
    #         num_procs=NUM_PROCS, parallel=False,
    #         pickle_path=PICKLE_PATH, desc=YEAR
    #     )
        FILE_NAME = f"file_*_{YEAR}"
        FILE_PATH = f"./data/aspect_data/judgements_pickle_{YEAR}/"
        SAVE_PATH = f"./data/aspect_data/trials_judgement_combined_full_{YEAR}.pickle"
        combine_pickle(FILE_NAME, FILE_PATH, SAVE_PATH)

        doc_text = pd.read_pickle(SAVE_PATH)
        print(doc_text.head())

        doc_text["year"] = YEAR
        doc_text["brief_t_and_s"] = doc_text["brief_title"] + " " + doc_text["brief_summary"]

        bc = BertClient()

        dir_path = f"./data/aspect_data/judgements_word_vec_{YEAR}"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        wv_len = len(list(doc_text["brief_t_and_s"]))

        print("Encoding all fields...")
        doc_wv = bc.encode(list(doc_text["brief_t_and_s"]))
        doc_wv_dict = {}
        for i in range(wv_len):
            doc_wv_dict[doc_text["id"][i]] = doc_wv[i]
        save_path = f"{dir_path}/doc_wv_all_baas_scibert_{YEAR}.pickle"
        pd.to_pickle(doc_wv_dict, save_path)

        print("Encoding brief title field...")
        doc_wv = bc.encode(list(doc_text["brief_title"]))
        doc_wv_dict = {}
        for i in range(wv_len):
            doc_wv_dict[doc_text["id"][i]] = doc_wv[i]
        save_path = f"{dir_path}/doc_wv_title_baas_scibert_{YEAR}.pickle"
        pd.to_pickle(doc_wv_dict, save_path)

        print("Encoding brief summary field...")
        doc_wv = bc.encode(list(doc_text["brief_summary"]))
        doc_wv_dict = {}
        for i in range(wv_len):
            doc_wv_dict[doc_text["id"][i]] = doc_wv[i]
        save_path = f"{dir_path}/doc_wv_summary_baas_scibert_{YEAR}.pickle"
        pd.to_pickle(doc_wv_dict, save_path)