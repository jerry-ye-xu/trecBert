import os
import pandas as pd
import pysolr

from collections import defaultdict
from whoosh.index import open_dir

from extract_trial_data import combine_pickle, store_as_segmented_pickle

def build_ct_judgement_df(path, index, pickle_path, prev_save, ctj_path, return_fields, doc_col, pm_col, year, treatment, use_solr=True, threshold=0.8, desc="ct_judgement"):

    ct_judge = process_ct_judgements(path, ctj_path, doc_col, pm_col, year, treatment, threshold)
    # print(ct_judge[doc_col].unique())

    # if os.path.exists(f"{pickle_path}/{prev_save}"):
    #     ct_judge_prev = pd.read_pickle(f"{pickle_path}/{prev_save}")
    #     t_id_list = list(set(ct_judge[doc_col].unique()).intersection(ct_judge_prev[doc_col].unique()))
    # else:
    #     t_id_list = list(set(ct_judge[doc_col].unique()))

    t_id_list = list(set(ct_judge[doc_col].unique()))

    if use_solr:
        num_procs = 1
        parallel = False
    else:
        num_procs = 4
        parallel = True

    print(f"len(t_id_list): {len(t_id_list)}")
    if len(t_id_list) > 0:
        store_as_segmented_pickle(
            t_ids=t_id_list,
            step=1000, index=index, qf="id^1",
            return_fields=return_fields, task="clinical",
            bm25f_params={}, num_procs=num_procs, parallel=parallel,
            pickle_path=pickle_path, desc=f"{desc}"
        )

def process_ct_judgements(path, ctj_path, doc_col, pm_col, year, treatment, threshold=None):
    ct_judge = pd.read_csv(path)
    ct_judge.loc[ct_judge[pm_col] == "Animal PM"] = "Not PM"
    print(ct_judge[pm_col].value_counts())

    ct_judge = treat_dup_cases(ct_judge, doc_col, pm_col, treatment, threshold)
    ct_judge.drop_duplicates(subset="trec_doc_id", inplace=True, keep="first")
    ct_judge["year"] = year
    ct_judge.to_pickle(f"{ctj_path}")
    return ct_judge

def treat_dup_cases(df, doc_col, pm_col, treatment, threshold=None):
    """
    treatment: ["all", "majority"]
    threshold: [0, 1], continuous value to determine cut-off for majority
    """

    doc_pm = df.groupby(by=[doc_col, pm_col]).apply(len).reset_index()
    pm_dup = doc_pm[doc_pm.duplicated(subset=[doc_col], keep=False)]

    if treatment == "all":
        return df[~df[doc_col].isin(pm_dup[doc_col].unique())]
    else:
        pm_dup.rename(columns={0: "pm_count"}, inplace=True)
        doc_id_to_remove, doc_id_to_update = count_dups(pm_dup, doc_col, pm_col, "pm_count", threshold)

        # update PM label to the majority class
        df_clean = df[~df[doc_col].isin(doc_id_to_remove)]
        print(df_clean[doc_col].nunique())
        update_pm_label(df_clean, doc_col, pm_col, doc_id_to_update)
        print(df_clean[doc_col].nunique())

        return df_clean

def count_dups(df, doc_col, pm_col, pm_count, threshold):
    cols = [doc_col, pm_col, pm_count]
    count_dict = defaultdict(dict)
    doc_pm_counts = [add_count(
        row[0],
        row[1],
        row[2],
        count_dict
    ) for row in df[cols].values]

    doc_id_to_remove = []
    doc_id_to_update = {}
    for key in count_dict.keys():
        ratio = count_dict[key]["Human PM"] / (count_dict[key]["Not PM"] + count_dict[key]["Human PM"])
        print(ratio)
        if ratio < threshold and ratio > (1 - threshold):
            print(f"appending {key}")
            doc_id_to_remove.append(key)
        else:
            majority = "Human PM" if (count_dict[key]["Human PM"] > count_dict[key]["Not PM"]) else "Not PM"
            doc_id_to_update[key] = majority
            print(f"NOT appending {key}, will need to update the minority pm_label.")

    return doc_id_to_remove, doc_id_to_update

def update_pm_label(df, doc_col, pm_col, doc_id_to_update):
    for key in doc_id_to_update.keys():
        df.loc[df[doc_col] == key, pm_col] = doc_id_to_update[key]

def add_count(doc_id, pm_label, count, count_dict):
    print(doc_id)
    print(pm_label)
    print(count)
    count_dict[doc_id][pm_label] = count

def check_dup_pm_label(df):
    check = df.groupby(by=["trec_doc_id", "pm_rel_desc"]).apply(len).reset_index()
    check2 = check[check.duplicated(subset=["trec_doc_id"], keep=False)]
    return check2.shape, check2

def fuse_ct_info_and_ctj(ct_info_path, ctj_path):
    ctj = pd.read_pickle(ctj_path)
    ctj = treat_dup_cases(ctj, "trec_doc_id", "pm_rel_desc", treatment="all")
    # check for duplicates with more than one pm_rel_desc label
    shape, df_check = check_dup_pm_label(ctj)
    print(shape)
    if shape[0]:
        print(df_check)
        raise ValueError("There's one doc id with different PM labels!")
    ctj.drop_duplicates(subset="trec_doc_id", inplace=True, keep="first")
    ctj_dict = ctj.set_index("trec_doc_id", verify_integrity=True).to_dict()["pm_rel_desc"]

    ct_info = pd.read_pickle(ct_info_path)
    ct_info = ct_info[ct_info["id"].isin(ctj.trec_doc_id.unique())]
    ct_info.drop_duplicates(subset="id", inplace=True, keep="first")

    ct_info["pm_rel_desc"] = ct_info["id"].map(ctj_dict)
    return ct_info
if __name__ == "__main__":

    for YEAR in [2017, 2018, 2019]:
        # YEAR = 2019
        DATA = f"./data/pm_labels_{YEAR}"
        JUDGEMENT_FILE = f"clinical_trials_judgments_{YEAR}.csv"

        FILE_NAME = f"file_*_ct_judgement_{YEAR}"
        FILE_PATH = f"./data/ct_judgement_{YEAR}"
        SAVE_PATH = f"./data/ct_judgement_{YEAR}/all_ct_judgement_trials_{YEAR}.pickle"
        PICKLE_PATH = f"./data/ct_judgement_{YEAR}"
        if not os.path.exists(PICKLE_PATH):
            os.mkdir(PICKLE_PATH)

        return_fields = [
            "score",
            "id", "brief_summary", "brief_title",
            "minimum_age", "gender",
            "primary_outcome", "detailed_description",
            "keywords", "official_title",
            "intervention_type",
            "intervention_name",
            "intervention_browse",
            "condition_browse",
            "inclusion", "exclusion",
        ]

        use_solr = True
        if YEAR == 2017:
            if use_solr:
                index_path = 'http://localhost:8983/solr/ct2017'
                index = pysolr.Solr(index_path, timeout=1200)
            else:
                ct_17_path = "./A2A4UMA/indices/ct17_whoosh"
                index = open_dir(ct_17_path)
        else:
            if use_solr:
                index_path = 'http://localhost:8983/solr/ct2019'
                index = pysolr.Solr(index_path, timeout=1200)
            else:
                ct_19_path = "./A2A4UMA/indices/ct19_whoosh"
                index = open_dir(ct_19_path)

        build_ct_judgement_df(
            f"{DATA}/{JUDGEMENT_FILE}", index, PICKLE_PATH,
            prev_save=SAVE_PATH, ctj_path=f"{FILE_PATH}/ctj_subset.pickle",
            return_fields=return_fields,
            doc_col="trec_doc_id", pm_col="pm_rel_desc", year=YEAR,
            treatment="majority", use_solr=use_solr, threshold=0.8, desc=f"ct_judgement_{YEAR}"
        )

        combine_pickle(FILE_NAME, FILE_PATH, SAVE_PATH)

    all_years = [2017, 2018, 2019]
    df_arr = [pd.read_pickle(f"./data/ct_judgement_{year}/all_ct_judgement_trials_{year}.pickle") for year in all_years]

    ALL_DATA_CT_INFO_PATH = f"./data/all_ct_judgement_trials_all_years.pickle"
    df_all_years = pd.concat(df_arr, ignore_index=True)
    df_all_years.to_pickle(ALL_DATA_CT_INFO_PATH)

    df_arr = [pd.read_pickle(f"./data/ct_judgement_{year}/ctj_subset.pickle") for year in all_years]

    for df in df_arr:
        print(df.columns)

    ALL_DATA_CTJ_PATH = f"./data/ctj_subset_all_years.pickle"
    df_all_years = pd.concat(df_arr, ignore_index=True)
    df_all_years.to_pickle(ALL_DATA_CTJ_PATH)

    df_final = fuse_ct_info_and_ctj(ALL_DATA_CT_INFO_PATH, ALL_DATA_CTJ_PATH)
    df_final.to_pickle("./data/all_ct_info_labelled.pickle")
