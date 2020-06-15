#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Tues Dec 17 09:32:01 2019

@author: xu081

Comments: main file for executing all the scripts to generate the data. 
"""

from extract_trial_data import *
from build_bert_ready_dataset import *

qf = "id^1"
TASK = "clinical"
TOPIC_NUM_MAX = 50
NUM_PROCS = 6
STEP = 1000
PARALLEL = True

BM25F_PARAMS={
    'B':0.0, 'K1':0.455
    'brief_title_B':0.507, 
    'official_title_B':0.567, 
    'detailed_description_B':0.6, 
    'condition_browse_B':0.1, 
}

RETURN_FIELDS = [
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

def combine_years(years):
    
    for year in years:
        extract_trials_by_id = (year, BM25F_PARAMS)
        combine_trials_with_topics(year)
    
    df_arr = [pd.read_pickle(f"../data/trials_topics_combined_full_{year}.pickle") for year in years]
    
    pd.concat(df_arr, ignore_index=True)

def combine_trials_with_topics(year):
    """
    
    years: int, one of [2017, 2018 or 2019]
    
    """
    ##### CLINICAL TRIALS #####
    
    labels_path = f"../data/pm_labels_2018/qrels-treceval-clinical_trials-{year}-v2.txt"
    features_path = f"../data/trials_pickle_2018/all_trials_{year}.pickle"

    df_labels = pd.read_csv(
        labels_path,
        names=["topic", "_", "id", "label"], sep=" "
    )
    df_features = pd.read_pickle(features_path)

    df_features["id"].astype(str)
    df_labels["id"].astype(str)

    print(df_features.shape)
    print(df_labels.shape)

    # Raw combined
    df = df_features.merge(df_labels, left_on="id", right_on="id", how="inner")
    print(df.iloc[:2, ])
    
    SUBSET_COLUMNS = [
    "id", "brief_summary", "brief_title",
    "topic", "label"
    ]
    
    ##### TOPICS #####
    
    TOPICS_PATH = "../data/pm_labels_{year}"
    TOPICS_XML = "topics{year}.xml"
    
    topics_dict = parse_topics(TOPICS_PATH, TOPICS_XML)
    
    ##### COMBINING TOPICS AND CLINICAL TRIALS #####
    
    df["topic_info"] = df["topic"].map(topics_dict)
    df.head()
    
    # Be sure to specify index!
    df[["disease", "gene", "age_disease"]] = pd.DataFrame(
        df["topic_info"].tolist(),
        index=df.index
    )
    
    df["age"] = df["age_disease"].apply(find_age)
    df["gender"] = df["age_disease"].apply(find_gender)
    
    del df["age_disease"]
    del df["topic_info"]
    
    df["year"] = year
    
    df.to_pickle(f"../data/trials_topics_combined_full_{year}.pickle", index=False)

def extract_trials_by_year(year, bm25f_params={}):
    """ 
    Obtain id's from PM data files and use it to extract
    the clinical trials data, grouped by year.
    
    Params
    ------
    
    year: int, one of [2017, 2018 or 2019]
    
    """
    
    if year == 2019:
        ct_path = "../A2A4UMA/indices/ct19_whoosh"
        ct_idx = open_dir(ct_path)
    else:
        # 2017 and 2018 use the same Whoosh index. 
        ct_path = "../A2A4UMA/indices/ct17_whoosh"
        ct_idx = open_dir(ct_path)
    
    clinical_path_data = f"../data/pm_labels_{year}"
    qrel_trial_ndcg = f"qrels-treceval*trials*{year}*.txt"
    
    t_ids = extract_ids(
        clinical_path_data, 
        qrel_trial_ndcg, 
        topic_num_max=TOPIC_NUM_MAX
    )
    print(len(t_ids))
    print(t_ids[:3])
    
    pickle_path = f"../data/trials_pickle_{year}/"
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    store_as_segmented_pickle(
        t_ids=t_ids, step=STEP,
        index=ct_idx, qf=qf,
        return_fields=RETURN_FIELDS,
        task=TASK, bm25f_params=bm25f_params,
        num_procs=NUM_PROCS, parallel=PARALLEL,
        pickle_path=pickle_path, desc=year
    )

    file_name = f"file_*_{year}"
    file_path = f"../data/trials_pickle_{year}/"
    save_path = f"../data/trials_pickle_{year}/all_trials_{year}.pickle"
    combine_pickle(file_name, file_path, save_path)
    
    
if __name__ == "__main__":
    years = [2017, 2018]
    
    combine_years(years)
    