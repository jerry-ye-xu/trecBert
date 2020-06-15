import os
import pandas as pd
import pysolr
import regex as re
import time

from collections import defaultdict

from typing import Any, List, Dict, Tuple, Sequence, Callable

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh import scoring

# from local_parser.extract_trial_data import store_ids, concatenate_returned_queries
# from local_parser.build_bert_ready_dataset import parse_topics, final_process_df

from local_parser.extract_trial_data import store_ids, concatenate_returned_queries
from local_parser.build_bert_ready_dataset import parse_topics, final_process_df

from query_expansion.expand_raw_baseline import query_expansion
from query_expansion.query_expansion_global_var import API_KEY

from BertSeqClassGlobalVar import global_var_cv

    ###############   PRE MODEL SCORING   ###############

RANKING = "ranking"
SUBSET_COLUMNS = [
    # "id", "topic",
    "brief_summary",
    "brief_title",
    "disease", "gene",
    "id_topic"
]

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

def get_doc_ids(ranking_dict: list, top_n: int, topics_subset: List[int]) -> Dict[str, List]:
    topic_doc_id_dict = defaultdict(list)

    #print_ranking_dict = ranking_dict[RANKING]["1"]
    #print(f"ranking_dict: {print_ranking_dict}")
    print(f"ranking_dict[RANKING].keys(): {ranking_dict[RANKING].keys()}")
    print(f"topics_subset: {topics_subset}")
    # time.sleep(10)
    for key in ranking_dict[RANKING].keys():
        # The key returned is a string object
        # whereas the topics_subset is an integer
        # print(f"topic_number is type {type(topics_subset[0])}")
        #print(f"key in ranking_dict[RANKING].keys() is type {type(key)}")
        if int(key) in topics_subset:
            # print("inside topics_subset")
            # time.sleep(3)
            idx = 0
            while idx < top_n:
                topic_doc_id_dict[key].append(ranking_dict[RANKING][key][idx][0])
                idx += 1

    # print(f"topic_doc_id_dict: {topic_doc_id_dict}")
    return topic_doc_id_dict

def check_missing_ids(
    topics_set: List[int],
    topic_dict_id_dict: Dict[str, List],
    trials_df: pd.DataFrame) -> List[str]:
    """

    Check for missing IDs one-by-one. This is very useful if
    the labelled datasets is missing something.

    Now, we don't need the labels to do the re-ranking as we will be
    outputting a pre-softmax that will be added to the preranking score.

    """
    missing_doc_id = []
    missing_topics = []
    # We filter by topics_set instead of all the keys in the dictionary
    # for key in topic_dict_id_dict.keys()):
    for key in [str(t) for t in topics_set]:
        for doc_id in topic_dict_id_dict[key]:
            if trials_df[(trials_df["id"] == doc_id) & (trials_df["topic"] == int(key))].shape[0] == 0:
                print(f"{doc_id} not found in DataFrame!")
                missing_doc_id.append(doc_id)
                missing_topics.append(key)
    return missing_doc_id, missing_topics


def create_id_list(
    topics_set: List[int],
    topic_dict_id_dict: Dict[str, List]):
    """
    Instead of checking missing ids one-by-one, we'll just build it from scratch.

    This works well since Lucene is MUCH faster.
    """
    doc_id_arr = []
    topics_arr = []
    # We filter by topics_set instead of all the keys in the dictionary
    # for key in topic_dict_id_dict.keys()):
    for key in [str(t) for t in topics_set]:
        for doc_id in topic_dict_id_dict[key]:
                # print(f"Appending {doc_id}")
                doc_id_arr.append(doc_id)
                topics_arr.append(key)
    return doc_id_arr, topics_arr


def get_missing_ids(
    missing_ids: List[str],
    index: str, qf: str,
    return_fields: List[str],
    task: str,
    num_procs: int,
    parallel: bool) -> Dict[str, Any]:

    res = store_ids(
        t_ids=missing_ids,
        index=index, qf=qf,
        return_fields=return_fields, task=task,
        bm25f_params={}, num_procs=num_procs,
        parallel=parallel
    )
    return res


def create_df_for_all(topic_doc_ids_dict, topic_col):
    df_base = []
    for key in topic_doc_ids_dict.keys():
        tmp_df = pd.DataFrame()
        topic_arr = [key] * len(topic_doc_ids_dict[key])

        tmp_df = pd.DataFrame(
            zip(topic_arr, topic_doc_ids_dict[key]),
            columns=[topic_col, "id"]
        )
        df_base.append(tmp_df)

    df_base_concat = pd.concat(df_base)

    df_base_concat[f"id_{topic_col}"] = df_base_concat["id"] + "_" + df_base_concat[topic_col]

    return df_base_concat


def process_df_missing(res, missing_topics, topic_col, topics_path, topics_xml):
    df = concatenate_returned_queries(res)

    print("concatenate_returned_queries(res) inside process_df_missing")
    # print(df)

    df_missing = pd.DataFrame(df)
    df_missing[topic_col] = missing_topics
    df_missing[topic_col] = df_missing[topic_col].astype(int)

    topics_dict = parse_topics(topics_path, topics_xml)

    ##### COMBINING TOPICS AND CLINICAL TRIALS #####

    # print(f"topics_dict: {topics_dict}")

    df_missing["topic_info"] = df_missing[topic_col].map(topics_dict)
    print("df_missing.head() in process_df_missing")
    print(df_missing.head())

    max_top_num = max(missing_topics)
    min_top_num = min(missing_topics)

    print(f"max_top_num: {max_top_num}")
    print(f"min_top_num: {min_top_num}")

    df_missing.to_pickle(f"./debug_data/df_missing_ingest_pipeline_{min_top_num}_{max_top_num}.pickle")

    return df_missing


def build_dataset_for_bert_cv(
    data, top_n, topics_set,
    subset_columns, return_fields,
    idx_paths, df_full_path,
    df_missing_save_path, fields_to_keep,
    base_ranker_type,
    # train_idx_set, test_idx_set,
    kfold_cv, topics_path, topics_xml,
    topic_col, topic_year_breakpoints, list_years):
    """
    idx_paths: [path1, path2, ...]
    """
    if type(data) is str:
        pre = pd.read_pickle(data)
    else:
        pre = data

    # Not supporting Whoosh indexing for cross-validation
    ct_idx_2017 = pysolr.Solr(idx_paths[0], timeout=1200)
    ct_idx_2019 = pysolr.Solr(idx_paths[1], timeout=1200)

    qf = "id^1"
    bm25f_params={}
    # year = year
    task = "clinical"
    num_procs = 1
    parallel = False
  
    # print(pre[0])
    print(topics_set)
    print("get_doc_ids parameters")
    # time.sleep(10)

    t_d_dict = get_doc_ids(pre[0], top_n, topics_set)

    df_full = pd.read_pickle(
        df_full_path
    )

    # split topics_set into whether you need
    # 2017 or 2019 index.

    orig2yrtop_dict = pd.read_pickle(global_var_cv["orig2yrtop_dict_path"])
    df_full["year_topic"] = df_full["year"].astype(str) + "_" + df_full["topic"].astype(str)
    df_full["topics_all"] = df_full["year_topic"].map(orig2yrtop_dict)
    df_full["topics_all"].astype(int)

    df_full = df_full[
        df_full["topics_all"].isin(topics_set)
    ]
    print(f"t_d_dict: {t_d_dict}")
    print(df_full.head())
    time.sleep(15)
    # Create a DataFrame to hold all
    # doc_id + topic combinations
    # this still works if topics are re-indexed from
    # 0 - 120 (removing the need for "year")
    df_base = create_df_for_all(t_d_dict, topic_col)
    df_base["id"] = df_base["id"].astype(str)
    df_base[topic_col] = df_base[topic_col].astype(str)

    # df_full["id"] = df_full["id"].astype(str)
    # df_full["topic"] = df_full["topic"].astype(str)
    # df_full["id_topic"] = df_full["id"] + "_" + df_full["topic"]

    topics_2017 = [t for t in topics_set if t <= 80]
    topics_2019 = [t for t in topics_set if t > 80]

    # df_2017 = df_full[df_full["topics_all"].isin(topics_2017)]
    # df_2019 = df_full[df_full["topics_all"].isin(topics_2019)]

    # missing_ids_2017, missing_topics_2017 = check_missing_ids(topics_set, t_d_dict, df_2017)

    # missing_ids_2019, missing_topics_2019 = check_missing_ids(topics_set, t_d_dict, df_2019)

    print(f"topics_2017: {topics_2017}")
    print(f"topics_2019: {topics_2019}")
    # time.sleep(10)

    df_missing_arr = []

    if len(topics_2017) > 0:
        df_missing_2017 = build_df_missing_for_cv(
            topics_2017, t_d_dict,
            topics_path, topics_xml, topic_col,
            topic_year_breakpoints, subset_columns,
            list_years, ct_idx_2017, qf, return_fields,
            task, num_procs, parallel
        )
        df_missing_arr.append(df_missing_2017)

    if len(topics_2019) > 0:
        df_missing_2019 = build_df_missing_for_cv(
            topics_2019, t_d_dict,
            topics_path, topics_xml, topic_col,
            topic_year_breakpoints, subset_columns,
            list_years, ct_idx_2019, qf, return_fields,
            task, num_procs, parallel
        )
        df_missing_arr.append(df_missing_2019)

    df_concat = pd.concat(
        df_missing_arr,
        axis=0,
        join="inner"
    )
    df_concat[f"id_{topic_col}"] = df_concat["id"] + "_" + df_concat[topic_col]

    return df_base.merge(df_concat[subset_columns], on=f"id_{topic_col}", how="left")

def build_df_missing_for_cv(
    topics_set, t_d_dict,
    topics_path, topics_xml,
    topic_col,
    topic_year_breakpoints,
    subset_columns,
    list_years, ct_idx, qf,
    return_fields,
    task, num_procs,
    parallel):
    ids_arr, topics_arr = create_id_list(topics_set, t_d_dict)

    res = get_missing_ids(
        missing_ids=ids_arr,
        index=ct_idx, qf=qf,
        return_fields=return_fields,
        task=task, num_procs=num_procs,
        parallel=parallel
    )
    print("res in build_df_missing_for_cv")
    # print(res)

    print(f"topic_col: {topic_col}")
    # time.sleep(5)

    df_missing = process_df_missing(
        res, topics_arr,
        topic_col,
        topics_path, topics_xml,
    )

    df_missing = final_process_df(
        df_missing, year=None, has_breakpoints=True,
        topic_year_breakpoints=topic_year_breakpoints,
        list_years=list_years
    )

    df_missing[topic_col] = df_missing[topic_col].astype(str)
    df_missing["id"] = df_missing["id"].astype(str)
    df_missing[f"id_{topic_col}"] = df_missing["id"] + "_" + df_missing[topic_col]

    # HERE IS WHERE I DO THE QE
    if "qe_all" in subset_columns:
        df_missing = query_expansion(
            df_missing,
            {"api_key": API_KEY},
            fields_to_keep
        )

    print("df_missing after process_df_missing()")
    print(df_missing.columns)
    print(df_missing.head())

    return df_missing

def build_dataset_for_bert(
    data, year, top_n, topics_set,
    subset_columns, return_fields,
    idx_path, df_full_path,
    df_missing_save_path, fields_to_keep,
    base_ranker_type, topics_path,
    topics_xml, topic_col):

    if type(data) is str:
        pre = pd.read_pickle(data)
    else:
        pre = data

    if "solr" in idx_path:
        ct_idx = pysolr.Solr(idx_path, timeout=1200)
    else:
        ct_idx = open_dir(idx_path)

    qf = "id^1"
    bm25f_params={}
    # year = year
    task = "clinical"
    num_procs = 1
    parallel = False

    t_d_dict = get_doc_ids(pre[0], top_n, topics_set)

    df_full = pd.read_pickle(
        df_full_path
    )

    df_full = df_full[
        (df_full[topic_col].isin(topics_set)) &
        (df_full["year"] == year)
    ]
    print("df_full.head() after filter")
    print(df_full.head())
    # time.sleep(5)

    missing_ids, missing_topics = check_missing_ids(topics_set, t_d_dict, df_full)

    print(f"missing_ids: {missing_ids}")
    print(f"missing_topics: {missing_topics}")
    # time.sleep(5)

    # This should not be changed... a new path will force the cache to be rebuilt for all files.
    if not os.path.exists(df_missing_save_path):
        os.makedirs(df_missing_save_path)

    # Check if cache file exists, otherwise download from scratch
    if len(missing_ids) > 0:
        try:
            if "qe_all" in subset_columns:
                df_missing = pd.read_pickle(f"{df_missing_save_path}/df_missing_{year}_top_{top_n}_{topics_set}_{base_ranker_type}_qe")
            else:
                df_missing = pd.read_pickle(f"{df_missing_save_path}/df_missing_{year}_top_{top_n}_{topics_set}_{base_ranker_type}")

            print(f"loading file from {df_missing_save_path} with attributes:\nyear: {year}\ntop_n: top_{top_n}\ntopics_set: {topics_set}\nbase_ranker_type: {base_ranker_type}")
        except FileNotFoundError:
            res = get_missing_ids(
                missing_ids=missing_ids,
                index=ct_idx, qf=qf,
                return_fields=return_fields,
                task=task, num_procs=num_procs,
                parallel=parallel
            )

            print(f"res is {res}")
            print(f"missing_topics: {missing_topics}")
            print(f"year: {year}")

            df_missing = process_df_missing(
                res, missing_topics,
                topic_col,
                topics_path, topics_xml
            )
            df_missing = final_process_df(
                df_missing, year=year,
                has_breakpoints=False,
                topic_year_breakpoints=None,
                list_years=None
            )

            print("df_missing after process_df_missing()")
            print(df_missing.columns)
            print(df_missing.head())

        # fields_to_keep = [
        #     "disease", "disease_kbqe_pn", "disease_kbqe_syn",
        #     "gene", "gene_kbqe_syn", "gene_kbqe_other"
        # ]

        # HERE IS WHERE I DO THE QE
        if "qe_all" in subset_columns:
            df_missing = query_expansion(
                df_missing,
                {"api_key": API_KEY},
                fields_to_keep
            )

            df_missing.to_pickle(f"{df_missing_save_path}/df_missing_{year}_top_{top_n}_{topics_set}_{base_ranker_type}_qe")
        else:
            df_missing.to_pickle(f"{df_missing_save_path}/df_missing_{year}_top_{top_n}_{topics_set}_{base_ranker_type}")

    df_base = create_df_for_all(t_d_dict, topic_col)

    df_base[topic_col] = df_base[topic_col].astype(str)
    df_full[topic_col] = df_full[topic_col].astype(str)

    df_base["id"] = df_base["id"].astype(str)
    df_full["id"] = df_full["id"].astype(str)

    df_full[f"id_{topic_col}"] = df_full["id"] + "_" + df_full[topic_col]

    if len(missing_ids) > 0:
        df_missing[topic_col] = df_missing[topic_col].astype(str)
        df_missing["id"] = df_missing["id"].astype(str)

        df_missing["id_topic"] = df_missing["id"] + "_" + df_missing[topic_col]

        df_concat = pd.concat(
            [
                df_full[subset_columns],
                df_missing[subset_columns]
            ],
            axis=0,
            join="inner"
        )
        print("df_base.merge(df_concat, on=\"id_topic\", how=\"left\")")
        print(df_base.merge(df_concat, on="id_topic", how="left"))
        return df_base.merge(df_concat, on="id_topic", how="left")

    print("df_base.merge(df_full, on=\"id_topic\", how=\"left\")")
    print(df_base.merge(df_full, on="id_topic", how="left"))
    return df_base.merge(df_full, on="id_topic", how="left")

    ###############   POST MODEL SCORING   ###############

def find_idx(ranking_dict: list, topic: str, doc_id: str) -> int:
    """

    Returns index position of the array that holds the relevant doc_id

    """
    idx = 0
    for x in ranking_dict[0]["ranking"][topic]:
        if x[0] == doc_id:
            return idx
        idx += 1

# def find_max_top_n(df_missing_save_path):
#     max_top_n = 0
#     for file in os.listdir(f"df_missing_save_path"):
#     print(file)
#     match = re.search(r'[0-9]+', file)
#     if match:
#         print(match.group())
#         if max_top_n < int(match.group()):
#             max_top_n = int(match.group())

#     print(max_top_n)
#     return max_top_n


if __name__ == "__main__":
    year = 2017
    top_n = 50
    data = "./A2A4UMA/pre_rerank_data_files/result_sets_pre_full.pickle"
    whoosh_idx_path = "./A2A4UMA/indices/ct17_whoosh"
    df_full_path = "./data/trials_topics_combined_all_years.pickle"
    save_path = "./A2A4UMA/pre_rerank_data_files/df_base_for_bert_full.pickle"
    subset_columns = SUBSET_COLUMNS
    return_fields = RETURN_FIELDS

    df_base = build_dataset_for_bert(
        data, year, top_n,
        subset_columns,
        return_fields,
        whoosh_idx_path, df_full_path
    )
    df_base.to_pickle(save_path)

    build_intermediate_missing_df = False

    if build_intermediate_missing_df:
        year = 2017
        top_n = 100
        whoosh_idx_path = "./A2A4UMA/indices/ct17_whoosh"
        df_full_path = "./data/trials_topics_combined_all_years.pickle"
        save_path = "./data/df_missing_2017_top_100_qe.pickle"

        subset_columns = SUBSET_COLUMNS.append("qe_all")
        return_fields = RETURN_FIELDS
        df_base = build_dataset_for_bert(
            data, year, top_n,
            subset_columns,
            return_fields,
            whoosh_idx_path, df_full_path
        )
        df_base.to_pickle(save_path)


    # qf = "id^1"
    # bm25f_params={}
    # year = 2017
    # task = "clinical"
    # num_procs = 4
    # parallel = True

    # ct_17_path = "./A2A4UMA/indices/ct17_whoosh"
    # ct_17_idx = open_dir(ct_17_path)

    # return_fields = [
    #     "score",
    #     "id", "brief_summary", "brief_title",
    #     "minimum_age", "gender",
    #     "primary_outcome", "detailed_description",
    #     "keywords", "official_title",
    #     "intervention_type",
    #     "intervention_name",
    #     "intervention_browse",
    #     "condition_browse",
    #     "inclusion", "exclusion",
    # ]

    # top_n = 50

    # pre = pd.read_pickle(
    # "./A2A4UMA/pre_rerank_data_files/result_sets_pre.pickle"
    # )

    # df_full = pd.read_pickle(
    #     "./data/trials_topics_combined_all_years.pickle"
    # )

    # df_full = df_full[df_full["year"] == year]

    # t_d_dict = get_doc_ids(pre[0], top_n)

    # missing_ids, missing_topics = check_missing_ids(t_d_dict, df_full)

    # res = get_missing_ids(
    #     missing_ids=missing_ids,
    #     index=ct_17_idx, qf=qf,
    #     return_fields=return_fields,
    #     task=task, num_procs=num_procs,
    #     parallel=parallel
    # )

    # df_missing = process_df_missing(res, missing_topics, year)

    # df_base = create_df_for_all(t_d_dict)

    # df_base["topic"] = df_base["topic"].astype(int)
    # df_full["topic"] = df_full["topic"].astype(int)
    # df_missing["topic"] = df_missing["topic"].astype(int)

    # df_base["id"] = df_base["id"].astype(str)
    # df_full["id"] = df_full["id"].astype(str)
    # df_missing["id"] = df_missing["id"].astype(str)


    # df_missing["id_topic"] = df_missing["id"] + "_" + df_missing["topic"]
    # df_full["id_topic"] = df_full["id"] + "_" + df_full["topic"]

    # SUBSET_COLUMNS = [
    #     "id",
    #     "brief_summary",
    #     "brief_title",
    #     "disease", "gene",
    #     "id_topic"
    # ]
    # df_concat = pd.concat(
    #     [
    #         df_full[SUBSET_COLUMNS],
    #         df_missing[SUBSET_COLUMNS]
    #     ],
    #     axis=0,
    #     join="inner"
    # )

    # df_base_merge = df_base.merge(df_concat, on="id_topic", how="left")
    # print(df_base_merge.head())
    # for col in df_base_merge.columns:
    #     print(df_base_merge[col].isna().value_counts())
