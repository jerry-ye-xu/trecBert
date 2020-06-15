#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 11:13:01 2019

@author: xu081

Comments: This Python encapsulates the functions and experimental code in notebooks 1-3. Below is an example of how to extract the 2017 clinical trial data and save it in a few segmented pickle files (memory issues), which are then combined.
"""

import os
import glob
import multiprocessing as mp
import pandas as pd
import pickle
import time

import pysolr

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh import scoring

def extract_ids(path, file_name, topic_num_max):
    t_ids = set()
    for folder in glob.glob(f"{path}/{file_name}"):
        with open(folder) as file:
            for line in file:
                if int(line.split(sep=" ")[0]) > topic_num_max:
                    return list(t_ids)
                t_id = line.split(sep=" ")[2]
                t_ids.add(t_id)
    return list(t_ids)

def create_t_dict(t_ids, index, qf, return_fields, task, bm25f_params):
    return [{
        "text": t,
        "index": index,
        "qf": qf,
        "return_fields": return_fields,
        "task": task,
        "bm25f_params": bm25f_params
    } for t in t_ids]

def run_query_parallel(*args):
    """
    Queries TREC Abstracts/Trials and returns their fields.

    See `run_query` function for more details.
    """
    kwargs = args[0]

    text = kwargs["text"]
    index = kwargs["index"]
    task = kwargs["task"]
    if 'bmq25f_params' not in kwargs:
        bm25f_params = {}
    else:
        bm25f_params = kwargs["bm25f_params"]

#     kwargs.pop("text"); kwargs.pop("index"); kwargs.pop("bm25f_params")
    del kwargs["text"]
    del kwargs["index"]
    del kwargs["bm25f_params"]
    del kwargs["task"]

    for k in kwargs:
        print(k, kwargs[k])
    return run_query(
        text=text, index=index,
        bm25f_params=bm25f_params,
        task=task, **kwargs
    )

def run_query(text, index, bm25f_params={}, task="clinical", **kwargs):
    """
    Queries TREC Abstracts/Trials and returns their fields.

    Params
    ------

    text: Text to formulate query
    index: Whoosh.index object made from index.open_dir(<path-to-index>)
    bm25f_params: Tuned parameters for BM25f model

    **kwargs:
        qf: query field, search 'text' parameter in specified query fields
        return_fields: specify which fields to return
        size: Maximum results to return
        max_year: Filters returns up until 'max_year'
        check_input: print out your inputs (sanity checks)

    """

    if type(index) is pysolr.Solr:
        kwargs['verb'] = 1

    #     base_field = "brief_summary^1" if task is "clinical" else "text^1"
        qf = "brief_summary^1 text^1" if 'qf' not in kwargs else kwargs['qf']
        return_fields = ['id','score'] if 'return_fields' not in kwargs else kwargs['return_fields'] #return fields

        size = 1000 if 'size' not in kwargs else kwargs['size']
        max_year = 2016 if 'max_year' not in kwargs else kwargs['max_year']
        parser = 'edismax' if 'parser' not in kwargs else kwargs['parser']

        # if 'verb' in kwargs:
        #     print(text)
        #     print(qf)
        #     print(bm25f_params)
        if len(bm25f_params) > 0:
            bm25.set_params(**bm25f_params)

        q_params={"fl": ','.join(return_fields),
                  #"fq": "body_text_en:[* TO *] AND date_i:[* TO "+str(max_year)+"]",
                  "fq": "date_i:[* TO "+str(max_year)+"]",
                  #"pf": "abstract_text_en^1.2 title_text_en^2",
                  # "start": "1",
                  "rows": str(size),  # return maximum 1000 results,
                  "defType": parser
                  }
        if max_year == 0 or max_year >= 2016:
            q_params.pop('fq')
        if len(qf) > 0:
            q_params["qf"]=qf
        result = index.search(text, **q_params)

        with open("./debug_data/check_results.pickle", "wb") as f:
            pickle.dump(result, f)

        output = []
        for doc in result.docs:
            # print(doc.keys())
            results_row = {}
            # if "score" in doc.keys():
            results_row['score'] = doc["score"]
            for f in return_fields:
                if f not in results_row:
                    if f in doc:
                        results_row[f] = doc[f]
                    else:
                        results_row[f] = ''
            output.append(results_row)


        return output, return_fields

    else:
    #     base_field = "brief_summary^1" if task is "clinical" else "text^1"
        qf = "brief_summary^1 text^1" if 'qf' not in kwargs else kwargs['qf']
        return_fields = ['id','score'] if 'return_fields' not in kwargs else kwargs['return_fields'] #return fields

        size = 1000 if 'size' not in kwargs else kwargs['size']
        max_year = 0 if 'max_year' not in kwargs else kwargs['max_year']
    #    parser='edismax' if 'parser' not in kwargs else kwargs['parser']

        qf_fields = [s.split("^")[0] for s in qf.split()]
        qf_boosts = [1 if len(s.split("^")) == 1 else float(s.split("^")[1]) for s in qf.split()]

    #     print(f"qf_fields: {qf_fields}")
    #     print(f"qf_boosts: {qf_boosts}")

        qff = [f for f, b in zip(qf_fields, qf_boosts) if b != 0]
        qfb = [b for f, b in zip(qf_fields, qf_boosts) if b != 0]

        boost_dict = {}
        for f, b in zip(qff, qfb):
            boost_dict[f] = b

        check_input = False if 'check_input' not in kwargs else kwargs["check_input"]
        # if check_input:
        #     print(f"text: {text}")
        #     print(f"query fields: {qf}")
        #     print(f"boost_dict: {boost_dict}")

        output = []
        if len(bm25f_params) > 0:
            w = scoring.BM25F(**bm25f_params)
        else:
            w = scoring.BM25F()
            print('Default scoring')
        with index.searcher(weighting=w) as searcher:
            query = MultifieldParser(qff, index.schema,
                                     fieldboosts=boost_dict,
                                     group=OrGroup).parse(text)
            if max_year > 0:
                mask_q = QueryParser("year", index.schema).parse("date_i:["+str(max_year)+" to]")
                results = searcher.search(query, limit=size, mask=mask_q)
            else:
                results = searcher.search(query, limit=size)

    #         print("Returning results")
            for r in results:
                results_row = {}
                results_row['score'] = r.score
                for f in return_fields:
                    if f not in results_row:
                        if f in r:
                            results_row[f] = r[f]
                        else:
                            results_row[f] = ''
                output.append(results_row)
        return output, return_fields

def store_ids(t_ids, index, qf,
              return_fields, task,
              bm25f_params, num_procs,
              parallel=True):
    start = time.time()
    if parallel:
        tuples = create_t_dict(
            t_ids=t_ids, index=index,
            bm25f_params=bm25f_params,
            task=task, qf=qf,
            return_fields=return_fields
        )
    #     print(tuples[:1])
        start = time.time()
        pool = mp.Pool(processes=num_procs)
        res = pool.map(run_query_parallel, tuples)
    else:
        res = []
        for t in t_ids:
            tmp_res = run_query(
                text=t, index=index,
                bm25f_params=bm25f_params,
                task=task, qf=qf,
                return_fields=return_fields,
            )
            res.append(tmp_res)
    end = time.time()
    print(f"Time elapsed: {end - start}")

    return res

def store_as_segmented_pickle(t_ids, step,
                              index, qf,
                              return_fields,
                              task, bm25f_params,
                              num_procs, parallel,
                              pickle_path, desc
                             ):
    num_t_ids = len(t_ids)
    start = time.time()
    for i in range(0, num_t_ids, step):
        if (num_t_ids - i) >= step:
            res = store_ids(
                t_ids=t_ids[i:i+step], index=index,
                qf=qf, return_fields=return_fields,
                task=task, bm25f_params=bm25f_params,
                num_procs=num_procs, parallel=parallel
            )
        else:
            res = store_ids(
                t_ids=t_ids[i:], index=index,
                qf=qf, return_fields=return_fields,
                task=task, bm25f_params=bm25f_params,
                num_procs=num_procs, parallel=parallel
            )
        with open(f"{pickle_path}/file_{i}_{desc}", "wb") as f:
    #         print(res)
            pickle.dump(res, f)
    end = time.time()
    print(f"Time elapsed: {end - start}")

def combine_pickle(file_name, file_path, save_path):
#     for folder in glob.glob(f"{file_path}/{pickle_name}"):
#         with open(folder) as file:
#             pass
    df = [pd.read_pickle(folder) for folder in glob.glob(f"{file_path}/{file_name}")]

    df_arr = []

    print(f"len(df): {len(df)}")
    print(f"len(df[6]): {len(df[3])}")

    # for i in range(len(df)):
    #     df_arr_tmp = concatenate_returned_queries(df[i])
    #     df_arr += df_arr_tmp

    for i in range(len(df)):
        df_arr_tmp = []

        for j in range(len(df[i])):
            df_arr_tmp += df[i][j][0]

        # print(f"i, len(df[i][j]): {i}, {len(df[i][j])}")
        df_arr += df_arr_tmp

    pd.DataFrame(df_arr).to_pickle(save_path)

def concatenate_returned_queries(df):
    df_arr = []

    for j in range(len(df)):
        df_arr_tmp = []
        df_arr_tmp += df[j][0]

        # print(f"j, len(df[i][j]): {j}, {len(df[j])}")
        df_arr += df_arr_tmp

    return df_arr

if __name__ == "__main__":
    qf = "id^1"
    bm25f_params={}
    YEAR = "2017"
    TASK = "clinical"
    NUM_PROCS = 1
#     NUM_T_IDS = len(t_ids) # calculated inside of function
    STEP = 1000
    use_solr = True

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

    if use_solr:
        index_path = 'http://localhost:8983/solr/ct2017'
        index = pysolr.Solr(index_path, timeout=1200)
    else:
        ct_17_path = "../A2A4UMA/indices/ct17_whoosh"
        index = open_dir(ct_17_path)

    # index_path = 'http://localhost:8983/solr/ct2017'
    # solr_index = pysolr.Solr(index_path, timeout=1200)
    # res_solr = run_query(text="NCT02210559", index=solr_index, bm25f_params=bm25f_params, task="clinical", qf=qf, return_fields=return_fields)

    # ct_17_path = "../A2A4UMA/indices/ct17_whoosh"
    # whoosh_index = open_dir(ct_17_path)
    # res_whoosh = run_query(text="NCT02210559", index=whoosh_index, bm25f_params=bm25f_params, task="clinical", qf=qf, return_fields=return_fields)

    # Test code to extract 2017 data
    PATH = f"./data/pm_labels_{YEAR}"
    QREL_TRIAL_NDCG = "qrels_treceval*trials*2017*.txt"

    t_ids = extract_ids(PATH, QREL_TRIAL_NDCG, topic_num_max=50)
    print(len(t_ids))
    print(t_ids[:3])

    PICKLE_PATH = f"./data/trials_pickle_{YEAR}_lucene"
    if not os.path.exists(PICKLE_PATH):
        os.mkdir(PICKLE_PATH)

    # STEP=1000
    # NUM_PROCS=1
    store_as_segmented_pickle(
        t_ids=t_ids, step=STEP,
        index=index, qf=qf,
        return_fields=return_fields,
        task=TASK, bm25f_params=bm25f_params,
        num_procs=NUM_PROCS, parallel=False,
        pickle_path=PICKLE_PATH, desc=YEAR
    )

    FILE_NAME = f"file_*_{YEAR}"
    FILE_PATH = f"./data/trials_pickle_{YEAR}_lucene"
    SAVE_PATH = f"./data/trials_pickle_{YEAR}_lucene/all_trials_{YEAR}.pickle"
    combine_pickle(FILE_NAME, FILE_PATH, SAVE_PATH)
