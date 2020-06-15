import copy
import numpy as np
import os
import pandas as pd
import time 

from collections import defaultdict

from typing import Any, List, Dict, Tuple, Sequence, Callable

from bert_seq_class.evalFunc import softmax
from BertSeqClassGlobalVar import global_var_cv, global_var

RANKING = "ranking"
ID_TOPIC_COL = "id_topic"
BERT_COL = "bert_score"
BERT_SCORE_INDEX = 1 # binary output
# RANKING_DICT_TYPE = List[
#         Dict[str, List[Dict[str, List[str, float, str, int]]
#         ]
#     ]

"""

We are expecting BERT logit scores to be raw scores pre-softmax.

np.array([
    [-0.05, 1.2],
    [1.1, -0.68].
    ...
    [1.3, -0.11].
])

Where the first element is score for class = 0, and second element the score for class = 1.

If you have multiple classes, then you probably should take argmax.

"""

def create_id_topic_key(
    doc_id: np.array,
    topics_id: np.array,
    reg_pred_arr: np.array,
    scaling_function: Callable[[float], float]) -> Dict[str, float]:

    df_merge_key = [x + "_" + str(y) for x, y in list(zip(doc_id, topics_id))]
    # print("df_merge_key")
    # print(df_merge_key)

    # softmax_pred = softmax(reg_pred_arr)
    # bert_scores = [score[1] for score in softmax_pred]
    # bert_scores = [scaling_function(reg_pred_arr[i][BERT_SCORE_INDEX]) for i in range(len(reg_pred_arr))]

    return dict(zip(df_merge_key, reg_pred_arr))

def add_scores_to_df(
    df_base_merge: pd.DataFrame,
    bert_scores_dict: Dict[str, float],
    global_var: Dict[str, str]) -> None:

    df_base_merge[BERT_COL] = df_base_merge[global_var["id_topic"]].map(bert_scores_dict)

    # print(df_base_merge[[global_var["id_topic"], BERT_COL]].head())

    return df_base_merge

# Pure reranking of top_n documents retrieved by the base ranker.
def update_scores(
    start_n: int,
    top_n: int,
    pre_ranking_dict: List[Any],
    df_base_merge_score_col: pd.Series,
    topics_set: int) -> List[Any]:

    reranking_dict = copy.deepcopy(pre_ranking_dict)
    
    print("topics_set are")
    print(topics_set)
    time.sleep(5)
    # len(reranking_dict[0]['ranking']['1'])

    t_counter = 1
    for t_num in topics_set:
        max_score = reranking_dict[0]['ranking'][str(t_num)][0][1]
        print(f"max_score: {max_score}")
        print("len(reranking_dict[0][\'ranking\'][str(t_num)]")
        print(len(reranking_dict[0]['ranking']))
        #time.sleep(3)
        for j in range(len(reranking_dict[0]['ranking'][str(t_num)])):
            if j < start_n:
                reranking_dict[0][RANKING][str(t_num)][j][1] += 2
            elif (j >= start_n) and (j < top_n):
                idx = (t_counter-1)*top_n + j
                #print(f"t_counter: {t_counter}")
                #print(f"idx: {idx}")
                new_score = df_base_merge_score_col[idx] + 1
                reranking_dict[0][RANKING][str(t_num)][j][1] = new_score
            else:
                reranking_dict[0][RANKING][str(t_num)][j][1] /= max_score
        t_counter += 1

    return reranking_dict

# Adding BERT scores to base ranker scores, requires user to decide how much to weight each score.
def augment_scores(
    top_n: int,
    pre_ranking_dict: List[Any],
    df_base_merge_score_col: pd.Series,
    topics_set: int) -> List[Any]:

    reranking_dict = copy.deepcopy(pre_ranking_dict)

    print(topics_set)

    t_counter = 1
    for t_num in topics_set:
        for j in range(top_n):
            idx = (t_counter-1)*top_n + j
            reranking_dict[0][RANKING][str(t_num)][j][1] += df_base_merge_score_col[idx]
        t_counter += 1

def subset_qrels(test_set_topics, qrels_all_path):
    pass
