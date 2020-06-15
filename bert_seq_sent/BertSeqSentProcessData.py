import regex as re
import pandas as pd

from typing import Any, List, Dict, Tuple, Sequence, Optional

def process_validation_data_raw(df: pd.DataFrame, global_var: Dict[str, any]) -> pd.DataFrame:
    df[global_var["seq_b_baseline"]] = df["brief_title"] + ". " + df["brief_summary"]
    df[global_var["seq_a_baseline"]] = df["disease"] + ", " + df["gene"]

    return df

def split_into_bert_input(
    df_input: pd.DataFrame,
    seq_a_col_name: str,
    seq_b_col_name: str,
    validate: bool,
    use_qe: bool,
    global_var: Dict[str, Any]) -> pd.DataFrame:

    if use_qe:
        seq_a = [process_qe_array(x) for x in list(df_input[seq_a_col_name])]
    else:
        seq_a = [process_raw_array(x) for x in list(df_input[seq_a_col_name])]

    seq_b = None
    if seq_b_col_name is not None:
        seq_b = [x for x in list(df_input[seq_b_col_name])]

    labels = None
    if not validate:
        labels = list(df_input[global_var["label"]])
        attribute_seq = list(df_input[global_var["attrib_seq"]])
    else:
        # We only need TOPIC as we have filtered out year already
        attribute_seq = list(df_input[global_var["topic"]])
    doc_ids = list(df_input[global_var["doc_id"]])

    return seq_a, seq_b, labels, attribute_seq, doc_ids

def split_into_sent_bert_input(
    df_input: pd.DataFrame,
    seq_a_col_name: str,
    seq_b_col_name: str,
    validate: bool,
    use_qe: bool,
    global_var: Dict[str, Any]) -> pd.DataFrame:

    if use_qe:
        df_input[seq_a_col_name] = df_input[seq_a_col_name].apply(process_qe_array)
    else:
        df_input[seq_a_col_name] = df_input[seq_a_col_name].apply(process_raw_array)

    df_input[global_var["seq_all_sent"]] = df_input[seq_a_col_name] + " " + df_input[seq_b_col_name]

    df_input[global_var["seq_all_sent"]] = df_input[global_var["seq_all_sent"]].apply(lambda x: x.split(". "))
    df_input = df_input.explode(global_var["seq_all_sent"]).reset_index()

    print(df_input[df_input["id"] == "NCT01874665"])


    seq_a = [x for x in list(df_input[global_var["seq_all_sent"]])]

    labels = None
    if not validate:
        labels = list(df_input[global_var["label"]])
        attribute_seq = list(df_input[global_var["attrib_seq"]])
    else:
        # We only need TOPIC as we have filtered out year already
        attribute_seq = list(df_input[global_var["topic"]])
    doc_ids = list(df_input[global_var["doc_id"]])

    seq_b = None
    return seq_a, seq_b, labels, attribute_seq, doc_ids

def process_raw_array(text: str) -> str:
    pattern = '[^A-Za-z0-9]+'
    replace = ' '

    text = text.replace("-", "")
    text = re.sub(pattern, replace, text).strip()

    return text

def process_qe_array(qe_arr: List[List[str]]) -> List[str]:
    pattern = '[^A-Za-z0-9]+'
    replace = ' '

    qe_arr = [x.replace("-", "") for x in qe_arr]
    qe_arr = [re.sub(pattern, replace, x).strip() for x in qe_arr]
    qe_arr = [t for aug in qe_arr for t in aug.split(sep=" ")]
    qe_arr = " ".join(qe_arr)

    return qe_arr

def subset_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    topic_subset: List[int],
    global_var: Dict[str, Any]) -> (pd.DataFrame, pd.DataFrame):

    df_train = df_train[df_train[global_var["topic"]].isin(topic_subset)]
    df_test = df_test[df_test[global_var["topic"]].isin(topic_subset)]

    return df_train, df_test

def trim_qe(arr: List[str], n_chars: int) -> List[str]:
    trimmed_arr = []
    for e in arr:
        if len(e) <= n_chars:
            trimmed_arr.append(e)
#         trimmed_arr.append(e[:n_chars])
    return trimmed_arr

def remove_empty_strings(arr: List[str]):
    return [x for x in arr if x != '']

print("BertSeqClassProcessData refreshed")
