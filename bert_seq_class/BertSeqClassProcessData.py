"""
Functions for

1) Building BERT datasets for finetuning (binary class ONLY)
2) Build BERT test sets using preranking dictionary from A2A
3) Build reranking dictionary from BERT results to be ingested into A2A

"""

import regex as re
import pandas as pd

from sklearn.model_selection import KFold

from typing import Any, List, Dict, Tuple, Sequence, Optional

FINETUNING_PICKLE_PATH = "../data/trials_topics_combined_all_years.pickle"

# SUBSET_COLUMNS = [
#     "id", "year",
#     "gene", "disease",
#     "brief_summary", "brief_title",
#     "topic", "label"
# ]

# # Column names
# YEAR = "year"
# LABEL = "label"
# TOPIC = "topic"
# ATTRIB_SEQ = "ty_id"
# DOC_ID = "id"
# SEQ_A_BASELINE = "d_and_g"
# # Structure of SEQ_A_BASELINE
# """
# A single item in the column is
# "a_massive_string"
# """

# SEQ_B_BASELINE = "brief_t_and_s"
# # Structure of SEQ_A_EXPANSION
# """
# A single item in the column is List[List[str]]
# [
#     ["str", "str", ..., "str"],
#     ["str", "str", ..., "str"]
# ]
# """
# SEQ_A_EXPANSION = "qe_all"
# # N_CHARS_TRIM = 100
# USE_QE = False
# # We only expand the query
# # SEQ_B_EXPANSION = "bts_expand"

# global_var = {
#     "subset_columns": SUBSET_COLUMNS,
#     "year": "year",
#     "label": "label",
#     "topic": "topic",
#     "attrib_seq": "ty_id",
#     "doc_id": "id",
#     "seq_a_baseline": "d_and_g",
#     "seq_b_baseline": "brief_t_and_s",
#     "seq_a_expansion": "qe_all"
# }

# from BertSeqClassGlobalVar import global_var, global_var_cv

"""
Functions for

1) Building BERT datasets for finetuning (binary class ONLY)
2) Build BERT test sets using preranking dictionary from A2A
3) Build reranking dictionary from BERT results to be ingested into A2A

"""

# if USE_QE and SEQ_A_EXPANSION not in SUBSET_COLUMNS:
#     SUBSET_COLUMNS += [SEQ_A_EXPANSION]

from typing import Any, List, Dict, Tuple, Sequence, Optional

#########    BERT TRAINING DATA    #########

def process_baseline_raw(
    df: pd.DataFrame,
    use_qe: bool,
    n_chars_trim: int,
    global_var: Dict[str, Any]) -> pd.DataFrame:

    if use_qe and global_var["seq_a_expansion"] not in global_var["subset_columns"]:
        global_var["subset_columns"].append(global_var["seq_a_expansion"])

    df_input = df[global_var["subset_columns"]].copy()

    # Restrict to binary class
    df_input[global_var["label"]] = df_input[global_var["label"]].replace(to_replace=2, value=1)

    # build seq_a and seq_b tokens
    if use_qe:
        df_input[global_var["seq_b_baseline"]] = df_input["brief_title"] + " " + df_input["brief_summary"]
        df_input[global_var["seq_a_expansion"]] = df_input[global_var["seq_a_expansion"]].apply(lambda x: trim_qe(x, n_chars_trim))
        df_input[global_var["seq_a_expansion"]] = df_input[global_var["seq_a_expansion"]].apply(remove_empty_strings)
    else:
        df_input[global_var["seq_b_baseline"]] = df_input["brief_title"] + " " + df_input["brief_summary"]
        df_input[global_var["seq_a_baseline"]] = df_input["disease"] + " " + df_input["gene"]

    # New ID for multiple year data sets
    df_input["year_topic"] = df_input[global_var["year"]].astype(str) + "_" + df_input[global_var["raw_topic"]].astype(str)
    print("Inside process_baseline_raw(), checking for year_topic mapping, all years present. Data will be split into training and testing later.")
    print(df_input["year_topic"].iloc[:5])

    ty_id = sorted(df_input["year_topic"].unique())

    ty_id_map = dict(zip(ty_id, [i for i in range(len(ty_id))]))
    print(ty_id_map)

    # Add this to ID
    df_input[global_var["attrib_seq"]] = df_input["year_topic"].map(ty_id_map)
    print(df_input[[
              global_var["raw_topic"],
              global_var["year"],
              "year_topic",
              global_var["attrib_seq"],
              global_var["doc_id"]
          ]].head())

    return df_input

def process_validation_data_raw(df: pd.DataFrame, global_var: Dict[str, any]) -> pd.DataFrame:
    df[global_var["seq_b_baseline"]] = df["brief_title"] + ". " + df["brief_summary"]
    df[global_var["seq_a_baseline"]] = df["disease"] + ", " + df["gene"]

    return df

def generate_kfold_split(df, topic_col, seed, num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # generator
    return kf.split(df[topic_col].unique())

def split_into_train_test_set(
    df_input: pd.DataFrame,
    seq_a_col_name: str,
    seq_b_col_name: str,
    train_year: List[int],
    test_year: List[int],
    topic_subset: Optional[List[int]],
    use_qe: bool,
    global_var: Dict[str, Any],
    kfold_cv: bool,
    train_set_topics: Optional[List[int]],
    test_set_topics: Optional[List[int]]) -> List[Any]:

    if kfold_cv:
        print(train_set_topics)
        print(test_set_topics)
        print(f"df_input: {df_input}")
        print(f"df_input.columns: {df_input.columns}")
        df_train = df_input[df_input[global_var["topic"]].isin(train_set_topics)]
        df_test = df_input[df_input[global_var["topic"]].isin(test_set_topics)]

    else:
        df_train = df_input[df_input[global_var["year"]].isin(train_year)]
        df_test = df_input[df_input[global_var["year"]].isin(test_year)]

    if topic_subset is not None:
        if not kfold_cv:
            df_train, df_test = subset_data(df_train, df_test, topic_subset, global_var)

        print(f"df_train.head(): {df_train.head()}")
        print(f"df_test.head(): {df_test.head()}")

        df_train = df_train.iloc[:16]
        df_test = df_test.iloc[:16]

    print(f"df_test.shape: {df_test.shape}")
    print(f"df_train.shape: {df_train.shape}")

    seq_a_train, seq_b_train, \
    labels_train, attribute_seq_train, \
    doc_ids_train = split_into_bert_input(
        df_train,
        seq_a_col_name, seq_b_col_name,
        validate=False,
        use_qe=use_qe,
        global_var=global_var
    )

    seq_a_test, seq_b_test, \
    labels_test, attribute_seq_test, \
    doc_ids_test = split_into_bert_input(
        df_test,
        seq_a_col_name, seq_b_col_name,
        validate=False,
        use_qe=use_qe,
        global_var=global_var
    )

    ret_arr = [
        seq_a_train, seq_b_train,
        labels_train, attribute_seq_train,
        doc_ids_train,
        seq_a_test, seq_b_test,
        labels_test, attribute_seq_test,
        doc_ids_test
    ]
    return ret_arr

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

# def split_qe_into_bert_input(
#     df_input: pd.DataFrame,
#     seq_a_col_name: str,
#     seq_b_col_name: str,
#     validate) -> pd.DataFrame:

#     # Remove special characters
#     # Remove spaces
#     # Remove duplicates??

#     seq_a = [process_qe_array(x) for x in list(df_input[seq_a_col_name])]
#     seq_b = [x for x in list(df_input[seq_b_col_name])]

#     labels = None
#     if not validate:
#         labels = list(df_input[LABEL])
#     attribute_seq = list(df_input[ATTRIB_SEQ])
#     doc_ids = list(df_input[DOC_ID])

#     return seq_a, seq_b, labels, attribute_seq, doc_ids

def process_raw_array(text: str) -> str:
    pattern = '[^A-Za-z0-9]+'
    replace = ' '

    if type(text) is float:
        print(text)

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

#########    BERT VALIDATION DATA    #########

def process_testing_raw():
    pass

#########    BERT VALIDATION DATA    #########

def remove_non_char(arr):
    """ keeps only brackets"""
    pass

if __name__ == "__main__":

    df = pd.read_pickle(FINETUNING_PICKLE_PATH)

    n_chars_trim = 100

    df_input = process_baseline_raw(df, USE_QE, n_chars_trim, global_var)
    seq_a_train, seq_b_train, \
    labels_train, attribute_seq_train, doc_ids_train, \
    seq_a_test, seq_b_test, \
    labels_test, attribute_seq_test, doc_ids_test = split_into_train_test_set(
            df_input,
            SEQ_A_BASELINE,
            SEQ_B_BASELINE,
            topic_subset=None,
            global_var=global_var
        )