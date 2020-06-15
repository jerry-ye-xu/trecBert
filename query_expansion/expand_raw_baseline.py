import pandas as pd

from collections import defaultdict
from typing import Any, List, Dict, Tuple, Sequence, Optional

from snomed_ncbi_qe import *

from query_expansion_global_var import SEQ_A_EXPANSION

def query_expansion(
    df: pd.DataFrame,
    params: Dict[str, Any],
    fields_to_keep: List[str]) -> pd.DataFrame:

    df["d_and_g"] = df["disease"] + ", " + df["gene"]

    uniq_d_g = df.drop_duplicates(subset=["d_and_g"]).reset_index()

    dg_qe_dict = defaultdict(list)
    for i in range(uniq_d_g.shape[0]):
        print(uniq_d_g["disease"][i])
        print(uniq_d_g["gene"][i])

        disease = uniq_d_g["disease"][i]
        gene = uniq_d_g["gene"][i]

        params = {'apikey':'8de61230-b5f0-42f4-8b3e-8f0b0f426cc1'}
        t = run(['gene', 'disease'],{'disease': disease, 'gene': gene}, **{'sparql_params': params})

        for key, value in t.items():
            if key in fields_to_keep:
                d_and_g = uniq_d_g["d_and_g"][i]
                dg_qe_dict[d_and_g].append(value)
        print(dg_qe_dict)

    df[SEQ_A_EXPANSION] = df["d_and_g"].map(dg_qe_dict)

    return df

def query_expansion_all(
    df: pd.DataFrame,
    params: Dict[str, Any]) -> pd.DataFrame:

    df["d_and_g"] = df["disease"] + ", " + df["gene"]

    uniq_d_g = df.drop_duplicates(subset=["d_and_g"]).reset_index()

    dg_qe_dict = defaultdict(list)
    for i in range(uniq_d_g.shape[0]):
        print(uniq_d_g["disease"][i])
        print(uniq_d_g["gene"][i])

        disease = uniq_d_g["disease"][i]
        gene = uniq_d_g["gene"][i]

        params = {'apikey':'8de61230-b5f0-42f4-8b3e-8f0b0f426cc1'}
        t = run(['gene', 'disease'],{'disease': disease, 'gene': gene}, **{'sparql_params': params})

        for key, value in t.items():
            d_and_g = uniq_d_g["d_and_g"][i]
            dg_qe_dict[d_and_g].append(value)

        print(dg_qe_dict)

    df[SEQ_A_EXPANSION] = df["d_and_g"].map(dg_qe_dict)

    return df


if __name__ == "__main__":

    #### Validation data set ####

    qe_bert_df = False
    qe_topic_id_full_df = True
    qe_paper_topic_id_full_df = True

    API_KEY = '8de61230-b5f0-42f4-8b3e-8f0b0f426cc1'

    if qe_bert_df:
        print("Query Expansion for trials_topics_combined_all_years data.\nIn other words the validation data")

        df = pd.read_pickle("./A2A4UMA/pre_rerank_data_files/df_base_for_bert_full.pickle")

        print(df.columns)
        print(df.head())

        df["d_and_g"] = df["disease"] + ", " + df["gene"]

        # # One example
        # disease = df["disease"].iloc[0]
        # gene =  df["gene"].iloc[0]

        # params = {'apikey':'8de61230-b5f0-42f4-8b3e-8f0b0f426cc1'}
        # t = run(['gene', 'disease'],{'disease': disease, 'gene': gene}, **{'sparql_params': params})

        # dg_qe_dict = defaultdict(dict)

        # for key, value in t.items():
        #     print(key)
        #     d_and_g = df["d_and_g"][0]
        #     dg_qe_dict[d_and_g][key] = value

        # print(dg_qe_dict)

        # print(df["d_and_g"].shape)
        # print(df["d_and_g"].nunique())


        uniq_d_g = df.drop_duplicates(subset=["d_and_g"]).reset_index()

        dg_qe_dict = defaultdict(list)
        for i in range(uniq_d_g.shape[0]):
            print(uniq_d_g["disease"][i])
            print(uniq_d_g["gene"][i])

            disease = uniq_d_g["disease"][i]
            gene = uniq_d_g["gene"][i]

            params = {'apikey':'8de61230-b5f0-42f4-8b3e-8f0b0f426cc1'}
            t = run(['gene', 'disease'],{'disease': disease, 'gene': gene}, **{'sparql_params': params})

            for key, value in t.items():
                d_and_g = uniq_d_g["d_and_g"][i]
                dg_qe_dict[d_and_g].append(value)

        df[SEQ_A_EXPANSION] = df["d_and_g"].map(dg_qe_dict)

        df.to_pickle("./A2A4UMA/pre_rerank_data_files/df_qe_bert_full.pickle")

    #### Training set ####

    if qe_topic_id_full_df:
        print("Query Expansion for trials_topics_combined_all_years data.\nIn otherwords the training data")
        df_training_full = pd.read_pickle("./data/trials_topics_combined_all_years.pickle")
        params = {'apikey': API_KEY}
        save_path = "./data/trials_topics_combined_all_years_qe.pickle"

        df_full_qe = query_expansion_all(
            df_training_full,
            params
        )
        print(df_full_qe.head())

        df_full_qe.to_pickle(save_path)

    fields_to_keep = [
        "disease", "disease_kbqe_pn", "disease_kbqe_syn",
        "gene", "gene_kbqe_syn", "gene_kbqe_other"
    ]

    if qe_paper_topic_id_full_df:
        print("Query Expansion for trials_topics_combined_all_years data.\nIn otherwords the training data, adding fields used by the paper we are trying to reproduce.")
        df_training_full = pd.read_pickle("./data/trials_topics_combined_all_years.pickle")
        params = {'apikey': API_KEY}
        save_path = "./data/trials_topics_combined_all_years_qe_paper.pickle"

        df_full_qe = query_expansion(
            df_training_full,
            params,
            fields_to_keep
        )
        print(df_full_qe.head())

        df_full_qe.to_pickle(save_path)

    # Testing

    disease = "pancreatic cancer"
    gene = "BRCA2"

    params = {'apikey': API_KEY}
    t = run(['gene', 'disease'],{'disease': disease, 'gene': gene}, **{'sparql_params': params})
    print(t)