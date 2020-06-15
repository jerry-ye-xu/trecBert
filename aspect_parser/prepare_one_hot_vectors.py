import pandas as pd
import xml.etree.ElementTree as et

import spacy

from collections import defaultdict
from tqdm import tqdm

from build_bert_ready_dataset import parse_topics

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

sp = spacy.load('en_core_web_sm')

# https://spacy.io/usage/linguistic-features#native-tokenizer-additions
# https://github.com/explosion/spaCy/blob/master/spacy/util.py
# https://github.com/explosion/spaCy/blob/master/spacy/lang/punctuation.py
# infixes = list(sp.Defaults.infixes) # get the default infixes but best to use

infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # EDIT: commented out regex that splits on hyphens between letters:
        #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
sp.tokenizer.infix_finditer = infix_re.finditer

def build_field_dict(topics_dict, idx, spacy_lm):
    """
    Grab topics_dict, which has format
        {
        1: [<gene>, <disease>, <age and gender>, <other>],
        2: [<gene>, <disease>, <age and gender>, <other>],
        ...
        n: [<gene>, <disease>, <age and gender>, <other>]
    }
    and return a one-hot vector of unique words along with an idx mapping
    to the position of the words
    """
    i = 0
    unique_tokens = {}
    for key, value in topics_dict.items():
        print(f"key: {key}\nvalue: {value}")
        field_tokens_spacy = [w.lemma_ for w in spacy_lm(value[idx])]
        field_tokens = [process_field_text(t) for t in field_tokens_spacy]
        for t in field_tokens:
            try:
                unique_tokens[t]
            except KeyError:
                if len(t) > 0:
                    unique_tokens[t] = i
                    i += 1
    return unique_tokens

def build_field_qe_dict(df_col, stop_words, spacy_lm):
    """
    Grab pd.Series with
        [
        word1, word2, ..., word_k
    ]
    and return a one-hot vector of unique
    words along with an idx mapping to the position of the words.
    """
    i = 0
    unique_tokens = {}
    for col in df_col:
        text = " ".join(col)
        print(f"text: {text}")
        field_tokens_spacy = [w.lemma_ for w in spacy_lm(text)]
        field_tokens = [process_field_text(t) for t in field_tokens_spacy]
        for t in field_tokens:
            try:
                unique_tokens[t]
            except KeyError:
                if (len(t) > 1) and (t not in stop_words):
                    unique_tokens[t] = i
                    i += 1
    return unique_tokens

def process_field_text(token, spacy_lm=sp):
#     print(token)
    token = token.strip()
    token = token.replace("(", "")
    token = token.replace(")", "")
    token = token.replace(",", "")
    token = token.replace(";", "")
    token = token.replace(":", "")
    token = token.lower()
#     token = lemmatise(token, spacy_lm)
    return token

def lemmatise(token, spacy_lm):
    # Returns a Spacy document, which is special
    # We however, only need the token.
    if len(token) == 0:
#         print(f"token is: {token}")
#         print("len of token is zero")
        token = "none"
    text_sp = spacy_lm(token)
    if len(text_sp.text) == 0:
        print(text_sp.text)
    return text_sp[0].lemma_

def count_field(query_field, field_dict, spacy_lm):
    """
    Given a query field (e.g. gene), count the exact matches for
    each gene that appears in the field_dict.

    The field_token_arr is a one-hot vector encoding for each word.
    Luckily there isn't too many as we are purely focusing on genes.
    """
    print_flag = False

    field_token_arr = [0]*(len(field_dict.keys()))

    field_tokens_spacy = [w.lemma_ for w in spacy_lm(query_field)]
    query_field = [process_field_text(t) for t in field_tokens_spacy]
#     print(f"query_field: {query_field}")
    for t in query_field:
#         print(f"t: {t}")
#         print(field_dict[t])
        if t in field_dict.keys():
            print_flag = True
            field_token_arr[field_dict[t]] += 1

    if print_flag:
#         print(f"sum of counts is: {sum(field_token_arr)}")
#         print(f"processed query_field is:\n{query_field}")
        print_flag = False
    return field_token_arr

def build_field_count(df_col_text, df_col_id, field_dict, spacy_lm):
    """
    word_count_vec_dict = {
        "2017_1_NCT01774162": [0, 0, 1, ..., 0],
        "2017_1_NCT..."     : [1, 1, 1, ..., 0],
        ...
        "2019_40_NCT..."    : [0, 1, 0, ..., 0],
    }
    """
    word_count_vec_dict = {}
    for i in tqdm(range(len(df_col_text))):
#         print(f"index of df is: {i}")
        count_vec = count_field(df_col_text[i], field_dict, spacy_lm)
        word_count_vec_dict[df_col_id[i]] = count_vec

    return word_count_vec_dict

if __name__ == "__main__":
    YEAR = 2018
    topics_path = f"./data/pm_labels_{YEAR}"
    topics_xml = f"topics{YEAR}.xml"

    tops = parse_topics(topics_path, topics_xml)

    file_path = f"./data/aspect_data/judgements_word_vec_{YEAR}/"

    ###
    ### QUERY FIELDS
    ###

    desc = f"full_{YEAR}"

    df = pd.read_pickle(f"./data/trials_topics_combined_full_{YEAR}.pickle")
    df["brief_t_and_s"] = df["brief_title"] + " " + df["brief_summary"]
    df["doc_id"] = df["year"].astype(str) + "_" + df["topic"].astype(str) + "_" + df["id"]
    print(df.shape)
    print(df.head())

    gene2id_dict = build_field_dict(topics_dict=tops, idx=1, spacy_lm=sp)
    gene_count_vec_dict = build_field_count(
        df_col_text=df["brief_t_and_s"],
        df_col_id=df["doc_id"],
        field_dict=gene2id_dict, spacy_lm=sp
    )
    gene_path = f"{file_path}/gene_word_vec_{desc}.pickle"
    pd.to_pickle(gene_count_vec_dict, gene_path)

    disease2id_dict = build_field_dict(topics_dict=tops, idx=0, spacy_lm=sp)
    disease_count_vec_dict = build_field_count(
        df_col_text=df["brief_t_and_s"],
        df_col_id=df["doc_id"],
        field_dict=disease2id_dict, spacy_lm=sp
    )
    disease_path = f"{file_path}/disease_word_vec_{desc}.pickle"
    pd.to_pickle(disease_count_vec_dict, disease_path)

    ###
    ### QUERY EXPANSION
    ###

    df_qe = pd.read_pickle("./data/trials_topics_combined_all_years_qe.pickle")
    df_qe = df_qe[df_qe["year"] == YEAR].copy()
    df_qe.reset_index(drop=True, inplace=True)
    df_qe["brief_t_and_s"] = df_qe["brief_title"] + " " + df_qe["brief_summary"]
    df_qe["doc_id"] = df_qe["year"].astype(str) + "_" + df_qe["topic"].astype(str) + "_" + df_qe["id"]
    print(df_qe.shape)
    print(df_qe.head())

    # See snomed_ncbi_qe.py for the indexing
    df_qe["gene_qe"] = df_qe["qe_all"].apply(lambda x: x[1:5])
    df_qe["disease_qe"] = df_qe["qe_all"].apply(lambda x: [x[0]] + x[5:])

    df_qe_unique_gd = df_qe.drop_duplicates(subset=["year", "topic"])
    # df_qe_unique_gd = df_qe_unique_gd[df_qe_unique_gd["year"] == YEAR].copy()
    df_qe_unique_gd.reset_index(drop=True, inplace=True)
    print(df_qe_unique_gd.shape)
    print(df_qe_unique_gd.head())

    gene2id_dict = build_field_qe_dict(
        df_col=df_qe_unique_gd["gene_qe"],
        stop_words=STOP_WORDS, spacy_lm=sp
    )
    gene_count_vec_dict = build_field_count(
        df_col_text=df_qe["brief_t_and_s"],
        df_col_id=df_qe["doc_id"],
        field_dict=gene2id_dict, spacy_lm=sp
    )
    gene_path = f"{file_path}/gene_qe_word_vec_{desc}.pickle"
    pd.to_pickle(gene_count_vec_dict, gene_path)

    disease2id_dict = build_field_qe_dict(
        df_col=df_qe_unique_gd["disease_qe"],
        stop_words=STOP_WORDS, spacy_lm=sp
    )
    disease_count_vec_dict = build_field_count(
        df_col_text=df_qe["brief_t_and_s"],
        df_col_id=df_qe["doc_id"],
        field_dict=disease2id_dict, spacy_lm=sp
    )
    disease_path = f"{file_path}/disease_qe_word_vec_{desc}.pickle"
    pd.to_pickle(disease_count_vec_dict, disease_path)