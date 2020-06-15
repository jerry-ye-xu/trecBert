import json
import os
import pandas as pd
import requests

from bs4 import BeautifulSoup
from random import randint, random, seed

def build_bioasq(data_path, col_names):

    with open(data_path, "r") as f:
        data = json.load(f)

    data_arr = pull_snippets_as_data(data, strong_label=2, weak_label=1)

    df = pd.DataFrame(data_arr, columns=col_names)
    print(df.head())

    pos_df = process_bioasq_df(df)

    return pos_df

def pull_snippets_as_data(data, strong_label=2, weak_label=1):
    data_arr = []
    for d_pt in data["questions"]:
        d_pt_id = d_pt["id"]
        try:
            data_arr.append((d_pt_id, f"{d_pt_id}_0",  d_pt["type"], "ideal_answer", d_pt["body"], d_pt["ideal_answer"], strong_label))
        except KeyError:
            print("ideal answer not found. Using exact answer.")
            data_arr.append((d_pt_id, f"{d_pt_id}_0", d_pt["type"], "exact_answer", d_pt["body"], d_pt["exact_answer"], strong_label))
        idx = 1
        for d_sn in d_pt["snippets"]:
            d_pt_id = d_pt["id"]
            data_arr.append((d_pt_id, f"{d_pt_id}_{idx}", d_pt["type"], "snippet_answer", d_pt["body"], d_sn["text"], weak_label))
            idx += 1
    print(data_arr[:5])
    return data_arr

def process_bioasq_df(df):
    df["answer"] = df["answer"].apply(convert_arr_to_string)
    print("df[\"answer\"].apply(len).mean():")
    print(df["answer"].apply(len).mean())
    df["answer_len"] = df["answer"].apply(len)

    df["answer"] = df["answer"].str.replace("\n", " ")
    df["question"] = df["question"].str.replace("\n", " ")
    df["answer"] = df["answer"].str.replace("\t", " ")
    df["question"] = df["question"].str.replace("\t", " ")

    # This could be cleaner...
    df1 = df[~((df["answer"].str.contains("yes", case=False)) & (df["answer_len"] < 5))]
    df2 = df1[~((df1["answer"].str.contains("no", case=False)) & (df1["answer_len"] < 5))]

    return df2

def convert_arr_to_string(df_col):
    # Data suggests that next sentences in array are very similar, no need for duplication.
    return df_col[0] if (type(df_col) == list) else df_col


#########################
## GET NEGATIVE LABELS ##
#########################

def get_abstract(url):
    page = requests.get(url)
    if page.status_code != requests.codes.ok:
        return 0
    print("url: {url}")
    print(page)
    soup = BeautifulSoup(page.content, 'html.parser')

    div_with_abstr = soup.find(class_='abstr')

    if div_with_abstr is None:
        return None

    # Need to reach inside (you can do it with a for-loop but as long as the
    # structure is unchanged, then this will be fine.)
    abstract = div_with_abstr.div.p.text

    return abstract

def eliminate_relevant_sent(abstract, onset, offset):
    return abstract[:onset] + abstract[offset:]

def split_into_sent(abstract):
    sent = abstract.split(". ")
    sent = [s+"." for s in sent]
    return sent

def create_neg_data_point(url, onset, offset):
    abstract = get_abstract(url)
    if abstract == 0:
        return "request.get returned bad status code."
    elif abstract is None:
        return "div_with_abstr = soup.find(class_='abstr') returned None..."

    abstract = eliminate_relevant_sent(
        abstract,
        onset,
        offset
    )
    abstr_sent = split_into_sent(abstract)
    abstr_sent = list(filter(lambda sent: sent != ".", abstr_sent))

    # Just be careful here.
    if len(abstr_sent) == 0:
        return "This snippet is empty when you remove the relevant sentence."

    ub = len(abstr_sent) - 1
    print(f"ub: {ub}")
    print(abstr_sent)
    return abstr_sent[randint(0, ub)]

def create_neg_dataset(bioasq_path, col_names):
    with open(bioasq_path, "r") as f:
        data = json.load(f)

    neg_label = 0
    strong_neg_label = -1
    data_size = len(data["questions"])

    neg_ids = None
    backup_path = "./data/bioasq_neg_sample_backup_25864.pickle"
    if os.path.exists(backup_path):
        neg_backup = pd.read_pickle(backup_path)
        neg_ids = neg_backup.qa_id.unique()

    neg_data_pts = []
    for i in range(len(data["questions"])):
        id_counter = 1000
        data_id = data["questions"][i]["id"]

        if (neg_ids is not None) and (data_id in neg_ids):
            print(f"qa_id {data_id} already has negative samples. Skipping to the next one.")
            continue

        data_type = data["questions"][i]["type"]
        question = data["questions"][i]["body"]

        print(f"QUESTION: {question}")

        for snip in data["questions"][i]["snippets"]:
            neg_sent = create_neg_data_point(
                snip["document"],
                snip["offsetInBeginSection"],
                snip["offsetInEndSection"]
            )
            data_id = data["questions"][i]["id"]
            neg_data_pts.append((
                data_id, f"{data_id}_{id_counter}", data_type,
                "snippet_neg", question, neg_sent, neg_label
            ))
            id_counter += 1

        min_span = max(0, i - 50)
        max_span = min(data_size - 1, 50 + i)
        print(f"min_span = {min_span}")
        print(f"max_span = {max_span}")
        for _ in range(5):
            j = randint(i + 1, max_span) if (round(random()) or i == 0) else randint(min_span, i - 1)
            print(data["questions"][j]["body"])
            print(f"question_idx: {j}")

            snip_external = data["questions"][j]["snippets"]
            snip_idx = randint(0, len(snip_external) - 1)

            print(f"snip_idx: {snip_idx}")

            strong_neg_sent = create_neg_data_point(
                snip_external[snip_idx]["document"],
                snip_external[snip_idx]["offsetInBeginSection"],
                snip_external[snip_idx]["offsetInEndSection"]
            )
            neg_data_pts.append((
                data_id, f"{data_id}_{id_counter}", data_type,
                "strong_snippet_neg", question, strong_neg_sent, strong_neg_label
            ))
            id_counter += 1

        if i % 10 == 0:
            df = pd.DataFrame(neg_data_pts, columns=col_names)
            df2 = process_bioasq_df(df)
            df2.to_pickle(f"./data/bioasq_neg_sample_backup.pickle")

    return neg_data_pts

def build_bioasq_neg(data_path, col_names):
    neg_data_arr = create_neg_dataset(data_path, col_names)
    df = pd.DataFrame(neg_data_arr, columns=col_names)
    df2 = process_bioasq_df(df)
    print(df2.head())

    return df2

if __name__ == "__main__":
    DATA_PATH = "./data/BioASQ-training8b/training8b.json"
    SAVE_PATH = "./data/bioasq_df.pickle"

    COL_NAMES = [
        "qa_id", "qa_snap_id", "type", "answer_type",
        "question", "answer",
        "label"
    ]

    bioasq_pos = build_bioasq(DATA_PATH, COL_NAMES)
    bioasq_pos.to_pickle("./data/bioasq_pos.pickle")
    bioasq_neg = build_bioasq_neg(DATA_PATH, COL_NAMES)
    bioasq_neg.to_pickle("./data/bioasq_neg.pickle")

    bioasq_df = pd.concat([bioasq_pos, bioasq_neg], ignore_index=True)
    bioasq_df.to_pickle(SAVE_PATH)
