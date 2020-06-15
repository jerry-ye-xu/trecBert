import glob
import os
import pandas as pd

import xml.dom.minidom as md

##
## Combine XML Files
##

def combine_xml_files(path):
    """
    Set the first file path as the head and append all
    subsequent file topics as children to the head after
    adjusting the topic number.
    """
    file_paths = sorted(glob.glob(path))
    DOMTreeHead = md.parse(file_paths[0])

    for i in range(1, len(file_paths)):
        DOMTreeChild = md.parse(file_paths[i])
        append_child_topics(DOMTreeHead, DOMTreeChild)

    return DOMTreeHead

def update_topic_num(max_topic_num, topics_child, index):
    """
    XML files are DOMTrees.
    """
    curr_num = int(topics_child[index].getAttribute("number"))
    print(f"curr_sum: {curr_num}")
    print(f"max_topic_num: {max_topic_num}")
    topics_child[index].setAttribute("number", str(curr_num + max_topic_num))
    print(f"str(curr_num + max_topic_num): {curr_num + max_topic_num}")

def append_child_topics(DOMTreeHead, DOMTreeChild):
    """
    XML files are DOMTrees.
    """
    topics_head = DOMTreeHead.documentElement.getElementsByTagName("topic")
    topics_child = DOMTreeChild.documentElement.getElementsByTagName("topic")

    max_topic_num = int(topics_head[-1].getAttribute("number"))
    print(f"max_topic_num: {max_topic_num}")

    for index in range(len(topics_child)):
        update_topic_num(max_topic_num, topics_child, index)
        DOMTreeHead.childNodes[0].appendChild(topics_child[index])

##
## Combine QRELS Files
##

def combine_qrels_files(path, col_names, sep, start_year):
    qrels = [pd.read_csv(p, names=col_names, sep=sep) for p in sorted(glob.glob(path))]

    for i in range(0, len(qrels)):
    #     add_topic_idx(qrels[i-1], [i])
        qrels[i]["year"] = start_year + i
        qrels[i]["year_topic"] = qrels[i]["year"].astype(str) + "_" + qrels[i]["topic"].astype(str) # year and topic

    qrels_arr = []
    for i in range(1, len(qrels)):
        add_topic_idx_qrels(qrels[i-1], qrels[i], qrels_arr, i)

    qrels_all = pd.concat(qrels)

    # yrtop ==> year and topic
    yrtop2orig_dict = qrels_all.set_index("topics_all").to_dict()["year_topic"]
    print("list(yrtop2orig_dict.keys())[:5]")
    print(list(yrtop2orig_dict.keys())[:5])
    orig2yrtop_dict = qrels_all.set_index("year_topic").to_dict()["topics_all"]
    print("list(orig2yrtop_dict.keys())[:5]")
    print(list(orig2yrtop_dict.keys())[:5])

    return qrels_all, yrtop2orig_dict, orig2yrtop_dict

def add_topic_idx_qrels(df_curr, df_next, qrels_arr, index):
    if index == 1:
        df_curr["topics_all"] = df_curr["topic"].astype(int)
        qrels_arr.append(df_curr)
    max_topic_num = max(df_curr["topics_all"].unique())
    print(f"max_topic_num: {max_topic_num}")

    df_next["topics_all"] = df_next["topic"].map(lambda x: x + max_topic_num)
    df_next["topics_all"] = df_next["topics_all"].astype(int)
    print(df_next.head())
    print(df_next.tail())

    qrels_arr.append(df_next)
    print(max_topic_num)

if __name__ == "__main__":
    SAVE_PATH = "./data/cv_files"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    QRELS_PATH = "./data/pm_labels_*/qrels_treceval_clinical_trials_*"
    COL_NAMES = ["topic", "unknown", "doc_id", "label"]
    SEP = " "
    START_YEAR = 2017

    qrels_all, yrtop2orig_dict, orig2yrtop_dict = combine_qrels_files(
        QRELS_PATH, COL_NAMES, SEP, START_YEAR
    )

    SAVE_COLUMNS = ["topics_all", "unknown", "doc_id", "label"]
    qrels_all[SAVE_COLUMNS].to_csv(
        f"{SAVE_PATH}/qrels_combined.txt",
        sep=" ", header=False, index=False
    )
    pd.to_pickle(yrtop2orig_dict, path=f"{SAVE_PATH}/yrtop2orig_dict.pickle")
    pd.to_pickle(orig2yrtop_dict, path=f"{SAVE_PATH}/orig2yrtop_dict.pickle")

    XML_PATH = "./data/pm_labels_*/topics*.xml"
    NewDOMTreeHead = combine_xml_files(XML_PATH)
    with open(f"{SAVE_PATH}/topics_combined.xml", "w") as f:
        NewDOMTreeHead.writexml(f)
