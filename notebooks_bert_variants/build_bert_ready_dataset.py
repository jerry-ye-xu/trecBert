#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 11:13:01 2019

@author: xu081

Comments: This file contains some helper functions used in processing clinical trials data and parse the XML topic files and combine them. The end result is a DataFrame with all attributes sitting in a single column.
"""

import re

import pandas as pd
import xml.etree.ElementTree as et
from collections import defaultdict

def parse_topics(topics_path, topics_xml):
    """ Specifc XML parsing of topics file for TREC PM dataset """
    topics_xml = et.parse(f"{topics_path}/{topics_xml}")
    topics_root = topics_xml.getroot()
    topic_root = topics_root
    
    topics_dict = defaultdict(list)
    for child in topics_root:
        feature_arr = []
        topic_num = child.attrib["number"]
        for cc in child:
            feature_arr.append(cc.text)
        if int(topic_num) % 10 == 0:
            print(f"child.attrib[\"number\"]: {topic_num}")
            print(feature_arr)
        topics_dict[int(topic_num)] = feature_arr
        
    return topics_dict

def find_age(string):
    # Match checks for patterns starting from the beginning
    x = re.match(r'\d+', string)
    if x is None:
        raise ValueError("Did not find age feature!")
    return int(x.group())

def find_gender(string):
    # Search checks for patterns throughout entire string
    x = re.search(r'(fe)?(male)+', string)
    if x is None:
        raise ValueError("Did not find gender feature!")
    return x.group()

def subset_data(df, col, p):
    num_lab_1 = df[col].value_counts()[0]
    num_lab_2 = df[col].value_counts()[1]

    lab1_subset = df[df[col] == 0].sample(int(p*num_lab_1))
    lab2_subset = df[df[col] == 1].sample(int(p*num_lab_2))
    
    return pd.concat([lab1_subset, lab2_subset])

if __name__ == "__main__":
    
    ##### CLINICAL TRIALS #####
    
    LABELS_PATH = "../data/pm_labels_2018/qrels-treceval-clinical_trials-2018-v2.txt"
    FEATURES_PATH = "../data/trials_pickle_2018/all_trials_2018.pickle"

    df_labels = pd.read_csv(
        LABELS_PATH,
        names=["topic", "_", "id", "label"], sep=" "
    )
    df_features = pd.read_pickle(FEATURES_PATH)

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
    
    TOPICS_PATH = "../data/pm_labels_2018"
    TOPICS_XML = "topics2018.xml"
    
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
    
    df.to_pickle("../data/trials_topics_combined_full.pickle", index=False)