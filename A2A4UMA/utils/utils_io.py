#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:19:03 2019

@author: ryb003
"""

import xml.dom.minidom
import copy
import json
import shlex
import subprocess
import pickle
import numpy as np
import os
import time

path=os.path.dirname(__file__)

def read_defaults(dir_path):
    with open(dir_path+'/defaults.json') as f:
        d = json.load(f)
    return d

defaults=read_defaults(path)

def read_a2a_request_config(file_path):
    f = open(file_path, "r")
    r={}
    for line in f:
        if len(line.strip())>0 and line.strip()[0] is not '#':
            r[line.split('=')[0].strip()]= json.loads('='.join(line.split('=')[1:]).strip().replace('\'','\"'))
    f.close()
    return r

def read_json_pipe_config(file_path):
    f = open(file_path, "r")
    result=[]
    for x in f:
        if len(x.strip())>0 and x.strip()[0] is not '#':
            result.append(json.loads(x.replace('\'','\"')))
    f.close()
    return result


def evaluate_sample(result_file_path, qrels_path, eval_script=defaults['sample_eval_script'].replace(" ", "\ ")):
    print ("Evaluating "+result_file_path)
    #print output_file_name
    command_line = 'perl ' +eval_script + " -q " +qrels_path + " "+ result_file_path
    print(command_line)
    args = shlex.split(command_line)
    output = subprocess.check_output(args)

    # print("output = subprocess.check_output(args) in evaluate_sample")
    # print(output)
    # time.sleep(10)

    return output

def evaluate_trec(result_file_path,
                  qrels_path,
                  eval_script=defaults['trec_eval_script'].replace(" ", "\ ")):
    #improve by making a hash-named temp and delete it afterwards
    #to make it able to be parallel
    command_line = eval_script +" -q -c -- " + qrels_path + " " + result_file_path
    print(f"command_line: {command_line}")
    args = shlex.split(command_line)
    print(f"args: {args}")
    output = subprocess.check_output(args)
    # print("output = subprocess.check_output(args) in evaluate trec")
    # print(output)
    # time.sleep(10)

    return output

def evaluate_results(result_file_path, result_folder, run_id, qrels_paths, do_sample_eval=True):

    print(f"result_file_path: {result_file_path}")
    print(f"result_folder: {result_folder}")
    print(f"run_id: {run_id}")
    print(f"qrels_paths: {qrels_paths}")
    print(f"do_sample_eval: {do_sample_eval}")

    trec_eval_file = result_folder+run_id+".treceval"

    trec_file = open(trec_eval_file, "w+")

    for line in evaluate_trec(
                    result_file_path.replace(" ", "\ "),
                    qrels_paths['trec'].replace(" ", "\ ")).splitlines():
        print(f"line: {line}")
        trec_file.write(str(line, 'utf-8') + "\n")

    if do_sample_eval:
        sample_eval_file = result_folder+run_id+".sampleeval"
        sample_file = open(sample_eval_file, "w+")
        for line in evaluate_sample(result_file_path.replace(" ", "\ "), qrels_paths['sample'].replace(" ", "\ ")).splitlines():
            sample_file.write(str(line, 'utf-8') + "\n")
        sample_file.close()

    trec_file.close()

def parse_topics(f, opt={}, names=["summary", "description", "note", 'disease', 'gene', 'demographic', 'other']):
    if 'run_types' in opt:
        names = opt['run_types']

    names_return = copy.deepcopy(names)
    topicList = []
    #print (str(f))
    DOMTree = xml.dom.minidom.parse(f)
    collection = DOMTree.documentElement
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        topicDict = dict()
        for name in names:
            topicDict["id"] = topic.getAttribute("number")
            try:
                #print (name + ' ' + run)
                topicDict[name] = topic.getElementsByTagName(name)[0].childNodes[0].data
            except Exception:
                #names_return.remove(name)
                continue
        topicList.append(topicDict)
    return topicList, names_return

def save_pickle(topics, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(topics, handle)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def results_2_trec_format(ranking, run_id, other_columns=[]):
    values=ranking['ranking']
    columns=ranking['headings']
    print(columns)
    id_ind=columns.index('id')
    score_ind=columns.index('score')
    other_indices=[columns.index(f) for f in other_columns]
    trec_vals=[]
    for topic_id in values:
        j=1
        for row in values[topic_id]:
            new_row=[topic_id, 'Q0', row[id_ind], j, row[score_ind], run_id]
            for ind in other_indices:
                new_row.append(row[ind])
            #print (new_row)
            trec_vals.append(np.array(new_row))
            j+=1
    c=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id']
    c.extend(other_columns)
    return np.array(trec_vals), c

