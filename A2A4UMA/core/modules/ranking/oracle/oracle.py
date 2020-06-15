#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:21:11 2019

@author: ryb003
"""
import numpy as np
import copy as cp
import os

def run(result_sets, topics, **kwargs):
    indexes = range(len(result_sets)) if 'inputs' not in kwargs else kwargs['inputs']
    score_col = 'score' if 'score_col' not in kwargs else kwargs['score_col']
    id_col = 'id' if 'id_col' not in kwargs else kwargs['id_col']
    q_rel_path =os.path.dirname(__file__)+'/'+kwargs['model_path']
    
    qr=np.loadtxt(q_rel_path)
    ## A) Build a fast  searchable structure from the qrel
    search_structure = {}
    for row in qr:
        topic_id=row[0]
        if topic_id not in search_structure:
            search_structure[topic_id]={}
        relevance_class=row[3]
        doc_id=row[2]
        if relevance_class>0:
            if relevance_class in search_structure[topic_id]:
                search_structure[topic_id][relevance_class].add(doc_id)
            else:
                search_structure[topic_id][relevance_class]=set([doc_id])
    final_results=[]
    for j in indexes:
    ## Re-rank
        result_set=cp.deepcopy(result_sets[j])
        ranking=result_set['ranking']
        columns=result_set['headings']
        score_index=columns.index(score_col)
        id_index=columns.index(id_col)
        
        for topic_id in ranking:
            re_ranked_topic_results=[[] for i in range((int(max(search_structure[int(topic_id)]))+1))]
            for row in ranking[topic_id]:
                doc_id=row[id_index]
                rescored_row=[]
                for relevance_score in search_structure[int(topic_id)]:
                    if int(doc_id) in search_structure[int(topic_id)][relevance_score]:
                        rescored_row=row
                        rescored_row[score_index]=1000*relevance_score
                        re_ranked_topic_results[int(relevance_score)].append(rescored_row)
                        break;
                if len(rescored_row)<1:
                    rescored_row=row
                    re_ranked_topic_results[0].append(rescored_row)
            topic_res=[]
            for rs in re_ranked_topic_results[::-1]:
                topic_res.extend(rs)
            result_set['ranking'][topic_id]=topic_res
        final_results.append(result_set)
    return final_results
        

#ad hoc testing
#rss=run(r, q_rel='../../../../data/qrels/qrels-treceval-2016.txt')
