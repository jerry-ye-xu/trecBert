#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:06:18 2019

@author: ryb003
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:21:11 2019

@author: ryb003
"""
import numpy as np
import copy as cp

def sort_by_score(record, score_index):
    return record[score_index]


def get_doc_dict(doc_id, solr_conn):
    q_params={"fl": '*', "rows": 1 }
    result = solr_conn.search('id:'+str(doc_id), **q_params)
    d={}
    for doc in result:
        for f in doc:
            d[str(f)]=str(doc[f]).strip()
    #print(doc_id)
    #print(d)
    return d


    

def run(result_sets, topics, **kwargs):
    indexes = [0, 1] if 'inputs' not in kwargs else kwargs['inputs']
    score_col = 'score' if 'score_col' not in kwargs else kwargs['score_col']
    alpha = [0.5, 0.5] if 'alpha' not in kwargs else kwargs['alpha']
    id_col = 'id' if 'id_col' not in kwargs else kwargs['id_col']
    norm = False if 'norm' not in kwargs else kwargs['norm']
    
    final_results=[]
    result_set=cp.deepcopy(result_sets[indexes[0]])
    result_set['ranking']={}
    result_set['headings']=['id','score']
    
    #(topic_id, doc_id) -> accumulated_score
    new_scores={}
    i=0
    for j in indexes:
        rs=result_sets[j]
        ranking=rs['ranking']
        columns=rs['headings']
        score_index=columns.index(score_col)
        id_index=columns.index(id_col)
        
        for topic_id in ranking:
            if norm and len(ranking[topic_id])>0:
                norm_term=ranking[topic_id][0][score_index]
            else:
                norm_term=1
            for row in ranking[topic_id]:
                doc_id=row[id_index]
                local_score=row[score_index]
                if (topic_id, doc_id) in new_scores:
                    new_scores[topic_id, doc_id]+=((local_score/norm_term)*alpha[i])
                else:
                    new_scores[topic_id, doc_id]=((local_score/norm_term)*alpha[i])
        i+=1
    for topid, docid in new_scores:
        if topid not in result_set['ranking']:
            result_set['ranking'][topid]=[]
        result_set['ranking'][topid].append([docid, new_scores[(topid, docid)]])
    ranking=result_set['ranking']
    for topic_id in ranking:
        ranking[topic_id].sort(key = lambda record: sort_by_score(record, score_index), reverse = True)
    final_results.append(result_set)
    return final_results
        

#ad hoc testing
#rss=run(r, q_rel='../../../../data/qrels/qrels-treceval-2016.txt')
