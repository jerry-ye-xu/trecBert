#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:36:07 2019

@author: ryb003
"""

import sys
sys.path.insert(0, '/Documents/trec_t2/A2A4UMA/utils/')
import text_utils as tu
import numpy as np
from scipy.stats import mstats


class feature_extractor:
    topic_elements=[]
    ind_fields=[]
    idf_dict={}
    
    def __init__(self, query_field, index_field, idf):
        self.topic_elements=query_field
        self.ind_fields=index_field
        self.idf_dict=idf
    
    def get_term_overlap(self, q, d):
        sq=set(tu.remove_stopwords_punctuation(str(q)).split())
        sd=set(tu.remove_stopwords_punctuation(str(d)).split())
        cnt=0.0
        for t in sq:
            if t in sd:
                cnt+=1
        return cnt/float(len(sq))
    
    def get_weighted_term_overlap(self, q, d, idf_dict):
        sq=set(tu.remove_stopwords_punctuation(str(q)).split())
        sd=set(tu.remove_stopwords_punctuation(str(d)).split())
        cnt=0.0
        norm=0.0
        for t in sq:
            if t in sd:
                if t in idf_dict:
                    cnt+=idf_dict[t]
            if t in idf_dict:
                norm+=idf_dict[t]
        if norm>0:
            return cnt/norm
        return 0
    
    def get_bigram_overlap(self, q, d):
        tq=tu.remove_stopwords_punctuation(str(q)).split()
        td=tu.remove_stopwords_punctuation(str(d)).split()
        bigramsq=set([' '.join(tq[x:x+2]) for x in range(len(tq)-1)])
        bigramsd=set([' '.join(td[x:x+2]) for x in range(len(td)-1)])
        cnt=0.0
        for t in bigramsq:
            if t in bigramsd:
                cnt+=1
        return cnt/float(len(bigramsq))
    
    def get_weighted_bigram_overlap(self, q, d, idf_dict):
        tq=tu.remove_stopwords_punctuation(str(q)).split()
        td=tu.remove_stopwords_punctuation(str(d)).split()
        bigramsq=set([' '.join(tq[x:x+2]) for x in range(len(tq)-1)])
        bigramsd=set([' '.join(td[x:x+2]) for x in range(len(td)-1)])
        cnt=0.0
        norm=0.0
        for t in bigramsq:
            ts=t.split()
            if t in bigramsd:
                if ts[0] in idf_dict and ts[1] in idf_dict:
                    cnt+=(idf_dict[ts[0]] * idf_dict[ts[1]])
            if ts[0] in idf_dict and ts[1] in idf_dict:
                norm+=(idf_dict[ts[0]] * idf_dict[ts[1]])
        if norm>0:
            return cnt/norm
        return 0
        
    def vectorize_data(self, list_of_topicsets, list_of_rankings, list_of_qrels=[], subsampling='random'):
        xy=[]
        for i in range (len(list_of_rankings)):
            rs=list_of_rankings[i]
            topics=list_of_topicsets[i]
            
            if len(list_of_qrels)>0:
                q_rel_path=list_of_qrels[i]
                search_structure = {}
                qr=np.loadtxt(q_rel_path)
                ## A) Build a fast  searchable structure from the qrel
                for row in qr:
                    topic_id=row[0]
                    if topic_id not in search_structure:
                        search_structure[topic_id]=set([])
                    doc_id=row[2]
                    if row[3]>0:
                        search_structure[topic_id].add(doc_id)
        
            
            rankings_per_topic=rs['ranking']
            columns=rs['headings']
            id_idx=columns.index('id')
            score_idx=columns.index('score')
            i=0
            for topic_id in rankings_per_topic:
                tmp=[]
                i+=1
                q_text=''
                for e in self.topic_elements:
                    q_text+= (' ' +topics[int(topic_id)-1][e])
                for record in rankings_per_topic[topic_id]:
                    d_text=''
                    for f in self.ind_fields:
                        d_text+= (' ' +record[columns.index(f)])
                    term_ovl=self.get_term_overlap(q_text, d_text)
                    wto=self.get_weighted_term_overlap(q_text, d_text, self.idf_dict)
                    bigr_ovl=self.get_bigram_overlap(q_text, d_text)
                    wbo=self.get_weighted_bigram_overlap(q_text, d_text, self.idf_dict)
                    raw_bm25=float(record[score_idx])
                    doc_id=int(record[id_idx])
                    row=[i*1000+float(topic_id),term_ovl,wto, bigr_ovl,wbo, raw_bm25]
                    if len(list_of_qrels)>0:
                        row.append(float(doc_id in search_structure[int(topic_id)]))
                    tmp.append(np.array(row))
                tmp=np.array(tmp)
                zscores=mstats.zscore(tmp[:,5])
                tmp=np.append(zscores.reshape(zscores.shape[0],1), tmp, axis=1)
                
                #if we're preparing the training data, we want the subsampling
                if len(list_of_qrels)>0 and subsampling is not None:
                    p=tmp[np.where(tmp[:,-1]==1)]
                    n=tmp[np.where(tmp[:,-1]==0)]
                    if subsampling is 'random':
                        np.random.shuffle(n)
                    ## top-n subsampling if not shuffled
                    tmp=np.vstack((p, n[:len(p)]))
                
                if len(xy) is 0:
                    xy=np.copy(tmp)
                else:
                    xy=np.vstack((xy,tmp))
        if len(list_of_qrels) is 0:
            return xy[:,[0,2,3,4,5]], ['bm25', 'to', 'wto', 'bo', 'wbo']
        return xy[:,[0,2,3,4,5,7]], ['bm25', 'to', 'wto', 'bo', 'wbo', 'y']

        