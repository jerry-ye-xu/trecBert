#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:03:32 2019

@author: ryb003
"""

import os
path=os.path.dirname(__file__)+'/'

import sys
sys.path.insert(0, path+'./')
import bm25_solr_params as bm25

import pysolr

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh import scoring

def run_query(text, index, bm25_params={}, **kwargs):#, qf="title_text_en^2 abstract_text_en^2 body_text_en^1.1", fields=['id','score'], size=1000, max_year=2016):
    if type(index) is pysolr.Solr:
        kwargs['verb']=1
        qf="text^1" if 'qf' not in kwargs else kwargs['qf']
        return_fields=['id','score'] if 'return_fields' not in kwargs else kwargs['return_fields'] #return fields
        size=1000 if 'size' not in kwargs else kwargs['size']
        max_year=2016 if 'max_year' not in kwargs else kwargs['max_year']
        parser='edismax' if 'parser' not in kwargs else kwargs['parser']

        if 'verb' in kwargs:
            print(text)
            print(qf)
            print(bm25_params)
        if len(bm25_params)>0:
            bm25.set_params(**bm25_params)

        q_params={"fl": ','.join(return_fields),
                  #"fq": "body_text_en:[* TO *] AND date_i:[* TO "+str(max_year)+"]",
                  "fq": "date_i:[* TO "+str(max_year)+"]",
                  #"pf": "abstract_text_en^1.2 title_text_en^2",
                  # "start": "1",
                  "rows": str(size),  # return maximum 1000 results,
                  "defType": parser
                  }
        if max_year==0 or max_year>=2016:
            q_params.pop('fq')
        if len(qf)>0:
            q_params["qf"]=qf
        result = index.search(text, **q_params)
        return result, return_fields
    else:
        kwargs['verb']=1
        qf="text^1" if 'qf' not in kwargs else kwargs['qf']
        return_fields=['id','score'] if 'return_fields' not in kwargs else kwargs['return_fields'] #return fields
        size=1000 if 'size' not in kwargs else kwargs['size']
        max_year=0 if 'max_year' not in kwargs else kwargs['max_year']
    #    parser='edismax' if 'parser' not in kwargs else kwargs['parser']
        qf_fields=[s.split("^")[0] for s in qf.split()]
        qf_boosts=[1 if len(s.split("^"))==1 else float(s.split("^")[1]) for s in qf.split()]
        qff=[f for f,b in zip(qf_fields,qf_boosts) if b!=0]
        qfb=[b for f,b in zip(qf_fields,qf_boosts) if b!=0]
        boost_dict={}
        for f,b in zip(qff, qfb):
            boost_dict[f]=b

        if 'verb' in kwargs:
            print(text)
            print(qf)
            print()
        output=[]
        if len(bm25_params)>0:
            w = scoring.BM25F(**bm25_params)
        else:
            w = scoring.BM25F()
            print('Default scoring')
        with index.searcher(weighting=w) as searcher:
            query = MultifieldParser(qff, index.schema,
                                     fieldboosts=boost_dict,
                                     group=OrGroup).parse(text)
            if max_year>0:
                mask_q = QueryParser("year", index.schema).parse("date_i:["+str(max_year)+" to]")
                results = searcher.search(query, limit=size, mask=mask_q)
            else:
                results = searcher.search(query, limit=size)
            for r in results:
                results_row={}
                results_row['score']=r.score
                for f in return_fields:
                    if f not in results_row:
    #                    print(r)
                        if f in r:
                            results_row[f]=r[f]
                        else:
                            results_row[f]=''
                output.append(results_row)
        return output, return_fields

#solr = pysolr.Solr("http://130.155.204.198:8983/solr/trec-cds-2016", timeout=1200)
#res1=run_query('adult^1 elderly^1 man^1 calf^1 pain^1 walking^1 uphill^1 history^1 ischemic^1 heart^1 disease^1 worsening^1 hypertension^1 despite^1 medication^1 compliance^1 physical^1 exam^1 right^1 carotid^1 bruit^1 lower^1 extremities^1 cool^1 diminished^1 dorsalis^1 pedis^1 pulses^1', solr, qf='title_text_en^2 abstract_text_en^2 body_text_en^1.1',max_year=2013, size=5)
#res2=run_query('adult^1 elderly^1 man^1 calf^1 pain^1 walking^1 uphill^1 history^1 ischemic^1 heart^1 disease^1 worsening^1 hypertension^1 despite^1 medication^1 compliance^1 physical^1 exam^1 right^1 carotid^1 bruit^1 lower^1 extremities^1 cool^1 diminished^1 dorsalis^1 pedis^1 pulses^1', solr, qf='text^1',max_year=2013, size=5)
#
#rs=[res1, res2]
#
#for r in rs:
#    for line in r:
#        print (line)
#    print ()