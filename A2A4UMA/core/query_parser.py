#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:10:01 2019

@author: ryb003
"""
import os
import sys
path=os.path.dirname(__file__)
sys.path.insert(0, path+'/utils/')
import utils_io as io

path=os.path.dirname(__file__)
defaults=io.read_defaults(path)

#strategies = ['max', 'add', 'sub', 'rm']
def combine_boosts(query_terms, term, boost, strategy):
    if strategy=='max':
        if term in query_terms:
            query_terms[term]=max(query_terms[term], boost)
        else:
            query_terms[term]=boost
    if strategy=='add':
        if term in query_terms:
            query_terms[term]=query_terms[term]+ boost
        else:
            query_terms[term]=boost
    if strategy=='sub':
        if term in query_terms:
            query_terms[term]=max(query_terms[term]- boost,0)
    if strategy=='rm':
        if term in query_terms:
            del query_terms[term]

def parse_query(topic, fields, **kwargs):
    boosts = [1]*len(fields) if 'boosts' not in kwargs else kwargs['boosts']
    operations = ['max']*len(fields) if 'operations' not in kwargs else kwargs['operations']
    index_fields=defaults['index_fields'] if 'index_fields' not in kwargs else kwargs['index_fields']
    field_boosts=defaults['field_boosts'] if 'field_boosts' not in kwargs else kwargs['field_boosts']
    pre_process=True if 'pre_process' not in kwargs else kwargs['pre_process']
    
    if pre_process:
        query_terms={}
        for i in range(0,len(fields)):
            if fields[i] in topic:
                terms=topic[fields[i]].replace(',','').replace('.','').split()
                for term in terms:
                    combine_boosts(query_terms, term, boosts[i], operations[i])
        q_string=''
        for term in query_terms:
            q_string+=(term+ '^'+str(query_terms[term])+' ')
        q_string=q_string.strip()
    else:
        q_string=''
        i=0
        for field in fields:
            if field in topic:
                q_string+=('('+topic[field]+')')
                if boosts[i]:
                    q_string+='^'+str(boosts[i])+' '
                else:
                    q_string+=(' ')
            i+=1
    qf_string=''
    for i in range(0,len(index_fields)):
        qf_string+=(' '+index_fields[i]+'^'+str(field_boosts[i]))
    return q_string, qf_string.strip()

