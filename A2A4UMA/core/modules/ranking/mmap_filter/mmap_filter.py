#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:06:18 2019
"""
import sys
import subprocess
import shlex
import os
import copy as cp



path=os.path.join(os.path.dirname(__file__), 'conf')
f = open(path,"r") 
baseDir = f.readline().strip()
f.close()

_metamap_concepts = ["dsyn","sosy","patf","diap","anab","lbpr","phsu","neop","topp"]

True
def normalize_white(s):
    return ' '.join(s.replace('\n', ' ').split())

def parse_output(output_lines, input_text):
    result=[]
    input_str=normalize_white(input_text)
    if output_lines==None:
        return []
    for line in output_lines:
        parts = str(line).split('|')
        try:
            negated=parts[5].endswith('1')
            preferred_name=parts[2]
            off=[int(s) for s in parts[7].split(':')]
            offset=(off[0], off[1]+off[0])
            surface=input_str[offset[0]:offset[1]]
            result.append((preferred_name, offset, surface, negated))
        except IndexError:
            continue
    return result


#FIX ME #TODO
def metamaplite(inp, concepts, restriction):
    if inp==None:
        return []
    input_str=normalize_white(inp)
    _metamap_args = shlex.split(baseDir+'metamaplite.sh --usecontext --pipe --indexdir='+baseDir+'data/ivf/strict --modelsdir='+baseDir+'data/models --specialtermsfile='+baseDir+'data/specialterms.txt'+restriction*(' --restrict_to_sts='+'\''+'\',\''.join(concepts)+'\''))
    proc = subprocess.Popen(_metamap_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(' '.join(_metamap_args))
    sout, serr = proc.communicate((input_str + '\n').encode('utf-8'))
    if proc.returncode != 0:
        sys.exit(serr)
    return sout.splitlines()


def sort_by_score(record, score_index):
    return record[score_index]




def run(result_sets, topics, **kwargs):
    indexes = range(len(result_sets)) if 'inputs' not in kwargs else kwargs['inputs']
    score_col = 'score' if 'score_col' not in kwargs else kwargs['score_col']
#    id_col = 'id' if 'id_col' not in kwargs else kwargs['id_col']
    concept_restriction = True if 'concept_restriction' not in kwargs else kwargs['concept_restriction']
    concepts = _metamap_concepts if 'concepts' not in kwargs else kwargs['concepts']
    fields=[] if 'fields' not in kwargs else kwargs['fields']
    column='mmap' if 'column' not in kwargs else kwargs['column']
    
    final_results=[]
    for j in indexes:
    ## Re-rank
        result_set=cp.deepcopy(result_sets[j])
        ranking=result_set['ranking']
        columns=result_set['headings']
        columns.append(column)
        score_index=columns.index(score_col)
        
        for topic_id in ranking:
            for row in ranking[topic_id]:
                doc_text=''
                for f in fields:
                    if f in columns:
                        f_index=columns.index(f)
                        doc_text+= row[f_index] + ' '
                annotations=parse_output(metamaplite(doc_text, concepts, concept_restriction), doc_text)
                row.append(annotations)
        for topic_id in ranking:
            ranking[topic_id].sort(key = lambda record: sort_by_score(record, score_index), reverse = True)
        final_results.append(result_set)
    return final_results
        

#ad hoc testing
#rss=run(r, q_rel='../../../../data/qrels/qrels-treceval-2016.txt')
