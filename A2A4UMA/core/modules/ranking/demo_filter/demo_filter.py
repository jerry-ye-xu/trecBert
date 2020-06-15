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
import copy as cp
import time



def sort_by_score(record, score_index):
    return record[score_index]

def get_doc_dict(row_col):
    gender_pos=row_col[1].index('gender')
    maximum_age_pos=row_col[1].index('maximum_age')
    minimum_age_pos=row_col[1].index('minimum_age')

    d={}

    if row_col[0][gender_pos]!='':
        d['gender']=row_col[0][gender_pos]
    if row_col[0][maximum_age_pos]!='':
        d['maximum_age']=row_col[0][maximum_age_pos]
    if row_col[0][minimum_age_pos]!='':
        d['minimum_age']=row_col[0][minimum_age_pos]

    return d

def run_eligibility_check(patient_dict, doc_id, row_col):
    age=patient_dict['age']
    gender=patient_dict['gender']

    doc=get_doc_dict(row_col)
    gender_inc=False
    # gender , max_age, min_age
    # gender
    if 'gender' not in doc or doc['gender'].lower()=='all':
        gender_inc=True
    elif gender.lower()=='female' and gender.lower() in doc['gender'].lower():
        gender_inc=True
    elif gender.lower()=='male' and gender.lower() in doc['gender'].lower():
        gender_inc=True

    # max_age
    max_age_inc=False
    if 'maximum_age' not in doc or int(doc['maximum_age'])==-1 or (int(doc['maximum_age'])/365.0)>=age:
        max_age_inc=True
    min_age_inc=False
    if 'minimum_age' not in doc or int(doc['minimum_age'])==-1 or (int(doc['minimum_age'])/365.0)<=(age+1): # relaxed constraint for close inclusion date?
        min_age_inc=True

    # if False in [gender_inc, min_age_inc, max_age_inc]:
    #     print(doc_id)
    #     print(patient_dict)
    #     print( [gender_inc, min_age_inc, max_age_inc])
    #     if 'gender' in doc:
    #         print(doc['gender'])
    #     else:
    #         print('No gender')

    #     if 'minimum_age' in doc:
    #         print(doc['minimum_age'])
    #     else:
    #         print('No minimum_age')

    #     if 'maximum_age' in doc:
    #         print(doc['maximum_age'])
    #     else:
    #         print('No maximum_age')

    # print(([gender_inc, min_age_inc, max_age_inc]))
    return min ([gender_inc, min_age_inc, max_age_inc])

def get_patient_info(demo_text):
    age=int(demo_text.split('-')[0])
    if 'female' in demo_text:
        gender='female'
    else:
        gender='male'
    return {'age':age, 'gender':gender}


def run(result_sets, topics, **kwargs):
    indexes = range(len(result_sets)) if 'inputs' not in kwargs else kwargs['inputs']
    score_col = 'score' if 'score_col' not in kwargs else kwargs['score_col']
    id_col = 'id' if 'id_col' not in kwargs else kwargs['id_col']


    final_results=[]
    for j in indexes:
    ## Re-rank
        result_set = cp.deepcopy(result_sets[j])
        ranking = result_set['ranking']
        columns = result_set['headings']
        # print(f"ranking: {ranking}")
        print(f"columns: {columns}")
        # time.sleep(10)
        score_index = columns.index(score_col)
        id_index = columns.index(id_col)

        print(f"id_index: {id_index}")
        print(f"score_index: {score_index}")
        # time.sleep(10)

        idx_counter = 0
        for topic_id in ranking:
            # since we have already filtered the test topics
            # the topic ids will no longer be in order and
            # thus you can't just simply '-1'.
            # int_id = int(topic_id) - 1
            print(f"----------------")
            print(f"idx_counter: {idx_counter}")
            print(f"topics[idx_counter]['id']: {topics[idx_counter][columns[0]]}")
            print(f"----------------")
            patient = get_patient_info(topics[idx_counter]['demographic'])
            for row in ranking[topic_id]:
                doc_id = row[id_index]
                if not run_eligibility_check(patient, doc_id, (row, columns)):
                    row[score_index] = 0

            idx_counter += 1

        for topic_id in ranking:
            ranking[topic_id].sort(key = lambda record: sort_by_score(record, score_index), reverse = True)
        final_results.append(result_set)
    return final_results


#ix=open_dir(path+"/../../../../indices/ct19_whoosh")
#doc=get_doc_dict('NCT00036816', ix)
#doc=iu.run_query('NCT00036816', ix, **{'qf':'id^1', 'return_fields':['gender', 'maximum_age', 'minimum_age']})[0][0]


#ad hoc testing
#rss=run(r, q_rel='../../../../data/qrels/qrels-treceval-2016.txt')
