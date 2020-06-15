#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:13:01 2019

@author: ryb003
"""

from itertools import product
import eval
import time as t
import random
import multiprocessing
import numpy as np

def line_search(ranges, steps, dependencies=[], g=0.8, metric='P_10', test=False, proc=1, rd=True):
    filename='logs/log_'+str(t.time()).split('.')[0]+'.txt'
    eval_count=0
    epoch=1
    best_vals=[(r[0]+ r[1])/2 for r in ranges]
    while epoch < 31:
        with open(filename, 'a') as file:
            file.write('[EPOCH]\t'+str(epoch)+'\n')
#        add random order here
        i=0
        idx=list(range(len(ranges)))
        if rd and epoch>1:
            np.random.shuffle(idx)
        for i in idx:
            r=ranges[i]
            if r[1]-r[0]>steps[i]:
                val=r[0]
                score_to_beat=0
                best_arg=-1
                while val<=r[1]:
                    best_vals[i]=val
    #                mean=eval(best_vals)
                    if test:
                        mean={}
                        mean[metric]=random.uniform(r[0], r[1])
                        eval_count+=1
                    else:
                        mean=eval.eval(best_vals, proc=proc)
                        with open(filename, 'a') as file:
                            file.write('[EVAL]\t'+str(best_vals)+'\t'+str(mean)+'\n')
                    if mean[metric]>score_to_beat:
                        best_arg=val
                        score_to_beat=mean[metric]
                    val+=steps[i]
                    ##
                with open(filename, 'a') as file:
                    file.write('[BEST]\t'+str(i)+'\t'+str(best_arg)+'\n')
                best_vals[i]=best_arg
                
        ## at the end of epoch recalculate ranges
        i=0
        for r in ranges:
            new_span = g*(r[1]-r[0])
            if best_vals[i]+new_span/2>=r[1]:
                r[0]=r[1]-new_span
            elif best_vals[i]-new_span/2<=r[0]:
                r[1]=r[0]+new_span
            else:
                r[1]=best_vals[i]+new_span/2
                r[0]=best_vals[i]-new_span/2
            i+=1
        epoch+=1
    return best_vals, eval_count
        
best, count=line_search([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1],
                  [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,1], [2017, 2017]],
                  [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 1], test=False, proc=multiprocessing.Pool(10))
