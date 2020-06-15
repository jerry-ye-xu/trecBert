# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:35:57 2019

@author: ryb003
"""

def run(topic_elements, topic, **kwargs):
    label_suffix = '_comb' if 'label_suffix' not in kwargs else kwargs['label_suffix']
    out_label='_'.join(topic_elements)
    string_parts=[topic[f] for f in topic_elements if f in topic]
    topic[out_label+label_suffix]=' '.join(string_parts)
    return topic# -*- coding: utf-8 -*-

#t=run(['a','b'], {'a':'1', 'b':'2'})