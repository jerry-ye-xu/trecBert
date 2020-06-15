#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:43:03 2020

@author: ryb003
"""
from subprocess import check_output
import xml.etree.ElementTree as ET

#stop solr given path

def stop_solr(path):
    out = check_output([path, 'stop'])
    print(out)
def start_solr(path):
    out = check_output([path, 'start'])
    print(out)

def modify_config(schema, b=.75, k1=1.2):
    tree=ET.parse(schema)
    sim=tree.find('similarity')
    change=False
    for child in sim:
        if 'name' in child.attrib and child.get('name')=='k1':
            if float(child.text)!=float(k1):
                child.text=str(k1)
                change=True

        elif 'name' in child.attrib and child.get('name')=='b':
            if float(child.text)!=float(b):
                child.text=str(b)
                change=True
    if change:
        tree.write(schema)
    return change

def set_params(schema, b, k1, solr_bin='./solr-8.2.0/bin/solr'):
    mod=modify_config(schema, b, k1)
    if mod:
        stop_solr(solr_bin)
        start_solr(solr_bin)

#set_params('/home/ryb003/solr-8.2.0/server/solr/ct2017/conf/managed-schema', .75, 1.2)





