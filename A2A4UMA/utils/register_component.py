#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:56:16 2019

@author: ryb003
"""

from distutils.dir_util import copy_tree
import utils_io as io
import os
path=os.path.dirname(__file__)+'/'


## mode 'alter' - changing meta config
# mode 'move' - copies and deletes source
# mode copy - copies from source
def register_component(name, source_dir, is_ranking, fixed_params={}, usr_params=[], model_id_path_pairs={}, mode='alter_replace'):
    
    conf=io.read_a2a_request_config()
    
    if mode in set(['alter_replace', 'copy', 'move']):
        conf['user_modifiable_params'][name]=usr_params
        if not is_ranking:
            conf['arg2mod'][name]=name
            conf['arg2params'][name]=fixed_params
            conf['arg2input_suffix'][name]='_name'
            conf['arg2operand'][name]='add'
        else:
            for model_id in model_id_path_pairs:    
                conf['arg2rank'][model_id]=[name,model_id_path_pairs[model_id]]
    
    ##TODO save the changes back to config
    
    if mode is 'move' or mode is 'copy':
        target_path=path+'../core/'+(is_ranking*'ranking/')
        os.makedirs(target_path, exist_ok=True)
        copy_tree(source_dir, target_path)
    
    
    if mode is 'move':
        #TODO delete stuff
        print('Removing source files')