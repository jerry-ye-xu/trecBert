#!/usr/bin/env python
import os
path=os.path.dirname(__file__)+'/'

import sys
sys.path.insert(0, path+'../utils/')

import copy

import utils_io as io
import index_utils as iu
import importlib
import query_parser as qp
import time as t
import numpy as np
import multiprocessing
from shutil import copyfile



# Pipes stdin into metamap program
# Prints a comma separated list of synonyms/concepts to stdout
defaults = io.read_defaults(path)

# filename - topic file
# fields - list of lists of input topic fields (elements) used at each step
# modules - list of names /one per execution step/
# kwarg_dicts - list of dictionaries
def run_topic_processing_pipeline(filename, step_config_maps, options={}):
    # parse the topics
    if isinstance(filename, str):
        topics, f = io.parse_topics(filename, options)
    else:
        topics = filename

    #initialize controller resources
    module_dict = {}
    for step in step_config_maps:
        if step['module'] not in module_dict:
            sys.path.insert(0, path+defaults['modules']+step['module'])
            mod = importlib.import_module(step['module'])
            module_dict[step['module']] = mod

    # execute the pipeline topic by topic
    for topic in topics:
        print(topic['id'])
        for step in step_config_maps:
            print(f"step: {step}")
            topic = module_dict[step['module']].run(
                step['fields'],
                topic,
                **step['params']
            )

    for m in module_dict:
        if 'cleanup' in dir(module_dict[m]):
            module_dict[m].cleanup()

    return topics

def run_ranking_pipeline(result_sets, topics, step_config_maps):
    for step in step_config_maps:

        print(f"The step in step_config_maps is:\n{step}")

        print(f"defaults[\'ranking_modules\']: {defaults['ranking_modules']}")
        print(f"step[\'module\']: {step['module']}")
        print(f"path: {path}")
        sys.path.insert(0, path+defaults['ranking_modules']+step['module'])

        print(path+defaults['ranking_modules']+step['module'])

        # bert_path = "core/modules/ranking/bert_models"
        # sys.path.insert(0, f"{path}/{bert_path}")
        # print(f"path: {path}")

        print("saving result_sets to pickle file")
        save_path_pre = "./A2A4UMA/pre_rerank_data_files/result_sets_pre_tmp_testing.pickle"

        io.save_pickle(result_sets, save_path_pre)

        mod = importlib.import_module(step['module'])
        print(f"mod: {mod}")

        # print(f"the topics are:\n{topics}")

        # results = result_sets
        results = mod.run(result_sets, topics, **step['params'])
        # print(results)
        print("saving results to pickle file, this is what post mod.run files should look like.")
        save_path_post = "./A2A4UMA/pre_rerank_data_files/results_post_tmp_testing.pickle"
        io.save_pickle(results, save_path_post)

        for r in results:
            if 'title_suffix' in step:
                r['title'] += step['title_suffix']
            else:
                r['title'] += ('_'+step['module'])

        # a = add and r = rerank?
        if step['output'] == 'a':
            result_sets.extend(results)
        elif step['output'] == 'r':
            result_sets = results

        # print(result_sets)

        # print(result_sets[0]["ranking"]["10"])
        # print(result_sets[0]["queries"]["10"])

        # for result_set in rankings:
        #     print(result_sets)


        # for k, v in result_sets[0]:
        #     print("key")
        #     print(k)
        #     print("value")
        #     print(v)

    return result_sets

def mp_worker(inputs):
    try:
        if inputs['has_qf']:
            ranking, rf=iu.run_query(inputs['q'], inputs['conn'], qf=inputs['qf'], **inputs['args'])
        else:
            ranking, rf=iu.run_query(inputs['q'], inputs['conn'], **inputs['args'])
        return (ranking,rf)
    except Exception as e:
        import traceback
        print ('Caught exception in worker thread ' + str(inputs))
        traceback.print_exc()
        raise e

# solr_conn is evident
# topics are the topics to parse queries from, so they are already processed, cleaned, expanded etc. it is a dictionary of topic elements
# topic_fields are a list of topic fields to be used in query aggregation, order matters
# other parser arguments cover the way the fields are aggregated into a query: operations (list; ('max'|'add'|'sub'|'rm')) and boosts (list; [num])  AND...
# ... index fields (list;[str]) and index fields boosts (list; [num]) ; if missing defaults will be used, although boosts without specifying fields will raise error
# other search args: comm. to specify year limit, size and return fields, defaulted if missing
def run_queries(solr_conn, topics, topic_fields,
                other_parser_args={}, proc=1,
                **other_search_args):

#     year_limit=2016, return_fields=['id', 'score']):
#     print ('PC:')
#     print(other_search_args)

    if proc == 1:
        results = {}
        queries = {}
        for topic in topics:
            q, qf = qp.parse_query(topic, topic_fields, **other_parser_args)
            print(f"q: {q}")
            print(f"qf: {qf}")
            if 'index_fields' in other_parser_args:
                    ranking, rf = iu.run_query(q, solr_conn, qf=qf, **other_search_args)
            else:
                ranking, rf = iu.run_query(q, solr_conn, **other_search_args)

            results[topic['id']] = []
            queries[topic['id']] = q

            for r in ranking:
                row = []
                for f in rf:
                    if f in r:
                        row.append(r[f])
                    else:
                        row.append('')
                results[topic['id']].append(row)
        return results, queries, rf
    else:
        data = []
        queries = {}
        results = {}
        for topic in topics:
            q, qf = qp.parse_query(topic, topic_fields, **other_parser_args)
            data_row = {
                'topic':topic, 'q':q, 'qf':qf,
                'args':other_search_args,
                'conn':solr_conn,
                'has_qf':'index_fields' in other_parser_args
            }
            data.append(data_row)
            queries[topic['id']] = q

        output = proc.map(mp_worker, data)

        i = 0
        for topic_out in output:
            results[topics[i]['id']]=[]
            ranking=topic_out[0]
            rf=topic_out[1]
            for r in ranking:
                row=[]
                for f in rf:
                    if f in r:
                        row.append(r[f])
                    else:
                        row.append('')
                results[topics[i]['id']].append(row)
            i+=1
        return results, queries, rf



 #one piperun spec / 1 dictionary for topic processing,
 #1..n query specs,  list of dictionaries
 # 1 fusion spec TODO
 # 1 solr connection for now
def run_full_process(solr_conn, topic_file, piperun_dicts,
                     query_exec_specs, reranking_steps,
                     out_dir=path+defaults['out_dir'],
                     other_columns=[], qrels_files={},
                     pipe='', query_specs='', ranking='',
                     do_sample=False, skip_eval=False,
                     skip_eval_by_id=[], proc=1, tids={}):
    rankings = []
    print("run_topic_processing_pipeline")
    topics = run_topic_processing_pipeline(topic_file, piperun_dicts)
    # print(f"topics: {topics}")

    # Subsetting during `run_topic_processing_pipeline` function is more clean but this makes it easier to see when tracking bugs.
    if reranking_steps[0]["params"]["test_pipeline"]:
        print("subsetting topics")
        # topics = topics[:2]
        topics = topics[:2]

    print("Running queries!")
    for query_step in query_exec_specs:
        r, q, col = run_queries(
            solr_conn, topics,
            query_step['topic_elements'],
            query_step['parse_args'],
            proc=proc,
            **query_step['search_args']
        )
        if 'title' not in query_step:
            # finds the firs part of a first fieldname used in query,
            title = query_step['topic_elements'][0].split('_')[0]
        else:
            title = query_step['title']
        # the above is notfolders rock solid, but since we use suffixes it should work
        rankings.append({
            'ranking': r,
            'queries': q,
            'headings': col,
            'title': title
        })

    # fusion and/or re-ranking goes here
    print("run_ranking_pipeline")
    rankings = run_ranking_pipeline(rankings, topics, reranking_steps)

    # write results
    print(f"writing results to {out_dir}")
    output_paths = []
    output_folders = []
    run_ids = []
    if len(out_dir) > 0:
        for result_set in rankings:
            run_id = result_set['title'] + '_' + str(int(t.time()*100))
            run_ids.append(run_id)
            values, columns = io.results_2_trec_format(result_set, run_id, other_columns)

            print("trec_vals")
            print(values)
            print("columns")
            print(columns)

            output_paths.append(out_dir + run_id + '/' + run_id + '.results')
            output_folders.append(out_dir + run_id + '/')

            os.makedirs(out_dir + run_id + '/', exist_ok=True)
            np.savetxt(output_paths[-1], values, delimiter="\t", fmt='%s')

    print(f"Evaluating results if skip_eval is true: {skip_eval}")
    # run evaluation program(s) and save results
    if not skip_eval:
        if len(qrels_files) > 0:
            for i in range(0,len(run_ids)):
                if i not in skip_eval_by_id:
                    io.evaluate_results(
                        output_paths[i], output_folders[i],
                        run_ids[i], qrels_files,
                        do_sample_eval=do_sample
                    )
                    if len(pipe) > 0:
                        copyfile(pipe, output_folders[i] + 'topic_processing.txt')
                    if len(query_specs) > 0:
                        copyfile(query_specs, output_folders[i] + 'query_specs.txt')
                    if len(ranking) > 0:
                        copyfile(ranking, output_folders[i] + 'ranking.txt')

    print("Returning rankings, topics, output_folders")
    return rankings, topics, output_folders

def run_full_process_multi_index(
    solr_conns, topic_file, piperun_dicts,
    query_exec_specs, reranking_steps,
    out_dir=path+defaults['out_dir'],
    other_columns=[], qrels_files={},
    pipe='', query_specs='', ranking='',
    do_sample=False, skip_eval=False,
    skip_eval_by_id=[], proc=1, tids={}):
    rankings = []
    print("run_topic_processing_pipeline")
    topics = run_topic_processing_pipeline(topic_file, piperun_dicts)
    # print(f"topics: {topics}")

    # Subsetting during `run_topic_processing_pipeline` function is more clean but this makes it easier to see when tracking bugs.
    if reranking_steps[0]["params"]["test_pipeline"]:
        print("subsetting topics")
        # topics = topics[:2]
        topics = topics[:2]

    print("Running queries!")
    for query_step in query_exec_specs:
        r, q, col = run_queries_multi_index(
            solr_conns, topics,
            query_step['topic_elements'],
            query_step['parse_args'],
            proc=proc,
            **query_step['search_args']
        )
        if 'title' not in query_step:
            # finds the firs part of a first fieldname used in query,
            title = query_step['topic_elements'][0].split('_')[0]
        else:
            title = query_step['title']
        # the above is notfolders rock solid, but since we use suffixes it should work
        rankings.append({
            'ranking': r,
            'queries': q,
            'headings': col,
            'title': title
        })

    # fusion and/or re-ranking goes here
    print("run_ranking_pipeline")
    rankings = run_ranking_pipeline(rankings, topics, reranking_steps)

    # write results
    print(f"writing results to {out_dir}")
    output_paths = []
    output_folders = []
    run_ids = []
    if len(out_dir) > 0:
        for result_set in rankings:
            run_id = result_set['title'] + '_' + str(int(t.time()*100))
            run_ids.append(run_id)
            values, columns = io.results_2_trec_format(result_set, run_id, other_columns)

            print("trec_vals")
            print(values)
            print("columns")
            print(columns)

            output_paths.append(out_dir + run_id + '/' + run_id + '.results')
            output_folders.append(out_dir + run_id + '/')

            os.makedirs(out_dir + run_id + '/', exist_ok=True)
            np.savetxt(output_paths[-1], values, delimiter="\t", fmt='%s')

    print(f"Evaluating results if skip_eval is true: {skip_eval}")
    # run evaluation program(s) and save results
    if not skip_eval:
        if len(qrels_files) > 0:
            for i in range(0,len(run_ids)):
                if i not in skip_eval_by_id:
                    io.evaluate_results(
                        output_paths[i], output_folders[i],
                        run_ids[i], qrels_files,
                        do_sample_eval=do_sample
                    )
                    if len(pipe) > 0:
                        copyfile(pipe, output_folders[i] + 'topic_processing.txt')
                    if len(query_specs) > 0:
                        copyfile(query_specs, output_folders[i] + 'query_specs.txt')
                    if len(ranking) > 0:
                        copyfile(ranking, output_folders[i] + 'ranking.txt')
            print("HELLO DO WE REACH HERE?")

    print("Returning rankings, topics, output_folders")
    return rankings, topics, output_folders

def run_queries_multi_index(
    solr_conns, topics, topic_fields,
    other_parser_args={}, proc=1,
    **other_search_args):

#     year_limit=2016, return_fields=['id', 'score']):
#     print ('PC:')
#     print(other_search_args)

    if proc == 1:
        results = {}
        queries = {}
        for topic in topics:
            q, qf = qp.parse_query(topic, topic_fields, **other_parser_args)
            print(f"q: {q}")
            print(f"qf: {qf}")

            #
            # @TODO: Extend this for general case (use switch statements?)
            #

            total_topic_id = int(topic["id"])
            print(f"combined_topics topic_id: {total_topic_id}")
            if total_topic_id > 80:
                print(f"Using solr_conns[1]: {solr_conns[1]}" )
                if 'index_fields' in other_parser_args:
                        ranking, rf = iu.run_query(q, solr_conns[1], qf=qf, **other_search_args)
                else:
                    ranking, rf = iu.run_query(q, solr_conns[1], **other_search_args)
            else:
                print(f"Using solr_conns[0]: {solr_conns[0]}" )
                if 'index_fields' in other_parser_args:
                        ranking, rf = iu.run_query(q, solr_conns[0], qf=qf, **other_search_args)
                else:
                    ranking, rf = iu.run_query(q, solr_conns[0], **other_search_args)

            results[topic['id']] = []
            queries[topic['id']] = q

            for r in ranking:
                row = []
                for f in rf:
                    if f in r:
                        row.append(r[f])
                    else:
                        row.append('')
                results[topic['id']].append(row)
        return results, queries, rf
    else:
        data = []
        queries = {}
        results = {}
        for topic in topics:
            q, qf = qp.parse_query(topic, topic_fields, **other_parser_args)
            data_row = {
                'topic':topic, 'q':q, 'qf':qf,
                'args':other_search_args,
                'conn':solr_conn,
                'has_qf':'index_fields' in other_parser_args
            }
            data.append(data_row)
            queries[topic['id']] = q

        output = proc.map(mp_worker, data)

        i = 0
        for topic_out in output:
            results[topics[i]['id']]=[]
            ranking=topic_out[0]
            rf=topic_out[1]
            for r in ranking:
                row=[]
                for f in rf:
                    if f in r:
                        row.append(r[f])
                    else:
                        row.append('')
                results[topics[i]['id']].append(row)
            i+=1
        return results, queries, rf

#    cleanup(path_entries_count, core_modules)

    print("Returning rankings, topics, output_folders")
    return rankings, topics, output_folders

