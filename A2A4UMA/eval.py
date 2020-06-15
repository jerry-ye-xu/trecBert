import argparse
import json
import os
import multiprocessing
import pandas as pd
import pysolr
import time
import sys

from math import exp
from sklearn.model_selection import KFold 
from whoosh.index import open_dir

path = os.path.dirname(__file__)
sys.path.insert(0, path+'/../')
import bert_seq_class

sys.path.insert(0, path+'/core')

import pipe_conductor as pc

from BertSeqClassProcessData import generate_kfold_split, process_baseline_raw
from BertSeqClassGlobalVar import global_var, global_var_cv

from dynamic_file_generation_for_kfold_cv_eval import reduce_topics_for_eval, reduce_qrels_for_eval

def process_results(filepath,  stats={'P_10', 'Rprec', 'recip_rank'}):
    for root, dirs, files in os.walk(filepath[0]):
        for file in files:
            if file.endswith(".treceval"):
                filepath=os.path.join(root, file)
                f = open(filepath, "r")
                mean_results={}
                per_topic_results={}
                for x in f:
                    l=x.split()
                    if len(l)>0 and l[1] in ['all'] and l[0] in stats:
                        mean_results[l[0]]=float(l[2])
                    if len(l)>0 and l[1] not in ['all'] and l[0] in stats:
                        if int(l[1]) not in per_topic_results:
                            per_topic_results[int(l[1])]={}
                        print(f"int(l[1]): {int(l[1])}")
                        print(f"l[0]: {l[0]}")
                        print(f"float(l[2]): {float(l[2])}")
                        per_topic_results[int(l[1])][l[0]]=float(l[2])
                # print(per_topic_results[int(l[1])])
                f.close()
                if 'infNDCG' in stats:
                    f = open(filepath.replace('.treceval', '.sampleeval'), "r")
                    for x in f:
                        l=x.split()
                        if l[0]=='infNDCG' and l[1]=='all':
                            mean_results[l[0]]=float(l[2])
                        elif l[0]=='infNDCG':
                            per_topic_results[int(l[1])][l[0]]=float(l[2])
                    f.close()
                return mean_results, per_topic_results

#####################################################
# vector x (por indice)
# 0-8 importancias (pesos) de los 9 campos
# 9-17 parametros de saturacion de cada uno de los campos
# 18 - K1 (parametro del modelo)
# 19 - w (parametro de interpolacion de importancia para los campos de las consultas)
# 20 - year: version del conjunto de datos (queda fijo en una evaluacion)

### RANGOS
# pesos: [0;1]
# saturaciones: [0; 1]
# K1 [0; 3]
# w [0;1]
# year {2017,2018, 2019}

###
# Nota 1: si importnacia de peso p_j == 0 -> saturacion b_j no interviene en el modelo, ej:
# si x[0]==0 -> da igual el valor de x[9]
# si x[1]==0 -> da igual el valor de x[10], etc...

def eval(fields, field_boosts, bm25f_weights, K1, field_weights, year, args_params, rerank, ix, proc=1):
    # Load bm25f_configs
    V = field_boosts; B = bm25f_weights; K1 = K1; w = field_weights; year = year

    if year not in [2017, 2018, 2019]:
        return (-1, None)

    topics = path+'/data/topics/topics'+str(year)+'.xml'
    qrels = {'trec':path+'/data/qrels/qrels-treceval-'+str(year)+'-ct.txt'}

    # if year != 2017:
    qrels['sample'] = path+'/data/qrels/qrels-sample-'+str(year)+'-ct.txt'

    bm25f_conf = {'K1': K1}
    for f, b in zip(fields, B):
        bm25f_conf[f + '_B'] = b

    save_trec_name = "test_run" if "save_trec_name" not in args_params else args_params["save_trec_name"]

    run_title = "{}_{}_{}_{}_{}".format(save_trec_name, year, args_params["start_n"], args_params["top_n"], args_params["ckpt_num"])
    query_specs = [{
        'title': run_title,
        # 'title': f"base_ranker_{year}",
        "topic_elements": ['disease', 'gene'],
         "parse_args": {
             'boosts': [w, 1-w],
             'pre_process': False,
             'index_fields': fields,
             'field_boosts': V
         },
         "search_args": {
             'size': 1000, 'max_year': 0,
             'return_fields': [
                 'id', 'score', 'gender',
                 'maximum_age', 'minimum_age'
             ],
         'bm25f_params': bm25f_conf
         }
    }]

    piperun_dict = []
    print(f"piperun_dict: {piperun_dict}")
    rankings, topics, folders = pc.run_full_process(
        ix, topics, piperun_dict,
        query_specs, rerank,
        qrels_files=qrels,
        out_dir=path+'/results/',
        do_sample='sample' in qrels,
        proc=proc
    )

    time.sleep(3)

    ix.close()
    print("rankings (not printing")
    # print(rankings)
    print("topics")
    print(topics)
    print("folders")
    print(folders)
    mean, per_topic = process_results(folders)

    mean['used_fields'] = sum([int(v > 0) for v in V])
    return mean

def eval_bm25_edismax(
    x, rerank, query_fields,
    i_path, topics, qrels,
    proc=1, tids={}, solr_conf_base='./solr-8.2.0/server/solr/'):
    V=x[:len(query_fields)]
    # B=x[len(fields):2*len(fields)]
    K1=x[len(query_fields)]
    B=x[len(query_fields)+1]
    w=x[len(query_fields)+2]
    print('K1: '+ str (K1))
    print('B: '+ str (B))
    print('w: '+ str (w))

    year=int(x[len(query_fields)+3])
    if year not in [2017, 2018, 2019]:
        print(f"year is not correct: {year}")
        return (-1, None)

    stats={'P_10', 'Rprec', 'recip_rank'}
    if year != 2017:
        qrels['sample'] = path+'/data/qrels/qrels-sample-'+str(year)+'-ct.txt'
        stats=stats={'P_10', 'Rprec', 'recip_rank', 'infNDCG'}

    solr=pysolr.Solr(i_path, timeout=1200)

    bm25f_conf={'k1':K1, 'b':B, 'schema':solr_conf_base+i_path.split('/')[-1] + '/conf/managed-schema'}

    piperun_dict=[]
    save_trec_name = "test_run" if "save_trec_name" not in args_params else args_params["save_trec_name"]

    # Adjust base ranker info here.
    run_title = "{}_{}_{}_{}_{}_{}".format(save_trec_name, year, args_params["start_n"], args_params["top_n"], args_params["base_ranker_type"], args_params["ckpt_num"])
    query_specs = [{
        'title': run_title,
        # 'title': f"base_ranker_{year}",
        "topic_elements": ['disease', 'gene'],
         "parse_args": {
             'boosts': [w, 1-w],
             'pre_process': False,
             'index_fields': query_fields,
             'field_boosts': V
         },
         "search_args": {
             'size': 1000, 'max_year': 0,
             'return_fields': [
                 'id', 'score', 'gender',
                 'maximum_age', 'minimum_age'
             ],
         'bm25f_params': bm25f_conf
         }
    }]
    print(f"tids: {tids}")
    # time.sleep(10)
    rankings, topics, folders=pc.run_full_process(solr, topics, piperun_dict,
                                               query_specs, rerank,
                                               qrels_files=qrels, out_dir=path+'/results/',
                                               do_sample='sample' in qrels,
                                               proc=proc, tids=tids)

    mean, per_topic=process_results(folders, stats=stats)
    print(f"tids: {tids}")
    time.sleep(10)
    if len(tids) == 1:
        mean=per_topic[int(list(tids)[0])]
    mean['used_fields']=sum([int(v>0) for v in V])

    return mean, folders

def eval_bm25_edismax_kfold_cv(
    x, rerank, query_fields,
    i_paths, topics_path, qrels_path,
    training_data_path,
    kfold_cv, num_folds, seed,
    proc=1, tids={},
    solr_conf_base='./solr-8.2.0/server/solr/',
    stats={'P_10', 'Rprec', 'recip_rank'}):

    V=x[:len(query_fields)]
    # B=x[len(fields):2*len(fields)]
    K1=x[len(query_fields)]
    B=x[len(query_fields)+1]
    w=x[len(query_fields)+2]
    print('K1: '+ str (K1))
    print('B: '+ str (B))
    print('w: '+ str (w))

    # stats={'P_10', 'Rprec', 'recip_rank'}
    # if year!=2017:
    #     qrels['sample']=path+'/data/qrels/qrels-sample-'+str(year)+'-ct.txt'
    #     stats=stats={'P_10', 'Rprec', 'recip_rank', 'infNDCG'}

    solr2017 = pysolr.Solr(i_paths[0], timeout=1200)
    solr2019 = pysolr.Solr(i_paths[1], timeout=1200)

    bm25f_conf={'k1': K1, 'b': B, 'schema': solr_conf_base+i_path.split('/')[-1] + '/conf/managed-schema'}


    piperun_dict=[]
    save_trec_name = "test_run" if "save_trec_name" not in args_params else args_params["save_trec_name"]

    # Adjust base ranker info here.
    # The batch number is added during the cv for-loop
    run_title = "{}_{}_{}_{}_seed{}_num_folds{}".format(save_trec_name, args_params["start_n"], args_params["top_n"], args_params["base_ranker_type"],
        seed, num_folds
    )
    query_specs = [{
        'title': run_title,
        # 'title': f"base_ranker_{year}",
        "topic_elements": ['disease', 'gene'],
         "parse_args": {
             'boosts': [w, 1-w],
             'pre_process': False,
             'index_fields': query_fields,
             'field_boosts': V
         },
         "search_args": {
             'size': 1000, 'max_year': 0,
             'return_fields': [
                 'id', 'score', 'gender',
                 'maximum_age', 'minimum_age'
             ],
         'bm25f_params': bm25f_conf
         }
    }]

    #
    # Dynamic alter topics in combined_topics.xml
    # Dynamically alter qrels file
    #

    cv_path = f"{path}/dynamic_cv_files"
    if not os.path.exists(cv_path):
        os.makedirs(cv_path)

    df = pd.read_pickle(training_data_path)
    df_input = process_baseline_raw(
        df, rerank[1]["params"]["use_qe"],
        rerank[1]["params"]["n_chars_trim"],
        global_var_cv
    )
    orig2yrtop_dict = pd.read_pickle(global_var_cv["orig2yrtop_dict_path"])
    df_input["topics_all"] = df_input["year_topic"].map(orig2yrtop_dict)
    df_input["topics_all"].astype(int)

    print("df_input unique topics")
    print(global_var_cv["topic"])
    print(sorted(df_input["topics_all"].unique()))
    print(df_input["topics_all"].nunique())
    # time.sleep(15)
    split_gen = generate_kfold_split(df_input, global_var_cv["topic"], seed, num_folds)

    #k_split = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # gen_obj = 

    #for train, test in k_split.split(df_input["topics_all"].unique()):
    #    print(train)
    #    print(test)
    #    break

    #time.sleep(15)

    batch_num = 1

    list_ckpt = rerank[1]["params"]["list_ckpt_num"]

    if len(list_ckpt) != num_folds:
        print(f"num_folds: {num_folds}")
        print(f"list_ckpt: {list_ckpt}")
        raise ValueError("Number of checkpoints specified (not necessarily unique) should be identical to the number of folds you use for CV.")

    tmp_name = rerank[1]["params"]["model_name"]
    print(f"tmp_name: {tmp_name}")
    # time.sleep(3)

    for training_topic_idx_set, test_topic_idx_set in split_gen:
        #if batch_num <= 4:
        #    print(f"Skipping the {batch_num} batch_number")
        #    #print(f"batch_num less than 4: {batch_num}")
        #    batch_num += 1
        #    time.sleep(2)
        #    continue
        training_topic_set = [df_input["topics_all"].unique()[i] for i in training_topic_idx_set]
        test_topic_set = [df_input["topics_all"].unique()[i] for i in test_topic_idx_set]
        rerank[1]["params"]["batch_num"] = batch_num
        rerank[1]["params"]["ckpt_num"] = list_ckpt[batch_num-1]
        topics_name = f"topics_folds{num_folds}_seed{seed}_batch{batch_num}.xml"

        print(training_topic_set)
        print(test_topic_set)
        print("len(training_topic_set)")
        print(len(training_topic_set))
        print("len(test_topic_set)")
        print(len(test_topic_set))
        # time.sleep(5)
        reduce_topics_for_eval(
            topics_path=topics_path,
            save_tmp_topics_path=f"{cv_path}/{topics_name}",
            training_topic_set=training_topic_set,
            tag="number"
        )

        col_names = ["topics_all", "unknown", "doc_id", "label"]
        qrels_name = f"qrels_folds{num_folds}_seed{seed}_batch{batch_num}.txt"
        reduce_qrels_for_eval(
            qrels_path=qrels_path,
            save_tmp_qrels_path=f"{cv_path}/{qrels_name}",
            col_names=col_names, training_topic_set=training_topic_set
        )

        dynamic_qrels = {
            'trec': f"{cv_path}/{qrels_name}"
        }
        topics_test_path = f"{cv_path}/{topics_name}"

        # Remove the last digit i.e. batch number and replace with the current
        # batch number
        query_specs[0]["title"] = "{}{}".format(query_specs[0]["title"][:-1], batch_num)

        rerank[1]["params"]["train_topic_set"] = training_topic_set
        rerank[1]["params"]["test_topic_set"] = test_topic_set

        rankings, topics, folders = pc.run_full_process_multi_index(
            [solr2017, solr2019], topics_test_path, piperun_dict,
            query_specs, rerank,
            qrels_files=dynamic_qrels, out_dir=path+'/results/',
            do_sample='sample' in dynamic_qrels,
            proc=proc, tids=tids
        )

        mean, per_topic=process_results(folders, stats=stats)
        if len(tids)==1:
            mean=per_topic[int(list(tids)[0])]
        mean['used_fields']=sum([int(v>0) for v in V])

        batch_num += 1
        # return mean, folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_pipeline",
        type=int,
        # action="store_true",
        required=False,
        help="[0, 1] boolean equivalent. Reduce the training data to a small subset to test the pipeline"
    )

    parser.add_argument(
        "--df_bert_path",
        type=str,
        help="Specify bert path to save the df_bert that will be generated. This temporary file will be loaded into the model for evaluation and then deleted at the end."
    )

    parser.add_argument(
        "--ltr_model",
        type=str,
        # action="store_true",
        help="Specify with LTR model to use. Depending on what is set here, the necessities for other flags may change."
    )

    parser.add_argument(
        "--trial_topic_path",
        default=None,
        type=str,
        required=False,
        help="Path of the trial-topic data. These will typically be related to the\"trials_topics_combined_all_years\" files. It may also coincide with the training data."
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=False,
        help="Specify the pretrained/finetuned model. This could be one of the baseline models e.g. \"bert-base-uncased\" or the path to a dir of the finetuned/ckpt model"
    )
    parser.add_argument(
        "--model_class_type",
        default=None,
        type=str,
        required=True,
        help="Specify the underlying architecture e.g. BERT or RoBERTa or XLM etc. Currently only supports BERT and RoBERTa"
    )

    # Not needed because we will not be saving anything during evaluation.
    # parser.add_argument(
    #     "--model_desc",
    #     default=None,
    #     type=str,
    #     required=False,
    #     help="Specify directory name of which finetuned model will be saved"
    # )

    parser.add_argument(
        "--save_ckpt_path",
        default="pretrained_models",
        type=str,
        required=False,
        help="Specify directory name of which finetuned model will be saved"
    )

    parser.add_argument(
        "--ckpt_num",
        default=None,
        type=int,
        required=False,
        help="Checkpoint number of saved finetuned model to load"
    )

    # Finetuning may use more than 1 year, but validation only uses 1 year!
    parser.add_argument(
        "--test_year",
        default=None,
        type=int,
        required=False,
        help="Which year to use as testing data"
    )
    parser.add_argument(
        "--train_year",
        default=None,
        nargs="+",
        type=int,
        required=False,
        help="Which years to use as training data"
    )

    parser.add_argument(
        "--use_gpu",
        type=int,
        required=False,
        help="Whether you want to use GPUs."
    )

    parser.add_argument(
        "--use_qe",
        action="store_true",
        help="Whether you want to use query expanded data"
    )

    parser.add_argument(
        "--n_chars_trim",
        default=100,
        type=int,
        required=False,
        help="Cut total characters for each expanded field in topics to <n_chars> specified"
    )
    parser.add_argument(
        "--start_n",
        default=0,
        type=int,
        required=False,
        help="How much of the top n preranking documents to maintain."
    )
    parser.add_argument(
        "--top_n",
        default=50,
        type=int,
        required=False,
        help="How much of the top n preranking documents to rerank."
    )

    parser.add_argument(
    "--save_trec_name",
        type=str,
        # action="store_true",
        required=False,
        help="name for the results, does not include model"
    )

    parser.add_argument(
    "--use_solr",
        action="store_true",
        help="use solr index. If false, will use Whoosh index instead."
    )

    parser.add_argument(
    "--base_ranker_type",
        type=str,
        required=True,
        help="Specify base ranker for Lucene (use_solr). Options are [DFR, BM25]"
    )

    parser.add_argument(
        "--kfold_cv",
        action="store_true",
        help="whether to use k-fold cross validation. You must also specify the k-parameter and "
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="number of folds for the k-fold."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed to use for k-fold splitting."
    )
    parser.add_argument(
        "--topic_year_breakpoints",
        default=None,
        nargs="+",
        type=int,
        required=False,
        help="Which years to use as training data"
    )
    parser.add_argument(
        "--list_years",
        default=None,
        nargs="+",
        type=int,
        required=False,
        help="Which years to use as training data"
    )
    parser.add_argument(
        "--list_ckpt_num",
        default=None,
        nargs="+",
        type=int,
        required=False,
        help="checkpoint numbers for each fold"
    )

    args = parser.parse_args()
    args_params = vars(args)

    print(args_params["test_year"])

    fields = [
        "brief_title", "official_title", "brief_summary",
        "detailed_description", "intervention_type",
        "intervention_name", "inclusion", "keywords",
        "condition_browse", "primary_outcome", "exclusion"
    ]

    # default value is 1
    field_boosts = [None] * 11
    field_boosts[0] = 0.16
    field_boosts[1] = 0.08
    field_boosts[2] = 0.78
    field_boosts[3] = 0.02
    field_boosts[4] = 1.0
    field_boosts[5] = 0.076
    field_boosts[6] = 0.628
    field_boosts[7] = 0.375
    field_boosts[8] = 0.02
    field_boosts[9] = 1.0
    field_boosts[10] = 1.0

    # bm25f_weights = [0.75]*11
    bm25f_weights = [1] * 11
    bm25f_weights[0] = 0.507
    bm25f_weights[1] = 0.567
    bm25f_weights[3] = 0.6
    bm25f_weights[8] = 0.1

    K1 = 0.455
    field_weights = 0.5
    # year = 2017

    print("Visual check of boosts, weights and parameters")
    print(f"fields: {fields}")
    print(f"field_boosts: {field_boosts}")
    print(f"bm25f_weights: {bm25f_weights}")
    print(f"K1: {K1}")
    print(f"field_weights: {field_weights}")
    # print(f"year: {year}")

    if args_params["test_year"] != 2019:
        if args_params["use_solr"]:
            i_path='http://localhost:8983/solr/ct2017'
        else:
            i_path = path+"/indices/ct17_whoosh"
    else:
        if args_params["use_solr"]:
            i_path='http://localhost:8983/solr/ct2019'
        else:
            i_path = path+"/indices/ct19_whoosh"

    if not args_params["use_solr"]:
        ix = open_dir(i_path)

    # BERT MODEL Variables
    num_labels = 2
    batch_size = 8
    use_ids = True
    use_topics = True
    df_missing_save_path = "./data/missing_cache"

    # We don't need both columns as they are produced
    # during the evaluation run, and if include them
    # into the subset columns then the pd.merge will
    # generate suffixes [id_x, id_y] etc.
    subset_columns = [
        # "id", "topic",
        "brief_summary",
        "brief_title",
        "disease", "gene",
        "id_{}".format(global_var["topic"])
    ]

    if args_params["use_qe"] and "qe_all" not in subset_columns:
        subset_columns.append("qe_all")

    if args_params["use_solr"]:
        return_fields = [
            "id", "score",
            "brief_title", "official_title", "brief_summary",
            "detailed_description", "intervention_type",
            "intervention_name", "inclusion", "keywords",
            "condition_browse", "primary_outcome", "exclusion"
        ]
    else:
        return_fields = [
            "score",
            "id", "brief_summary", "brief_title",
            "minimum_age", "gender",
            "primary_outcome", "detailed_description",
            "keywords", "official_title",
            "intervention_type",
            "intervention_name",
            "intervention_browse",
            "condition_browse",
            "inclusion", "exclusion",
        ]
    # For QE
    fields_to_keep = [
        "disease", "disease_kbqe_pn", "disease_kbqe_syn",
        "gene", "gene_kbqe_syn", "gene_kbqe_other"
    ]

    seq_a_baseline = "d_and_g"
    seq_b_baseline = "brief_t_and_s"
    seq_a_expansion = "qe_all"
    bert_score = "bert_score"

    save_flag = False # MUST be False, since we are evaluating.

    rerank = []
    if args_params["ltr_model"] == "base_ranker":
        rerank.append({
            'module': 'demo_filter',
            'output': 'r',
            'params': {
                'inputs': [0],
                'model_path': i_path,
                'test_pipeline': args_params["test_pipeline"]
            },
            'title_suffix': '_Filtered'
        })
        if args_params["kfold_cv"]:
            rerank[1]["params"]["kfold_cv"] = args_params["kfold_cv"]
            rerank[1]["params"]["num_folds"] = args_params["num_folds"]
            rerank[1]["params"]["seed"] = args_params["seed"]
            # rerank[1] batch will be slot in later

            rerank[1]["params"]["idx_path_2017"] = "http://localhost:8983/solr/ct2017"
            rerank[1]["params"]["idx_path_2019"] = "http://localhost:8983/solr/ct2019"

            # for parse_topics in build_bert_ready_dataset
            # this is the same for 'all_topics_path'
            cv_files_all = "./data/cv_files"
            rerank[1]["params"]['topics_path'] = cv_files_all
            rerank[1]["params"]['topics_xml'] = "topics_combined.xml"
            rerank[1]["params"]['topic_year_breakpoints'] = args_params["topic_year_breakpoints"]
            rerank[1]["params"]['list_years'] = args_params["list_years"]
            rerank[1]["params"]['list_ckpt_num'] = args_params["list_ckpt_num"]

            print("topic_year_breakpoints")
            print(rerank[1]["params"]['topic_year_breakpoints'])
            print("list_years")
            print(rerank[1]["params"]['list_years'])
            print("subset_columns for kfold_cv")
            rerank[1]["params"]['subset_columns'] = [
                    # global_var_cv["topic"],
                    "brief_summary",
                    "brief_title",
                    "disease", "gene",
                    "id_{}".format(global_var_cv["topic"])
                ]
            time.sleep(10)
    elif (args_params["ltr_model"] == "bert") or (args_params["ltr_model"] == "roberta"):
        # We apply the demographics filter here first before
        # passing into BERT reranker.
        rerank.append({
            'module': 'demo_filter',
            'output': 'r',
            'params': {
                'inputs': [0],
                'model_path':i_path,
                'test_pipeline': args_params["test_pipeline"]
            },
            'title_suffix': '_Filtered'
            })
        rerank.append({
            'module': 'bert_model_eval',
            'output': 'r',
            'params': {
                'kfold_cv': args_params['kfold_cv'],
                'test_year': args_params["test_year"],
                'train_year': args_params["train_year"],
                'start_n': args_params["start_n"],
                'top_n': args_params["top_n"],
                'idx_path': i_path,
                'topics_path': "./data/pm_labels_{}".format(args_params["test_year"]),
                'topics_xml': "topics{}.xml".format(args_params["test_year"]),
                'trial_topic_path': args_params["trial_topic_path"],
                'df_missing_save_path': df_missing_save_path,
                'subset_columns': subset_columns,
                'return_fields': return_fields,
                'fields_to_keep': fields_to_keep,
                'df_cols': [
                    seq_a_baseline,
                    seq_b_baseline,
                    seq_a_expansion,
                    bert_score
                ],
                'ckpt_num': args_params["ckpt_num"],
                'save_ckpt_path': args_params["save_ckpt_path"],
                'model_name': args_params["model_name"],
                'model_class_type': args_params["model_class_type"],
                'base_ranker_type': args_params["base_ranker_type"],
                # 'model_desc': args_params["model_desc"],
                'test_pipeline': args_params["test_pipeline"],
                'use_qe': args_params["use_qe"],
                'math_func': exp,
                'num_labels': num_labels,
                'batch_size': batch_size,
                'use_ids': use_ids,
                'use_gpu': args_params["use_gpu"],
                'n_chars_trim': args_params["n_chars_trim"],
                'save_flag': save_flag,
                'save_ckpt_path': args_params["save_ckpt_path"]
            },
            'title_suffix': f'_{args_params["model_name"]}'
        })
        if args_params["kfold_cv"]:
            rerank[1]["params"]["kfold_cv"] = args_params["kfold_cv"]
            rerank[1]["params"]["num_folds"] = args_params["num_folds"]
            rerank[1]["params"]["seed"] = args_params["seed"]
            # rerank[1] batch will be slot in later

            rerank[1]["params"]["idx_path_2017"] = "http://localhost:8983/solr/ct2017"
            rerank[1]["params"]["idx_path_2019"] = "http://localhost:8983/solr/ct2019"

            # for parse_topics in build_bert_ready_dataset
            # this is the same for 'all_topics_path'
            cv_files_all = "./data/cv_files"
            rerank[1]["params"]['topics_path'] = cv_files_all
            rerank[1]["params"]['topics_xml'] = "topics_combined.xml"
            rerank[1]["params"]['topic_year_breakpoints'] = args_params["topic_year_breakpoints"]
            rerank[1]["params"]['list_years'] = args_params["list_years"]
            rerank[1]["params"]['list_ckpt_num'] = args_params["list_ckpt_num"]

            print("topic_year_breakpoints")
            print(rerank[1]["params"]['topic_year_breakpoints'])
            print("list_years")
            print(rerank[1]["params"]['list_years'])
            print("subset_columns for kfold_cv")
            rerank[1]["params"]['subset_columns'] = [
                    # global_var_cv["topic"],
                    "brief_summary",
                    "brief_title",
                    "disease", "gene",
                    "id_{}".format(global_var_cv["topic"])
                ]

            # time.sleep(20)

    elif args_params["ltr_model"] == "bert_sent":
            rerank.append({
                'module': 'bert_sent_model_eval',
                'output': 'r',
                'params': {
                    'kfold_cv': args_params['kfold_cv'],
                    'test_year': args_params["test_year"],
                    # 'train_year': args_params["train_year"],
                    'start_n': args_params["start_n"],
                    'top_n': args_params["top_n"],
                    'idx_path': i_path,
                    'df_bert_path': args_params["df_bert_path"],
                    'trial_topic_path': args_params["trial_topic_path"],
                    'df_missing_save_path': df_missing_save_path,
                    'subset_columns': subset_columns,
                    'return_fields': return_fields,
                    'fields_to_keep': fields_to_keep,
                    'df_cols': [
                        seq_a_baseline,
                        seq_b_baseline,
                        seq_a_expansion,
                        bert_score
                    ],
                    'ckpt_num': args_params["ckpt_num"],
                    'save_ckpt_path': args_params["save_ckpt_path"],
                    'model_name': args_params["model_name"],
                    'base_ranker_type': args_params["base_ranker_type"],
                    # 'model_desc': args_params["model_desc"],
                    'test_pipeline': args_params["test_pipeline"],
                    'use_qe': args_params["use_qe"],
                    'pool_function': max,
                    'math_func': exp,
                    'num_labels': num_labels,
                    'batch_size': batch_size,
                    'use_topics': use_topics,
                    'use_gpu': args_params["use_gpu"],
                    'n_chars_trim': args_params["n_chars_trim"],
                    'save_flag': save_flag,
                    'save_ckpt_path': args_params["save_ckpt_path"]
                },
                'title_suffix': f'_{args_params["model_name"]}'
            })
    else:
        print("ltr_model is not one of [base_ranker, bert, bert_sent]")

    if not args_params["use_solr"] and not args_params["kfold_cv"]:
        ranking = eval(
            return_fields, field_boosts,
            bm25f_weights,
            K1, field_weights, args_params["test_year"],
            args_params, rerank,
            ix=ix, proc=1,
        )

    print(len(return_fields))
    print(args_params.items())
    print("check args_params")
    print(args_params.keys())
    print(args_params.values())
    # time.sleep(10)

    if args_params["use_solr"] and not args_params["kfold_cv"]:
        # The default configs work best.
        x = [1]*1 + [1.2] + [0.75] + [0.5] + [args_params["test_year"]]
        topics= path+'/data/topics/topics'+str(args_params["test_year"])+'.xml'
        qrels={'trec':path+'/data/qrels/qrels-treceval-'+str(args_params["test_year"])+'-ct.txt'}

        eval_bm25_edismax(
            x, rerank=rerank, query_fields=["text"],
            i_path=i_path, topics=topics, qrels=qrels,
            proc=1, tids={}
        )

    if args_params["use_solr"] and args_params["kfold_cv"]:
        # The default configs work best.

        #
        # @TODO: create topics_combined.xml file
        #

        training_data_path = "./data/trials_topics_combined_all_years.pickle"

        all_topics_path = f"{cv_files_all}/topics_combined.xml"
        all_qrels_path = f"{cv_files_all}/qrels_combined.txt"
        topics = f"{path}/{all_topics_path}"

        # we'll just have to load both.
        i_path_2017 = 'http://localhost:8983/solr/ct2017'
        i_path_2019 = 'http://localhost:8983/solr/ct2019'

        x = [1]*1 + [1.2] + [0.75] + [0.5]

        eval_bm25_edismax_kfold_cv(
            x, rerank=rerank, query_fields=["text"],
            i_paths=[i_path_2017, i_path_2019],
            topics_path=all_topics_path, qrels_path=all_qrels_path,
            training_data_path=training_data_path,
            kfold_cv=args_params["kfold_cv"], num_folds=args_params["num_folds"], seed=args_params["seed"],
            proc=1, tids={}
        )

# '../data/topics/topics'+str(2017)+'.xml'
