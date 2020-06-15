import time

from bert_seq_class.BertSeqClassProcessData import *
from bert_seq_class.BertSeqClassFinetune import *
from bert_seq_class.BertSeqClassDataLoader import *
from bert_seq_class.BertSeqClassRun import run_validate_data
from bert_seq_class.BertSeqClassGlobalVar import global_var, global_var_cv
from bert_seq_class.evalFunc import softmax

from local_parser.ingest_preranking import build_dataset_for_bert, build_dataset_for_bert_cv

from local_parser.output_reranking import create_id_topic_key, add_scores_to_df, update_scores

def run(result_sets, topics, **kwargs):
    # LOAD THE KWARGS
    kfold_cv = kwargs["kfold_cv"]
    if kfold_cv:
        train_topic_set = kwargs["train_topic_set"]
        test_topic_set = kwargs["test_topic_set"]

        idx_path_2017 = kwargs["idx_path_2017"]
        idx_path_2019 = kwargs["idx_path_2019"]

        topic_year_breakpoints = kwargs["topic_year_breakpoints"]
        list_years = kwargs["list_years"]

        num_folds = kwargs["num_folds"]
        seed = kwargs["seed"]
        batch_num = kwargs["batch_num"]

        topic_col = global_var_cv["topic"]
        GLOBAL_VAR = global_var_cv
    else:
        test_year = kwargs["test_year"]
        train_year = kwargs["train_year"]

        idx_path = kwargs["idx_path"]
        topic_col = global_var["topic"]
        GLOBAL_VAR = global_var

    start_n = kwargs["start_n"]
    top_n = kwargs["top_n"]

    trial_topics_path = kwargs["trial_topic_path"]
    df_missing_save_path = kwargs["df_missing_save_path"]
    subset_columns = kwargs["subset_columns"]
    return_fields = kwargs["return_fields"]
    fields_to_keep = kwargs["fields_to_keep"]

    use_qe = kwargs["use_qe"]

    seq_a = kwargs["df_cols"][2] if use_qe else kwargs["df_cols"][0]
    seq_b = kwargs["df_cols"][1]
    bert_score = kwargs["df_cols"][3]

    model_name = kwargs["model_name"]
    model_class_type = kwargs["model_class_type"]
    base_ranker_type = kwargs["base_ranker_type"]
    # model_desc = kwargs["model_desc"]
    ckpt_num = kwargs["ckpt_num"]

    math_func = kwargs["math_func"]
    num_labels = kwargs["num_labels"]
    batch_size = kwargs["batch_size"]
    use_ids = kwargs["use_ids"]
    use_gpu = kwargs["use_gpu"]
    n_chars_trim = kwargs["n_chars_trim"]

    save_ckpt_path = kwargs["save_ckpt_path"]
    save_flag = kwargs["save_flag"]

    topics_path = kwargs["topics_path"]
    topics_xml = kwargs["topics_xml"]

    # BUILD DATASET FOR BERT

    # Extract the topic ids
    topics_set = [int(topics[i]['id']) for i in range(len(topics))]
    num_topics = len(topics_set)
    print(f"topics_set: {topics_set}")
    print(num_topics)
    time.sleep(10)
    # Need to incorporate QE here
    if kfold_cv:
        df_bert = build_dataset_for_bert_cv(
            result_sets, top_n, test_topic_set,
            subset_columns, return_fields,
            [idx_path_2017, idx_path_2019],
            trial_topics_path,
            df_missing_save_path, fields_to_keep,
            base_ranker_type, # train_topic_set, test_topic_set,
            kfold_cv, topics_path, topics_xml,
            topic_col, topic_year_breakpoints, list_years
        )
    else:
        df_bert = build_dataset_for_bert(
            result_sets, test_year, top_n, topics_set,
            subset_columns, return_fields,
            idx_path, trial_topics_path,
            df_missing_save_path, fields_to_keep,
            base_ranker_type, topics_path, topics_xml, topic_col
        )
    print(df_bert.columns)
    print("df_bert.head()")
    print(df_bert.head())
    time.sleep(10)
    print(df_bert["disease"].isna().value_counts())
    print(df_bert["gene"].isna().value_counts())

    # PROCESS INTO BERT INPUT SEQ
    # if test:
        # df_bert.drop("id_y", inplace=True, axis=1)
        # df_bert = df_bert.rename(columns={"id_x": "id"})

    df_input = process_validation_data_raw(df_bert, GLOBAL_VAR)
    print("df_input.head()")
    print(df_input.head())
    print(df_input["disease"].isna().value_counts())
    print(df_input["gene"].isna().value_counts())

    seq_a_validate, seq_b_validate, \
    _, attribute_seq_validate, doc_ids_validate = split_into_bert_input(
        df_input,
        seq_a,
        seq_b,
        validate=True,
        use_qe=use_qe,
        global_var=GLOBAL_VAR
    )
    attribute_seq_validate = [int(topic_num) for topic_num in attribute_seq_validate]

    validation_data_for_bert = [
        seq_a_validate, seq_b_validate,
        attribute_seq_validate, doc_ids_validate
    ]

    # save_ckpt_path is the finetuned path.
    # model_name and model_desc are identical when loading from ckpt (see Makefile)

    if kfold_cv:
        model_save_desc = f"{model_name}_folds{num_folds}_seed{seed}_batch_num{batch_num}"
    else:
        model_save_desc = f"{model_name}_{train_year}"

    reg_pred_arr, topics_id, doc_id = run_validate_data(
        save_ckpt_path, model_save_desc,
        model_class_type,
        None, save_flag,
        ckpt_num, num_labels, batch_size,
        use_ids, use_gpu,
        validation_data_for_bert
    )

    reg_pred_probs = softmax(reg_pred_arr)[:, 1]
    print("reg_pred_probs[:5]")
    print(reg_pred_probs[:5])
    print(len(reg_pred_probs))
    print("len(reg_pred_probs)")
    print("doc_id")
    print(doc_id)
    print("topics_id")
    print(topics_id)
    time.sleep(5)

    bert_scores_dict = create_id_topic_key(
        doc_id, topics_id, reg_pred_probs, math_func
    )

    print("bert_scores_dict")
    print(bert_scores_dict)
    # time.sleep(5)

    df_scores = add_scores_to_df(df_input, bert_scores_dict, GLOBAL_VAR)

    print("df_scores")
    print(df_scores)
    print("df_input")
    print(df_input.columns)
    time.sleep(15)

    reranking_dict = update_scores(
        start_n,
        top_n,
        result_sets,
        df_scores[bert_score],
        topics_set
    )

    return reranking_dict
