import pickle

from statistics import mean

from bert_seq_sent.BertSeqSentFinetune import *
from bert_seq_sent.BertSeqSentRun import run_validate_data
from bert_seq_sent.BertSeqSentGlobalVar import global_var
from bert_seq_sent.BertSeqSentProcessData import process_validation_data_raw
from local_parser.ingest_preranking import build_dataset_for_bert

from bert_seq_sent.evalFunc import pool

from local_parser.output_reranking import create_id_topic_key, add_scores_to_df, update_scores

def run(result_sets, topics, **kwargs):
    # LOAD THE KWARGS
    test_year = kwargs["test_year"]
    # train_year = kwargs["train_year"]
    top_n = kwargs["top_n"]
    idx_path = kwargs["idx_path"]
    df_bert_path = kwargs["df_bert_path"]
    trial_topics_path = kwargs["trial_topic_path"]
    df_missing_save_path = kwargs["df_missing_save_path"]
    subset_columns = kwargs["subset_columns"]
    return_fields = kwargs["return_fields"]
    fields_to_keep = kwargs["fields_to_keep"]
    pool_function = kwargs["pool_function"]

    use_qe = kwargs["use_qe"]

    seq_a = kwargs["df_cols"][2] if use_qe else kwargs["df_cols"][0]
    seq_b = kwargs["df_cols"][1]
    bert_score = kwargs["df_cols"][3]

    model_name = kwargs["model_name"]
    base_ranker_type = kwargs["base_ranker_type"]
    # model_desc = kwargs["model_desc"]
    ckpt_num = kwargs["ckpt_num"]

    math_func = kwargs["math_func"]
    num_labels = kwargs["num_labels"]
    batch_size = kwargs["batch_size"]
    use_topics = kwargs["use_topics"]
    use_gpu = kwargs["use_gpu"]
    n_chars_trim = kwargs["n_chars_trim"]

    save_ckpt_path = kwargs["save_ckpt_path"]
    save_flag = kwargs["save_flag"]

    # BUILD DATASET FOR BERT

    # Extract the topic ids
    topics_set = [int(topics[i]['id']) for i in range(len(topics))]
    num_topics = len(topics_set)
    print(f"topics_set: {topics_set}")

    # Need to incorporate QE here
    df_bert = build_dataset_for_bert(
        result_sets, test_year, top_n, topics_set,
        subset_columns, return_fields,
        idx_path, trial_topics_path,
        df_missing_save_path, fields_to_keep,
        base_ranker_type
    )

    print(df_bert.columns)
    print(f"df_bert.topic.unique: {df_bert.topic.unique()}")
    print("df_bert.head()")
    print(df_bert.head())
    print(df_bert["disease"].isna().value_counts())
    print(df_bert["gene"].isna().value_counts())

    # PROCESS INTO BERT INPUT SEQ
    # if test:
        # df_bert.drop("id_y", inplace=True, axis=1)
        # df_bert = df_bert.rename(columns={"id_x": "id"})

    df_bert = process_validation_data_raw(df_bert, global_var)
    print(f"df_bert.topic.unique: {df_bert.topic.unique()}")
    print("df_bert.head() after process_validation_data_raw")
    print(df_bert.head())
    print(df_bert["disease"].isna().value_counts())
    print(df_bert["gene"].isna().value_counts())

    df_bert.to_pickle(f"{df_bert_path}")

    # save_ckpt_path is the finetuned path.
    # model_name and model_desc are identical when loading from ckpt (see Makefile)

    reg_pred_arr, topics_id, doc_id = run_validate_data(
        df_bert_path, save_ckpt_path,
        f"{model_name}",
        None, save_flag,
        ckpt_num, num_labels, batch_size,
        use_topics, use_gpu, use_qe,
        test_year, n_chars_trim,
    )

    print(f"topics_id: {topics_id}")

    with open("./debug_data/reg_pred_arr.pickle", "wb") as f:
        pickle.dump(reg_pred_arr, f)

    with open("./debug_data/topics_id.pickle", "wb") as f:
        pickle.dump(topics_id, f)

    with open("./debug_data/doc_id.pickle", "wb") as f:
        pickle.dump(doc_id, f)

    doc_id_pooled, topics_id_pooled, reg_pred_pooled = pool(
        doc_id, topics_id, reg_pred_arr, mean
    )

    bert_scores_dict = create_id_topic_key(
        doc_id_pooled, topics_id_pooled, reg_pred_pooled, math_func
    )
    with open("./debug_data/bert_scores_dict.pickle", "wb") as f:
        pickle.dump(bert_scores_dict, f)

    df_scores = add_scores_to_df(df_bert, bert_scores_dict)

    print("df_scores")

    with open("./debug_data/df_scores.pickle", "wb") as f:
        pickle.dump(df_scores, f)

    reranking_dict = update_scores(
        top_n,
        result_sets,
        df_scores[bert_score],
        topics_set
    )

    with open("./debug_data/reranking_dict.pickle", "wb") as f:
        pickle.dump(reranking_dict, f)

    return reranking_dict
