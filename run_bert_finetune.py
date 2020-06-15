#import argparse
#import json

import pandas as pd
#import time 

#from transformers import BertModel, BertConfig, BertTokenizer,BertForSequenceClassification
from typing import Any, List, Dict, Tuple, Sequence, Optional

#from bert_seq_class.BertSeqClassProcessData import *
#from bert_seq_class.BertSeqClassFinetune import *
#from bert_seq_class.BertSeqClassDataLoader import *
#from bert_seq_class.BertSeqClassRun import run_training
#from bert_seq_class.BertSeqClassGlobalVar import global_var, global_var_cv

def run_finetune_cv(
    test_pipeline: bool,
    training_data_path: str,
    load_pretrained_path: str,
    model_name: str,
    model_class_type: str,
    model_desc: str,
    save_ckpt_path: str,
    save_flag: bool,
    load_model_type: str,
    ckpt_num: Optional[int],
    max_token_len: int,
    eval_metrics: List[str],
    use_gpu: bool, use_ids: bool, use_qe: bool,
    batch_size: int, epochs: int,
    num_labels: int, n_chars_trim: int,
    global_var_cv: Dict[str, Any],
    kfold_cv: bool,
    num_folds: Optional[int],
    seed: Optional[int]) -> Optional[Any]:

    df = pd.read_pickle(training_data_path)
    df_input = process_baseline_raw(df, use_qe, n_chars_trim, global_var_cv)

    orig2yrtop_dict = pd.read_pickle(global_var_cv["orig2yrtop_dict_path"])
    df_input["topics_all"] = df_input["year_topic"].map(orig2yrtop_dict)
    df_input["topics_all"].astype(int)
    print(df_input[["year_topic", "year", "topic", "id", "topics_all"]].head())
    print(f"df_input.topics_all.nunique(): {df_input.topics_all.nunique()}")
    print(f"df_input.topics_all.unique(): {df_input.topics_all.unique()}")

    return

    if use_qe:
        seq_a_col = global_var_cv["seq_a_expansion"]
    else:
        seq_a_col = global_var_cv["seq_a_baseline"]

    split_gen = generate_kfold_split(
        df_input, global_var_cv["topic"],
        seed, num_folds
    )

    batch_num = 1
    for train_set_topics_idx, test_set_topics_idx in split_gen:
        #if batch_num > 4:
        #    print("Skipping the fifth batch_number")
        #    #print(f"batch_num less than 4: {batch_num}")
        #    batch_num += 1
        #    time.sleep(2)
        #    continue
        train_set_topics = [df_input["topics_all"].unique()[i] for i in train_set_topics_idx]
        test_set_topics = [df_input["topics_all"].unique()[i] for i in test_set_topics_idx]
        if test_pipeline:
            topic_subset = [train_set_topics[i] for i in range(1, 9)]
        else:
            topic_subset = None

        model_desc_batch = f"{model_desc}_folds{num_folds}_seed{seed}/{model_desc}_folds{num_folds}_seed{seed}_batch_num{batch_num}"
        run_finetune(
            df_input,
            test_pipeline,
            training_data_path,
            load_pretrained_path,
            model_name,
            model_class_type,
            model_desc_batch,
            save_ckpt_path,
            save_flag,
            load_model_type,
            ckpt_num,
            None, # train_year
            None, # test_year
            max_token_len,
            eval_metrics,
            use_gpu,
            use_ids,
            use_qe,
            batch_size,
            epochs,
            num_labels,
            n_chars_trim,
            global_var_cv,
            kfold_cv,
            train_set_topics,
            test_set_topics
        )
        batch_num += 1

def run_finetune(
    df_input: pd.DataFrame,
    test_pipeline: bool,
    training_data_path: str,
    load_pretrained_path: str,
    model_name: str,
    model_class_type: str,
    model_desc: str,
    save_ckpt_path: str,
    save_flag: bool,
    load_model_type: str,
    ckpt_num: Optional[int],
    train_year: Optional[List[int]],
    test_year: Optional[List[int]],
    max_token_len: int,
    eval_metrics: List[str],
    use_gpu: bool, use_ids: bool, use_qe: bool,
    batch_size: int, epochs: int,
    num_labels: int, n_chars_trim: int,
    global_var: Dict[str, Any],
    kfold_cv: bool = None,
    train_set_topics: Optional[List[int]] = None,
    test_set_topics: Optional[List[int]] = None) -> Optional[Any]:

    if test_pipeline:
        topic_subset = [i for i in range(1, 9)]
    else:
        topic_subset = None

    print(f"global_var: {global_var}")

    if use_qe:
        seq_a_col = global_var["seq_a_expansion"]
    else:
        seq_a_col = global_var["seq_a_baseline"]

    seq_a_train, seq_b_train, \
    labels_train, attribute_seq_train, doc_ids_train, \
    seq_a_test, seq_b_test, \
    labels_test, attribute_seq_test, doc_ids_test = split_into_train_test_set(
        df_input,
        seq_a_col,
        global_var["seq_b_baseline"],
        train_year,
        test_year,
        topic_subset=topic_subset,
        use_qe=use_qe,
        global_var=global_var,
        kfold_cv=kfold_cv,
        train_set_topics=train_set_topics,
        test_set_topics=test_set_topics
    )

    training_data = (seq_a_train, seq_b_train, labels_train)
    testing_data = (seq_a_test, seq_b_test, labels_test, attribute_seq_test, doc_ids_test)

    output = run_training(
        model_class_type,
        model_name,
        model_desc,
        load_pretrained_path,
        load_model_type,
        save_ckpt_path,
        save_flag, # no need to save model if testing
        ckpt_num,
        training_data,
        testing_data,
        max_token_len,
        eval_metrics, # we create the dictionary inside
        use_gpu, use_ids,
        batch_size, epochs, num_labels
    )

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_pipeline",
        type=int,
        # action="store_true",
        required=True,
        help="[0, 1] boolean equivalent. Reduce the training data to a small subset to test the pipeline"
    )
    parser.add_argument(
        "--training_data_path",
        default=None,
        type=str,
        required=True,
        help="Path of the training data. These will typically be related to the\"trials_topics_combined_all_years\" files"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="Specify the pretrained/finetuned model. This could be one of the baseline models e.g. \"bert-base-uncased\" or the path to a dir of the finetuned/ckpt model"
    )
    parser.add_argument(
        "--model_desc",
        default=None,
        type=str,
        required=True,
        help="Specify directory name of which finetuned model will be saved"
    )
    parser.add_argument(
        "--load_pretrained_path",
        # default="save_model",
        type=str,
        required=True,
        help="Specify directory name of where pretrained model is saved"
    )
    parser.add_argument(
        "--load_model_type",
        default=None,
        type=str,
        required=True,
        help="Load a preloaded (e.g. bert-base-uncased), pretrained (e.g. sciBERT) or ckpt model. The options are [preloaded, pretrained, checkpoint]. Coordinate with --model_name flag"
    )
    parser.add_argument(
        "--model_class_type",
        default=None,
        type=str,
        required=True,
        help="Specify the underlying architecture e.g. BERT or RoBERTa or XLM etc. Currently only supports BERT and RoBERTa"
    )

    parser.add_argument(
        "--save_ckpt_path",
        # default="save_model",
        type=str,
        required=True,
        help="Specify directory name of which finetuned model will be saved"
    )
    parser.add_argument(
        "--ckpt_num",
        default=None,
        type=int,
        required=False,
        help="Checkpoint number of saved finetuned model to load"
    )

    parser.add_argument(
        "--save_flag",
        # action="store_true",
        type=int,
        required=True,
        help="Whether to save the checkpoint of the finetuned model"
    )
    parser.add_argument(
        "--save_argparser_filename",
        default="argparser_args.json",
        type=str,
        # action="store_true",
        required=False,
        help="Specify the JSON file name of args specified by argparser."
    )

    parser.add_argument(
        "--test_year",
        default=None,
        nargs="+",
        type=int,
        required=True,
        help="Which years to use as testing data"
    )
    parser.add_argument(
        "--train_year",
        default=None,
        nargs="+",
        type=int,
        required=True,
        help="Which years to use as training data"
    )

    parser.add_argument(
        "--max_token_len",
        default=None,
        type=int,
        required=True,
        help="Max len of token for <seq_a, seq_b>"
    )
    parser.add_argument(
        "--eval_metrics",
        default=None,
        nargs="+",
        type=str,
        required=True,
        help="Specify which metrics you want the testing data to be evaluated on. See evalFunc.py file for more details"
    )

    parser.add_argument(
        "--use_gpu",
        default=0,
        type=int,
        help="Specify how many (if any) gpus you want to use."
    )
    parser.add_argument(
        "--use_ids",
        action="store_true",
        help="Whether you want to use document ids (relevant for TREC PM)"
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
        "--batch_size",
        default=8,
        type=int,
        required=True,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        required=True,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--num_labels",
        default=2,
        type=int,
        required=True,
        help="Number of classes the data has"
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

    args = parser.parse_args()
    print(args)
    print(type(args))
    print(vars(args))

    if (args.load_model_type == "checkpoint") & (args.ckpt_num is not None):
        model_name = f"{args.model_name}_{args.train_year}"
    # elif (args.load_model_type == "pretrained"):
    #     model_name = f"{args.model_name}"
    else:
        model_name = args.model_name

    print(model_name)

    if args.kfold_cv:
        run_finetune_cv(
            args.test_pipeline,
            args.training_data_path,
            args.load_pretrained_path,
            model_name,
            args.model_class_type,
            f"{args.model_desc}",
            args.save_ckpt_path,
            args.save_flag,
            args.load_model_type,
            args.ckpt_num,
            args.max_token_len,
            args.eval_metrics,
            args.use_gpu,
            args.use_ids,
            args.use_qe,
            args.batch_size,
            args.epochs,
            args.num_labels,
            args.n_chars_trim,
            global_var_cv,
            args.kfold_cv,
            args.num_folds,
            args.seed
        )
        if args.save_flag:
            with open(f"{args.save_ckpt_path}/{args.model_desc}_folds{args.num_folds}_seed{args.seed}/argparser.json", "w") as f:
                json.dump(vars(args), f)
    else:
        df = pd.read_pickle(args.training_data_path)
        df_input = process_baseline_raw(df, args.use_qe, args.n_chars_trim, global_var)

        run_finetune(
            df_input,
            args.test_pipeline,
            args.training_data_path,
            args.load_pretrained_path,
            model_name,
            args.model_class_type,
            f"{args.model_desc}_{args.train_year}",
            args.save_ckpt_path,
            args.save_flag,
            args.load_model_type,
            args.ckpt_num,
            args.train_year,
            args.test_year,
            args.max_token_len,
            args.eval_metrics,
            args.use_gpu,
            args.use_ids,
            args.use_qe,
            args.batch_size,
            args.epochs,
            args.num_labels,
            args.n_chars_trim,
            global_var,
            None,
            None,
            None
        )
        # This should be saved to the same path
        # args_file = "finetuning_args.json"
        if args.save_flag:
            with open(f"{args.save_ckpt_path}/{args.model_desc}_{args.train_year}/argparser.json", "w") as f:
                json.dump(vars(args), f)
