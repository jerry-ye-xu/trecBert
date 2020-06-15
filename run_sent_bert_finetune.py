import argparse
import json

import pandas as pd

from transformers import BertModel, BertConfig, BertTokenizer,BertForSequenceClassification
from typing import Any, List, Dict, Tuple, Sequence, Optional

# from bert_seq_sent.BertSeqClassProcessData import *
from bert_seq_sent.BertSeqSentFinetune import *
# from bert_seq_bert_seq_sent.BertSeqClassDataLoader import *
from bert_seq_sent.BertSeqSentRun import run_training
from bert_seq_sent.BertSeqSentGlobalVar import global_var

def run_finetune(
    test_pipeline: bool,
    training_data_path: str,
    testing_data_path: str,
    load_pretrained_path: str,
    model_name: str,
    model_desc: str,
    save_ckpt_path: str,
    save_flag: bool,
    load_model_type: str,
    ckpt_num: Optional[int],
    test_year: List[int],
    max_token_len: int,
    eval_metrics: List[str],
    use_gpu: bool, use_ids: bool, use_qe: bool,
    batch_size: int, epochs: int,
    num_labels: int, n_chars_trim: int,
    global_var: Dict[str, Any]) -> Optional[Any]:

    if test_pipeline:
        training_data_path = f"./data/bioasq_df_train_tiny_sample.pickle"
        testing_data_path = f"./data/bioasq_df_test_tiny_sample.pickle"

    output = run_training(
        model_name,
        model_desc,
        load_pretrained_path,
        load_model_type,
        save_ckpt_path,
        save_flag, # no need to save model if testing
        ckpt_num,
        training_data_path,
        testing_data_path,
        max_token_len,
        eval_metrics, # we create the dictionary inside
        use_gpu, use_ids,
        batch_size, epochs, num_labels,
        use_qe, test_year, n_chars_trim
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
        help="Path of the bioasq training data."
    )
    parser.add_argument(
        "--testing_data_path",
        default=None,
        type=str,
        required=True,
        help="Path of the bioasq testing data."
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

    # parser.add_argument(
    #     "--test_year",
    #     default=None,
    #     nargs="+",
    #     type=int,
    #     required=True,
    #     help="Which years to use as testing data"
    # )
    # parser.add_argument(
    #     "--train_year",
    #     default=None,
    #     nargs="+",
    #     type=int,
    #     required=True,
    #     help="Which years to use as training data"
    # )

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
    # parser.add_argument(
    #     "--use_qe",
    #     action="store_true",
    #     help="Whether you want to use query expanded data"
    # )
    # parser.add_argument(
    #     "--n_chars_trim",
    #     default=100,
    #     type=int,
    #     required=False,
    #     help="Cut total characters for each expanded field in topics to <n_chars> specified"
    # )

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
        "--use_qe",
        action="store_true",
        help="Whether to use QE, for quick evaluation"
    )
    parser.add_argument(
        "--test_year",
        type=int,
        help="Test year for clinical trial data to do evaluation."
    )
    parser.add_argument(
        "--n_chars_trim",
        default=200,
        type=int,
        help="character trim for "
    )

    args = parser.parse_args()
    print(args)
    print(type(args))
    print(vars(args))

    if (args.load_model_type == "checkpoint") & (args.ckpt_num is not None):
        model_name = f"{args.model_name}"
    # elif (args.load_model_type == "pretrained"):
    #     model_name = f"{args.model_name}"
    else:
        model_name = args.model_name

    print(model_name)

    run_finetune(
        args.test_pipeline,
        args.training_data_path,
        args.testing_data_path,
        args.load_pretrained_path,
        model_name,
        f"{args.model_desc}",
        args.save_ckpt_path,
        args.save_flag,
        args.load_model_type,
        args.ckpt_num,
        args.test_year,
        args.max_token_len,
        args.eval_metrics,
        args.use_gpu, args.use_ids, args.use_qe,
        args.batch_size, args.epochs,
        args.num_labels, args.n_chars_trim,
        global_var
    )
    # This should be saved to the same path
    # args_file = "finetuning_args.json"
    if args.save_flag:
        with open(f"{args.save_ckpt_path}/{args.model_desc}/argparser.json", "w") as f:
            json.dump(vars(args), f)
