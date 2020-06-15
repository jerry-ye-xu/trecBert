import argparse
import json

import pandas as pd

from transformers import BertModel, BertConfig, BertTokenizer,BertForSequenceClassification
from typing import Any, List, Dict, Tuple, Sequence, Optional

# from bert_seq_sent.BertSeqClassProcessData import *
from bert_seq_sent.BertSeqSentFinetune import *
from bert_seq_sent.BertSeqSentGlobalVar import global_var

from local_parser.output_reranking import create_id_topic_key, add_scores_to_df, update_scores

from bert_seq_sent.evalFunc import num_correctly_classified, calc_roc

CONFIG_FILE = "config.json"
MODEL_WEIGHTS = "pytorch_model.bin"
VOCAB_FILE = "vocab.txt"
TOKEN_CLASS = BertTokenizer

def run_training(
    model_name: str,
    model_desc: str,
    load_model_path: str,
    load_model_type: str,
    save_ckpt_path: str,
    save_flag: bool,
    ckpt_num: Optional[int],
    bioasq_train_path: str,
    bioasq_test_path: str,
    max_token_len: int,
    eval_metrics: List[str],
    use_gpu: bool, use_topics: bool,
    batch_size: int, epochs: int,
    num_labels: int,
    use_qe: bool,
    test_year: Optional[int],
    n_chars_trim: Optional[int]) -> Optional[Any]:
    """

    Params
    ------
    model_name: name of model, see load_model_type parameter
    model_desc: description of model, used in saving checkpoints
    load_model_type: One of [preloaded, pretrained, checkpoint].
                'preloaded' - base model from HuggingFace e.g. "bert-base-uncased"
                'pretrained' - load BERT variant e.g. sciBERT; "model_name" is the path.
                'checkpoint' - load finetuned BERT from checkpoint; "model_name" is the path right before the checkpoint directory.
    save_ckpt_path: path where finetuned model checkpoints are to be saved
    save_flag: boolean whether to save ckpts or not. This is not necessary if we are running test data.
    ckpt_num: checkpoint number that contains saved model during finetuning
    training_data: dataset for training
    test_data:     dataset for testing

    """

    MODEL_DESC = f"{model_desc}"
    TOKEN_CLASS = BertTokenizer
    model_path = ""

    if load_model_type == "preloaded":
        # MODEL_NAME = f"{model_name}"
        CONFIG = BertConfig.from_pretrained(model_name, num_labels=num_labels)

    elif load_model_type == "pretrained":
        model_path = f"{load_model_path}/{model_name}" # a path, osdir was set earlier (in Colab).
        CONFIG = BertConfig.from_pretrained(f"{model_path}/{CONFIG_FILE}", num_labels=num_labels)

    elif load_model_type == "checkpoint":
        model_path = f"{save_ckpt_path}/{model_name}" # a path, osdir was set earlier.
        CONFIG = BertConfig.from_pretrained(f"{model_path}/checkpoint-{ckpt_num}", num_labels=num_labels)

    else:
        raise ValueError("load_model must be one of [\"preloaded\", \"pretrained\", \"checkpoint\"].")

    print(f"load_model_type: {load_model_type}")
    if load_model_type == "preloaded":
        print(f"model_name: {model_name}")
        print(f"MODEL_DESC: {MODEL_DESC}")
        bert_model = BertForSentFinetune(
            model_name, CONFIG, num_labels, MODEL_DESC,
            save_ckpt_path=save_ckpt_path, save_flag=save_flag,
            vocab_file=None, model_weights=None
        )
    else:
        print(f"model_path: {model_path}")
        print(f"MODEL_DESC: {MODEL_DESC}")
        bert_model = BertForSentFinetune(
            model_path, CONFIG, num_labels, MODEL_DESC,
            save_ckpt_path=save_ckpt_path, save_flag=save_flag,
            vocab_file=VOCAB_FILE, model_weights=MODEL_WEIGHTS
        )

    bert_model.max_token_len = max_token_len
    global_steps, total_training_loss = bert_model.train(
        epochs, batch_size, use_gpu,
        bioasq_train_path
    )

    eval_metrics_dict = {}
    if "accuracy" in eval_metrics:
        eval_metrics_dict["accuracy"] = num_correctly_classified
    if "roc_curve" in eval_metrics:
        eval_metrics_dict["roc_curve"] = calc_roc

    if bioasq_test_path is not None:
        y_truth, y_pred, qa_ids_arr, validation_acc = bert_model.evaluate(
            bioasq_test_path,
            eval_metrics_dict,
            use_topics=use_topics,
            specify_dataset="bioasq",
            validate=False,
            use_gpu=use_gpu,
            batch_size=batch_size,
            global_var=global_var,
            use_qe=use_qe,
            test_year=test_year,
            n_chars_trim=n_chars_trim
        )

    return y_truth, y_pred, qa_ids_arr, validation_acc

def run_validate_data(
    df_bert_path: str,
    finetune_model_path: str,
    model_name: str,
    # model_desc: str, # no need for this
    save_ckpt_path: str,
    save_flag: bool,
    ckpt_num: Optional[int],
    num_labels: int,
    batch_size: int,
    use_topics: bool,
    use_gpu: int,
    use_qe: bool,
    test_year: int,
    n_chars_trim: int) -> List[List[np.array]]:

    """

    Note: labels_validate is not available for this.

    Params
    ------
    finetune_model_path: path where finetuned models are saved.
    model_name: name of model, this should be identical to
    model_desc: description of model, used in saving checkpoints (not needed in this function)
    ckpt_num: checkpoint number that contains saved model during finetuning
    validation_data: dataset for validation; if validation_data is used, then there is no need for training and testing data.

    """

    MODEL_NAME = f"{finetune_model_path}/{model_name}/checkpoint-{ckpt_num}" # a path, osdir was set earlier.
    CONFIG_FILE = "config.json"
    MODEL_WEIGHTS = "pytorch_model.bin"
    VOCAB_FILE = "vocab.txt"
    CONFIG = BertConfig.from_pretrained(f"{MODEL_NAME}", num_labels=num_labels)
    MODEL_DESC = f"{model_name}"
    TOKEN_CLASS = BertTokenizer

    bert_model = BertForSentFinetune(
        MODEL_NAME, CONFIG, num_labels, MODEL_DESC,
        save_ckpt_path=save_ckpt_path, save_flag=save_flag,
        vocab_file=VOCAB_FILE, model_weights=MODEL_WEIGHTS
    )

    _, reg_pred_arr, topics_id, doc_id, _, = bert_model.evaluate(
        eval_data_path=df_bert_path,
        eval_metrics_dict={},
        use_topics=True,
        specify_dataset="clinical",
        validate=True,
        use_gpu=use_gpu,
        batch_size=batch_size,
        global_var=global_var,
        use_qe=use_qe,
        test_year=test_year,
        n_chars_trim=n_chars_trim,
    )

    print("set(topics_id)")
    print(set(topics_id))

    return reg_pred_arr, topics_id, doc_id

print("BertSeqClassRun refreshed")

if __name__ == "__main__":
    pass