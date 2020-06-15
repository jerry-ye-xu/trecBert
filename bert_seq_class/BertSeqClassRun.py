import time

from transformers import (
    BertConfig, BertTokenizer,
    RobertaConfig, RobertaTokenizer
)

from evalFunc import *

from BertSeqClassProcessData import *
from BertSeqClassFinetune import *
from BertSeqClassDataLoader import *

from local_parser.output_reranking import create_id_topic_key, add_scores_to_df, update_scores

def run_training(
    model_class_type: str,
    model_name: str,
    model_desc: str,
    load_model_path: str,
    load_model_type: str,
    save_ckpt_path: str,
    save_flag: bool,
    ckpt_num: Optional[int],
    training_data: List[List[Any]],
    testing_data: List[List[Any]],
    max_token_len: int,
    eval_metrics: List[str],
    use_gpu: bool, use_ids: bool,
    batch_size: int, epochs: int,
    num_labels: int) -> Optional[Any]:
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

    CONFIG_FILE = "config.json"
    MODEL_WEIGHTS = "pytorch_model.bin"
    VOCAB_FILE = "vocab.txt"

    TOKEN_CLASS = RobertaTokenizer if model_class_type == "roberta" else BertTokenizer
    CONFIG_CLASS = RobertaConfig if model_class_type == "roberta" else BertConfig
    MODEL_CLASS = RobertaForSequenceClassification if model_class_type == "roberta" else BertForSequenceClassification

    # var holder if the load_model_type
    model_path = ""
    MODEL_DESC = f"{model_desc}"

    if load_model_type == "preloaded":
        # MODEL_NAME = f"{model_name}"
        CONFIG = CONFIG_CLASS.from_pretrained(model_name, num_labels=num_labels)
        bert_dataloader = BertSeqClassDataHolder(
            model_name, CONFIG, num_labels, TOKEN_CLASS,
            vocab_file=None
        )
    elif load_model_type == "pretrained":
        model_path = f"{load_model_path}/{model_name}" # a path, osdir was set earlier (in Colab).
        CONFIG = CONFIG_CLASS.from_pretrained(f"{model_path}/{CONFIG_FILE}", num_labels=num_labels)

        bert_dataloader = BertSeqClassDataHolder(
            model_path, CONFIG, num_labels, TOKEN_CLASS, vocab_file=VOCAB_FILE
        )
    elif load_model_type == "checkpoint":
        model_path = f"{save_ckpt_path}/{model_name}" # a path, osdir was set earlier.
        CONFIG = CONFIG_CLASS.from_pretrained(f"{model_path}/checkpoint-{ckpt_num}", num_labels=num_labels)

        bert_dataloader = BertSeqClassDataHolder(
            model_path, CONFIG, num_labels, TOKEN_CLASS, vocab_file=VOCAB_FILE
        )
    else:
        raise ValueError("load_model must be one of [\"preloaded\", \"pretrained\", \"checkpoint\"].")

    # print(f"model_name: {model_name}")
    # print(f"model_path: {model_path}")

    seq_a_train, seq_b_train, labels_train = training_data
    seq_a_test, seq_b_test, labels_test,\
        attribute_seq_test, doc_ids_test = testing_data

    bert_dataloader.load_pre_split_train_dataset(
        seq_a_train, seq_b_train, labels_train,
        None, None,
        batch_size=batch_size
    )
    bert_dataloader.load_pre_split_test_dataset(
        seq_a_test, seq_b_test, labels_test,
        attribute_seq_test, doc_ids_test,
        batch_size=batch_size
    )

    training_data_loader, testing_data_loader = bert_dataloader.get_train_test_data()

    print(f"load_model_type: {load_model_type}")
    if load_model_type == "preloaded":
        print(f"model_name: {model_name}")
        print(f"MODEL_DESC: {MODEL_DESC}")
        bert_model = BertForSeqFinetune(
            model_name, CONFIG, num_labels, MODEL_DESC,
            save_ckpt_path=save_ckpt_path, save_flag=save_flag,
            hf_model_class=MODEL_CLASS, hf_token_class=TOKEN_CLASS,
            model_class_type=model_class_type,
            vocab_file=None, model_weights=None
        )
    else:
        print(f"model_path: {model_path}")
        print(f"MODEL_DESC: {MODEL_DESC}")
        bert_model = BertForSeqFinetune(
            model_path, CONFIG, num_labels, MODEL_DESC,
            save_ckpt_path=save_ckpt_path, save_flag=save_flag,
            hf_model_class=MODEL_CLASS, hf_token_class=TOKEN_CLASS,
            model_class_type=model_class_type,
            vocab_file=VOCAB_FILE, model_weights=MODEL_WEIGHTS
        )

    bert_model.load_dataloader_train_and_test(
        training_data_loader,
        testing_data_loader
    )

    bert_model.max_token_len = max_token_len
    global_steps, total_training_loss = bert_model.train(epochs, batch_size, use_gpu)

    eval_metrics_dict = {}
    if "accuracy" in eval_metrics:
        eval_metrics_dict["accuracy"] = num_correctly_classified
    if "precision_recall_by_topic" in eval_metrics:
        eval_metrics_dict["precision_recall_by_topic"] = eval_pr_per_topics
    if "roc_curve" in eval_metrics:
        eval_metrics_dict["roc_curve"] = calc_roc

    y_truth, y_pred, topics_arr, doc_ids_arr, \
    pr_dict, validation_acc = bert_model.evaluate(
        eval_metrics_dict,
        use_ids=use_ids,
        validate=False,
        use_gpu=use_gpu
    )

    return y_truth, y_pred, topics_arr, doc_ids_arr, pr_dict, validation_acc

def run_validate_data(
    finetune_model_path: str,
    model_name: str,
    model_class_type: str,
    # model_desc: str,
    save_ckpt_path: str,
    save_flag: bool,
    ckpt_num: Optional[int],
    num_labels: int,
    batch_size: int,
    use_ids: bool,
    use_gpu: int,
    validation_data: Optional[List[List[Any]]]) -> List[List[np.array]]:

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
    VOCAB_FILE = "vocab.json" if model_class_type == "roberta" else "vocab.txt"

    TOKEN_CLASS = RobertaTokenizer if model_class_type == "roberta" else BertTokenizer
    CONFIG_CLASS = RobertaConfig if model_class_type == "roberta" else BertConfig
    MODEL_CLASS = RobertaForSequenceClassification if model_class_type == "roberta" else BertForSequenceClassification

    CONFIG = CONFIG_CLASS.from_pretrained(f"{MODEL_NAME}", num_labels=num_labels)
    MODEL_DESC = f"{model_name}"

    print(CONFIG_CLASS)
    print(TOKEN_CLASS)
    print(MODEL_CLASS)
    time.sleep(10)

    bert_dataloader = BertSeqClassDataHolder(
        MODEL_NAME, CONFIG, num_labels,
        TOKEN_CLASS, vocab_file=VOCAB_FILE
    )

    seq_a_validate, seq_b_validate, \
        attribute_seq_validate, doc_ids_validate = validation_data

    print("set(attribute_seq_validate)")
    print(set(attribute_seq_validate))

    bert_dataloader.load_pre_split_test_dataset(
        seq_a_validate, seq_b_validate, None,
        attribute_seq_validate, doc_ids_validate,
        batch_size=batch_size
    )

    _, validate_data_loader = bert_dataloader.get_train_test_data()

    bert_model = BertForSeqFinetune(
        MODEL_NAME, CONFIG, num_labels, MODEL_DESC,
        save_ckpt_path=save_ckpt_path, save_flag=save_flag,
        hf_model_class=MODEL_CLASS, hf_token_class=TOKEN_CLASS,
        model_class_type=model_class_type,
        vocab_file=VOCAB_FILE, model_weights=MODEL_WEIGHTS
    )

    bert_model.load_dataloader_validate(
        validate_data_loader
    )

    _, reg_pred_arr, topics_id, doc_id, _, _ = bert_model.evaluate(
        eval_metrics_dict={},
        use_ids=True,
        validate=True,
        use_gpu=use_gpu
    )

    print("set(topics_id)")
    print(set(topics_id))

    return reg_pred_arr, topics_id, doc_id

print("BertSeqClassRun refreshed")

if __name__ == "__main__":
    pass