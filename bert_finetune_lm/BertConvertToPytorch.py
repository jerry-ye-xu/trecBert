from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertForPreTraining, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer


def save_model_in_pt_format(INPUT_PATH, OUTPUT_PATH):
    CONFIG = BertConfig(
        f"{INPUT_PATH}/config.json"
        , num_labels=2
    )

    model = BertForSequenceClassification.from_pretrained(
        f"{INPUT_PATH}/pytorch_model.bin",
        config=CONFIG
    )
    model.save_pretrained(CONFIG)

    # Slightly different for tokenizer.
    tokenizer = BertTokenizer.from_pretrained(
        f"{INPUT_PATH}/vocab.txt",
        do_lower_case=True
    )
    tokenizer.save_pretrained()

if __name__ == "__main__":
    MODEL_PATH = "./data/pretrained_models"
    INPUT_PATH = f"{MODEL_PATH}/BLUE_BERT"
    INPUT_PATH = f"{MODEL_PATH}/biobert_pubmed"
    OUTPUT_PATH = f"{MODEL_PATH}/BLUE_BERT_HF"
    save_model_in_pt_format(INPUT_PATH, OUTPUT_PATH)