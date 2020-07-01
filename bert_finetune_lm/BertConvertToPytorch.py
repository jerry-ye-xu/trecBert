from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertForPreTraining, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer


def save_model_in_pt_format(INPUT_PATH, OUTPUT_PATH):
    CONFIG = BertConfig(
        f"{INPUT_PATH}"
    )

    print(CONFIG)

    model, loading_info = BertForPreTraining.from_pretrained(
        f"{INPUT_PATH}",
        output_loading_info=True
    )
    model.save_pretrained(OUTPUT_PATH)

    # Slightly different for tokenizer.
    tokenizer = BertTokenizer.from_pretrained(
        f"{INPUT_PATH}",
        do_lower_case=True
    )
    tokenizer.save_pretrained(OUTPUT_PATH)

if __name__ == "__main__":
    MODEL_PATH = "./data/pretrained_models"
    INPUT_PATH = f"{MODEL_PATH}/BLUE_BERT"
    OUTPUT_PATH = f"{MODEL_PATH}/BLUE_BERT_HF"
    save_model_in_pt_format(INPUT_PATH, OUTPUT_PATH)