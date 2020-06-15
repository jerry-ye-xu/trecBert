from tqdm import tqdm, trange
from typing import Any, List, Dict, Tuple, Sequence, Callable

import torch

from transformers import load_tf_weights_in_bert
from transformers import BertTokenizer, RobertaTokenizer

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from bert_seq_class.customDataset import DatasetWithStrID, id_collate

class BertSeqClassDataHolder():
    def __init__(
        self,
        model_name, config, num_labels,
        hf_token_class,
        vocab_file=None):
        """

        Params
        ------

        model_name: model_name e.g. "bert-base-uncased" or path
        config: BertConfig object that is initialised with the same model_name
        num_labels: number of classes for finetuning categorical data
        model_desc: description of model used to name checkpoints that are saved with training
        hf_model_class: HuggingFace token class

        """

        ###   MODEL VARIABLES   ###

        self.model_name = model_name
        self.config = config # initialised outside of class
        self.model = None
        self.tokenizer = None

        self.hf_token_class = hf_token_class

        ###   DATA VARIABLES   ###

        self.X_train = None
        self.X_test = None

        self.X_train_token_ids = None
        self.X_test_token_ids = None

        self.X_mask = None
        self.X_mask_test = None

        self.y_train = None
        self.y_test = None

        self.attrib_train = None
        self.attrib_test = None

        self.doc_ids_train = None
        self.doc_ids_test = None

        self.random_state = 2018
        self.max_token_len = 256

        self.num_labels = num_labels

        self._specify_token(
            self.model_name, self.config, self.num_labels,
            vocab_file=vocab_file
        )

        # What we provide the BERT models
        self.training_data_loader = None
        self.testing_data_loader = None

    def get_train_test_data(self) -> List[DataLoader]:
        return self.training_data_loader, self.testing_data_loader

    def get_bert_tokenizer(self) -> BertTokenizer:
        return self.hf_token_class

    #########    HELPER FUNCTIONS    #########

    def _specify_token(
        self,
        model_name, config, num_labels,
        vocab_file=None):
        """
        The naming conventions for loading a pretrained model is:

        "config.json"
        "vocab.txt"
        "pytorch_model.bin"

        To be explicit, we'll force the user to specify their files. The `config.json`
        file specified outside of the class, so account for the remaining two.

        If we are loading the files from Tensorflow, then we need to pass in a
        boolean (in this case from_tf)
        """

        if vocab_file is not None:
            self.tokenizer = self.hf_token_class.from_pretrained(
                f"{model_name}/{vocab_file}",
                do_lower_case=True
            )
        else:
            self.tokenizer = self.hf_token_class.from_pretrained(
                model_name,
                do_lower_case=True
            )

    def _truncate_seq_pair(self, pair_a, pair_b=None):
        """Truncates a sequence pair to the maximum length."""

        if pair_b is None:
            if len(pair_a) > (self.max_token_len - 2):
                return pair_a[:(self.max_token_len - 2)], pair_b
        else:
            while True:
                total_length = len(pair_a) + len(pair_b)
                if total_length <= (self.max_token_len - 3):
                    break
                if len(pair_a) > len(pair_b):
                    pair_a.pop()
                else:
                    pair_b.pop()

        return pair_a, pair_b

    def _tokenize_seq(self, pair_a, pair_b):

        pair_a, pair_b = self._truncate_seq_pair(pair_a, pair_b)

        # print(f"pair_a: {pair_a}")
        pair_a = ["[CLS]"] + pair_a + ["[SEP]"]
        seg_ids_a = [0] * len(pair_a)

        if pair_b is not None:
            pair_b = pair_b + ["[SEP]"]
            seg_ids_b = [1] * len(pair_b)

            pair_ab = pair_a + pair_b
            seg_ids = seg_ids_a + seg_ids_b
            input_mask = [1] * (len(pair_a) + len(pair_b))
        else:
            pair_ab = pair_a
            seg_ids = seg_ids_a
            input_mask = [1] * len(pair_a)

        pair_ab_token_id = [self.tokenizer.convert_tokens_to_ids(token) for token in pair_ab]

        # Pad the rest
        # We only pad the tokens that have been converted to ids
        # corresponding to the BERT vocabulary book.
        while len(pair_ab_token_id) < self.max_token_len:
            pair_ab_token_id.append(0)
            seg_ids.append(0)
            input_mask.append(0)

        return pair_ab_token_id, seg_ids, input_mask

    def _build_tokenised_dataset_seq(self, seq_a, seq_b):
        seq_a_tokenised = [self.tokenizer.tokenize(s) for s in seq_a]

        # Use naming convention consistent with example code in
        # HuggingFace repo. `seg_ids` corresponds to `token_type_ids`
        # and thus the name change.
        input_ids = []
        token_type_ids = []
        attention_masks = []

        if seq_b is not None:
            seq_b_tokenised = [self.tokenizer.tokenize(s) for s in seq_b]

            for pair_a, pair_b in tqdm(zip(seq_a_tokenised, seq_b_tokenised), desc="SEQ_A_and_B"):
                pair_ab_token_id, seg_ids, input_mask = self._tokenize_seq(pair_a, pair_b)

                input_ids.append(pair_ab_token_id)
                token_type_ids.append(seg_ids)
                attention_masks.append(input_mask)
        else:
          for pair_a in tqdm(seq_a_tokenised, desc="SEQ_A"):
                pair_a_token_id, seg_ids, input_mask = self._tokenize_seq(pair_a, None)

                input_ids.append(pair_a_token_id)
                token_type_ids.append(seg_ids)
                attention_masks.append(input_mask)

        return input_ids, token_type_ids, attention_masks


    #########    LOADING FUNCTIONS    #########

    """
    DataLoader, TensorData objects used for final output into a BERT model.
    """

    def _load_train_custom_dataset(self, batch_size):

        """

        We use this function to load doc_ids that are strings into batches such that at evaluation time we can track the documents being evaluated at the ID level.

        This is essential for information retrieval since we are reranking based on the topic ID.

        Note: Training data does not currently use attrib_seq and doc_ids

        """

        self.training_data_loader = DataLoader(
            DatasetWithStrID(
                self.X_train,
                self.X_mask,
                self.X_train_token_ids,
                self.y_train,
                None,
                None,
            ),
            shuffle=True,
            batch_size=batch_size,
            collate_fn=default_collate
        )

    def _load_test_custom_dataset(self, batch_size):

        # `attrib_seq` is used when we want to calculate
        # precision-recall accuracy for each topic
        if self.attrib_test is None:
            self.attrib_test = [0b0 for i in range(len(self.X_test))]

        if self.y_test is None:
            self.y_test = [0b0 for i in range(len(self.X_test))]

        self.testing_data_loader = DataLoader(
            DatasetWithStrID(
                self.X_test,
                self.X_mask_test,
                self.X_test_token_ids,
                self.y_test,
                self.attrib_test,
                self.doc_ids_test,
            ),
            shuffle=True,
            batch_size=batch_size,
            collate_fn=id_collate
        )

    def _load_train_dataset(self, batch_size):

        self.training_data_loader = DataLoader(
            TensorDataset(
                torch.tensor(self.X_train),
                torch.tensor(self.X_mask),
                torch.tensor(self.X_train_token_ids),
                torch.tensor(self.y_train),
            ),
            shuffle=True,
            batch_size=batch_size
        )

    def _load_test_dataset(self, batch_size):

        # # For general purposes, not TREC specific
        # if doc_ids_train is None:
        #     doc_ids_train = [0b0 for i in range(len(X_train))]

        if self.doc_ids_test is None:
            self.doc_ids_test = [0b0 for i in range(len(self.X_test))]

        # `attrib_seq` is used when we want to calculate
        # precision-recall accuracy for each topic
        if self.attrib_test is None:
            self.attrib_test = [0b0 for i in range(len(self.X_test))]

        if self.y_test is None:
            self.y_test = [0b0 for i in range(len(self.X_test))]

        self.testing_data_loader = DataLoader(
            TensorDataset(
                torch.tensor(self.X_test),
                torch.tensor(self.X_mask_test),
                torch.tensor(self.X_test_token_ids),
                torch.tensor(self.y_test),
                torch.tensor(self.attrib_test),
                torch.tensor(self.doc_ids_test)
            ),
            shuffle=False,
            batch_size=batch_size
        )


    #########    LOAD PRE-SPLIT DATA    #########

    def load_pre_split_train_dataset(
        self,
        seq_a_train, seq_b_train, labels_train,
        attrib_train, doc_ids_train,
        batch_size):

        """

        This function takes in <seq_a_train, seq_b_train> and <seq_a_test, seq_b_train>, in the sense that the user has already split up the dataset, and simply needs to tokenizer it for usage in BERT.

        Note: The doc_ids must be an array of numerics.

        Params
        ------
        see `create_train_and_test_dataset` function.

        """

        X_train, X_train_token_ids, X_mask = self._build_tokenised_dataset_seq(seq_a_train, seq_b_train)

        self.X_train = X_train
        self.X_train_token_ids = X_train_token_ids
        self.X_mask = X_mask
        self.y_train = labels_train

        # Should both be None
        self.attrib_train = attrib_train
        self.doc_ids_train = doc_ids_train

        self._load_train_dataset(batch_size)

        # if any([1 if type(x) is str else 0 for x in doc_ids_train]):
        #     self._load_train_custom_dataset(
        #         X_train, X_mask, X_train_token_ids, labels_train,
        #         batch_size
        #     )
        # else:
        #     self._load_train_dataset(
        #         X_train, X_mask, X_train_token_ids, labels_train,
        #         batch_size
        #     )

    def load_pre_split_test_dataset(
        self,
        seq_a_test, seq_b_test, labels_test,
        attrib_test, doc_ids_test,
        batch_size):

        """

        This function takes in <seq_a_train, seq_b_train> and <seq_a_test, seq_b_train>, in the sense that the user has already split up the dataset, and simply needs to tokenizer it for usage in BERT.

        Note: The doc_ids must be an array of numerics.

        Params
        ------
        see `create_train_and_test_dataset` function.

        """
        X_test, X_test_token_ids, X_mask_test = self._build_tokenised_dataset_seq(seq_a_test, seq_b_test)

        self.X_test = X_test
        self.X_test_token_ids = X_test_token_ids
        self.X_mask_test = X_mask_test
        self.y_test = labels_test
        self.attrib_test = attrib_test
        self.doc_ids_test = doc_ids_test

        if any([1 if type(x) is str else 0 for x in doc_ids_test]):
            self._load_test_custom_dataset(batch_size)
        else:
            self._load_test_dataset(batch_size)


    #########    LOAD UNSPLIT DATA    #########

    def _split_data_by_attribute(self, seq_a, seq_b, labels, ids, attribute_seq, test_size):
        # random.seed(self.random_state)
        uniq_attrib = set(attribute_seq)

        random.seed(self.random_state)
        attrib_for_test = random.sample(uniq_attrib, int(len(uniq_attrib)*test_size))

        print(f"attrib_for_test: {attrib_for_test}")

        seq_a_test = []
        labels_test = []
        attrib_test = []
        doc_ids_test = []

        seq_a_train = []
        labels_train = []
        attrib_train = []
        doc_ids_train = []

        if seq_b is None:
            seq_b_test = None
            seq_b_train = None
        else:
            seq_b_test = []
            seq_b_train = []

        idx = 0
        for attrib in attribute_seq:
            if attrib in attrib_for_test:
                seq_a_test.append(seq_a[idx])
                labels_test.append(labels[idx])
                attrib_test.append(attrib)
                doc_ids_test.append(ids[i])
                if seq_b is not None:
                    seq_b_test.append(seq_b[idx])
            else:
                seq_a_train.append(seq_a[idx])
                labels_train.append(labels[idx])
                attrib_train.append(attrib)
                doc_ids_train.append(ids[i])
                if seq_b is not None:
                    seq_b_train.append(seq_b[idx])
            idx += 1

        ret_arr = [
            seq_a_train, seq_b_train,
            labels_train, attrib_train,
            doc_ids_train,
            seq_a_test, seq_b_test,
            labels_test, attrib_test,
            doc_ids_test
        ]

        return ret_arr

    def create_train_and_test_dataset(
        self,
        seq_a, seq_b,
        labels, doc_ids,
        test_size, batch_size,
        split_by_attribute=None,
        attribute_seq=None,
        attribute_split_ratio=None):
        """

        This function takes in <seq_a, seq_b> pair and splits either:
            1) Randomly whilst maintaining balanced classes
            2) Splits the test and training set by an attribute e.g. topics

        This is required for TREC PM datasets, where we randomise topics so that during validation the test set topics are mutually exclusive from the training set topics

        Params
        ------
        seq_a: Array of text strings containing the first sentence pair
        seq_b: Likewise, for the second sentence pair
        labels: Labels of <seq_a, seq_b>
        doc_ids: ids that are required to tie the data back to a particular identifier. This MUST be an array of numerics.
        test_size: Hold out percentage
        batch_size: Number of training samples per backprop
        split_by_attribute: Choose specific attribute to split up training and test set.
        attribute_seq: If attribute is `split_by_attribute`, then an array of attributes corresponding to the data must be passed.

        """

        if split_by_attribute is not None:
            if (attribute_seq is None) and (attribute_split_ratio is not None):
                raise ValueError("Array of attributes must be passed if split_by_attribute is used.")

            seq_a_train, seq_b_train, y_train, attrib_train, doc_ids_train, seq_a_test, seq_b_test, y_test, attrib_test, doc_ids_test = self._split_data_by_attribute(
                    seq_a, seq_b, labels, doc_ids,
                    attribute_seq, test_size)

            X_train, X_train_token_ids, X_mask = self._build_tokenised_dataset_seq(seq_a_train, seq_b_train)
            X_test, X_test_token_ids, X_mask_test = self._build_tokenised_dataset_seq(seq_a_test, seq_b_test)

            self.X_train = X_train
            self.X_test = X_test

            self.X_train_token_ids = X_train_token_ids
            self.X_test_token_ids = X_test_token_ids

            self.X_mask = X_mask
            self.X_mask_test = X_mask_test

            self.y_train = y_train
            self.y_test = y_test

            self.attrib_train = attrib_train
            self.attrib_test = attrib_test

            self.doc_ids_train = doc_ids_train
            self.doc_ids_test = doc_ids_test

        else:
            # Convert to tokenised_text and save as self.input_ids
            input_ids, token_type_ids, attention_masks = self._build_tokenised_dataset_seq(seq_a, seq_b)

            # To ensure stratify goes correctly (actually for any of this to go
            # correctly) we need to set the random_state.
            print("Splitting the tokenised dataset into training and test set.")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                input_ids, labels,
                random_state=self.random_state,
                test_size=test_size,
                stratify=labels
            )
            self.X_mask, self.X_mask_test, _, _ = train_test_split(
                attention_masks, labels,
                random_state=self.random_state,
                test_size=test_size,
                stratify=labels
            )

            self.X_train_token_ids, self.X_test_token_ids, _, _ = train_test_split(
                token_type_ids, labels,
                random_state=self.random_state,
                test_size=test_size,
                stratify=labels
            )

            attrib_test = None
            if attribute_seq is not None:
                self.attrib_train, self.attrib_test, _, _ = train_test_split(
                    attribute_seq, labels,
                    random_state=self.random_state,
                    test_size=test_size,
                    stratify=labels
                )

            if doc_ids is not None:
                self.docs_ids_train, self.docs_ids_test, _, _ = train_test_split(
                    doc_ids, labels,
                    random_state=self.random_state,
                    test_size=test_size,
                    stratify=labels
                )

        self._load_train_dataset(batch_size)
        self._load_test_dataset(batch_size)

print("BertSeqClassDataLoader refreshed")
