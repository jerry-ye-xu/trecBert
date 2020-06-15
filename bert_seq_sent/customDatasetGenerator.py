import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from local_parser.ingest_preranking import build_dataset_for_bert

from bert_seq_sent.BertSeqSentProcessData import process_validation_data_raw, split_into_sent_bert_input
from bert_seq_sent.BertSeqSentGlobalVar import global_var

class BioASQDatasetGenerator(Dataset):
    def __init__(
        self,
        model_name,
        config,
        vocab_file,
        num_labels,
        max_token_len,
        bioasq_path,
        batch_size,
        tokenizer):

        """
        Generator specifically for BioASQ json file.
        """

        ###   MODEL VARIABLES   ###

        self.model_name = model_name
        self.config = config # initialised outside of class
        self.model = None
        self.tokenizer = tokenizer

        ###   DATA VARIABLES   ###

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_token_len = max_token_len

        self.bioasq_path = bioasq_path
        # with open(self.bioasq_path, "r") as f:
        #     self.data = json.load(f)
        self.data = pd.read_pickle(self.bioasq_path)
        # print(f"self.data.shape[0]: {self.data.shape[0]}")
        if num_labels == 2:
            self.data["label"] = self.data["label"].apply(lambda x: 0 if x < 1 else 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """ Returns one sample of data"""

        # print(f"using __getitem__ with index: {index}")
        data_pt_qst = self.data["question"][index]
        data_pt_ans = self.data["answer"][index]

        # print(f"self.data[\"answer\"][index] is {data_pt_ans}")
        # print(f"self.data[\"question\"][index] is {data_pt_qst}")

        try:
            label = int(self.data["label"][index])
        except KeyError:
            label = -1
        qa_id = self.data["qa_id"][index]

        seq_tokens, token_type_id, attention_mask = self._build_tokenised_seq(data_pt_qst, data_pt_ans)

        ret_arr = [
            torch.tensor(seq_tokens),
            torch.tensor(token_type_id),
            torch.tensor(attention_mask),
            torch.tensor(label),
            qa_id
        ]

        return ret_arr

    ################################
    ####    HELPER FUNCTIONS    ####
    ################################

    def _truncate_seq_pair(self, seq_a, seq_b=None):
        """Truncates a sequence pair to the maximum length."""

        if seq_b is None:
            if len(seq_a) > (self.max_token_len - 2):
                return seq_a[:(self.max_token_len - 2)], seq_b
        else:
            while True:
                total_length = len(seq_a) + len(seq_b)
                if total_length <= (self.max_token_len - 3):
                    break
                if len(seq_a) > len(seq_b):
                    seq_a.pop()
                else:
                    seq_b.pop()

        return seq_a, seq_b

    def _tokenize_seq(self, seq_a, seq_b):

        seq_a, seq_b = self._truncate_seq_pair(seq_a, seq_b)

        # print(f"seq_a: {seq_a}")
        seq_a = ["[CLS]"] + seq_a + ["[SEP]"]
        seg_ids_a = [0] * len(seq_a)

        if seq_b is not None:
            seq_b = seq_b + ["[SEP]"]
            seg_ids_b = [1] * len(seq_b)

            seq_ab = seq_a + seq_b
            seg_ids = seg_ids_a + seg_ids_b
            input_mask = [1] * (len(seq_a) + len(seq_b))
        else:
            seq_ab = seq_a
            seg_ids = seg_ids_a
            input_mask = [1] * len(seq_a)

        seq_ab_token_id = [self.tokenizer.convert_tokens_to_ids(token) for token in seq_ab]

        # Pad the rest
        # We only pad the tokens that have been converted to ids
        # corresponding to the BERT vocabulary book.
        while len(seq_ab_token_id) < self.max_token_len:
            seq_ab_token_id.append(0)
            seg_ids.append(0)
            input_mask.append(0)

        return seq_ab_token_id, seg_ids, input_mask

    def _build_tokenised_seq(self, seq_a, seq_b=None):
        seq_a_id = self.tokenizer.tokenize(seq_a)
        if seq_b is not None:
            seq_b_id = self.tokenizer.tokenize(seq_b)

        # For self._tokenizer_seq: input_id, token_type_id, attention_mask
        return self._tokenize_seq(seq_a_id, seq_b_id)

class clinicalDatasetGenerator(Dataset):
    def __init__(
        self,
        model_name,
        config,
        vocab_file,
        num_labels,
        max_token_len,
        ct_path,
        use_qe,
        n_chars_trim,
        global_var,
        batch_size,
        # validate,
        tokenizer):

        """
        Generator specifically for BioASQ json file.
        """

        ###   MODEL VARIABLES   ###

        self.model_name = model_name
        self.config = config # initialised outside of class
        self.model = None
        self.tokenizer = tokenizer

        ###   DATA VARIABLES   ###

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.use_qe = use_qe
        self.n_chars_trim = n_chars_trim
        self.global_var = global_var

        self.ct_path = ct_path
        # with open(self.bioasq_path, "r") as f:
        #     self.data = json.load(f)

        self.data = pd.read_pickle(self.ct_path)

        self.seq_a, self.seq_b, self.labels, self.attribs, self.doc_ids = split_into_sent_bert_input(
            self.data,
            global_var["seq_a_expansion"] if self.use_qe else global_var["seq_a_baseline"],
            global_var["seq_b_baseline"],
            validate=True,
            use_qe=self.use_qe,
            global_var=global_var
        )

        assert len(self.seq_a) == len(self.attribs)
        assert len(self.attribs) == len(self.doc_ids)

        # print(f"set(self.attribs): {set(self.attribs)}")
        # print(self.data["topic"].value_counts())
        # print(f"len(self.attribs): {len(self.attribs)}")
        # print(f"len(self.seq_a): {len(self.seq_a)}")
        # print(f"len(self.doc_ids): {len(self.doc_ids)}")
        with open("./debug_data/self_attribs.pickle", "wb") as f:
            pickle.dump(self.attribs, f)
        # print(self.attribs)

    def __len__(self):
        return len(self.seq_a)

    def __getitem__(self, index):
        """ Returns one sample of data"""
        if self.labels is not None:
            label = int(self.labels[index])
        else:
            label = -1
        doc_id = self.doc_ids[index]
        topic_id = int(self.attribs[index])
        # print(f"index: {index}")
        # print(f"self.attribs[index]: {self.attribs[index]}")
        # print(f"getting topic id: {topic_id}")

        if self.seq_b is not None:
            seq_tokens, token_type_id, attention_mask = self._build_tokenised_seq(self.seq_a[index], self.seq_b[index])
        else:
            seq_tokens, token_type_id, attention_mask = self._build_tokenised_seq(self.seq_a[index], None)

        ret_arr = [
            torch.tensor(seq_tokens),
            torch.tensor(token_type_id),
            torch.tensor(attention_mask),
            torch.tensor(label),
            torch.tensor(topic_id),
            doc_id
        ]

        return ret_arr

    ################################
    ####    HELPER FUNCTIONS    ####
    ################################

    def _truncate_seq_pair(self, seq_a, seq_b=None):
        """Truncates a sequence pair to the maximum length."""

        if seq_b is None:
            if len(seq_a) > (self.max_token_len - 2):
                return seq_a[:(self.max_token_len - 2)], seq_b
        else:
            while True:
                total_length = len(seq_a) + len(seq_b)
                if total_length <= (self.max_token_len - 3):
                    break
                if len(seq_a) > len(seq_b):
                    seq_a.pop()
                else:
                    seq_b.pop()

        return seq_a, seq_b

    def _tokenize_seq(self, seq_a, seq_b):

        seq_a, seq_b = self._truncate_seq_pair(seq_a, seq_b)

        # print(f"seq_a: {seq_a}")
        seq_a = ["[CLS]"] + seq_a + ["[SEP]"]
        seg_ids_a = [0] * len(seq_a)

        if seq_b is not None:
            seq_b = seq_b + ["[SEP]"]
            seg_ids_b = [1] * len(seq_b)

            seq_ab = seq_a + seq_b
            seg_ids = seg_ids_a + seg_ids_b
            input_mask = [1] * (len(seq_a) + len(seq_b))
        else:
            seq_ab = seq_a
            seg_ids = seg_ids_a
            input_mask = [1] * len(seq_a)

        seq_ab_token_id = [self.tokenizer.convert_tokens_to_ids(token) for token in seq_ab]

        # Pad the rest
        # We only pad the tokens that have been converted to ids
        # corresponding to the BERT vocabulary book.
        while len(seq_ab_token_id) < self.max_token_len:
            seq_ab_token_id.append(0)
            seg_ids.append(0)
            input_mask.append(0)

        return seq_ab_token_id, seg_ids, input_mask

    def _build_tokenised_seq(self, seq_a, seq_b=None):
        seq_a_id = self.tokenizer.tokenize(seq_a)

        seq_b_id = None
        if seq_b is not None:
            seq_b_id = self.tokenizer.tokenize(seq_b)

        # For self._tokenizer_seq: input_id, token_type_id, attention_mask
        return self._tokenize_seq(seq_a_id, seq_b_id)

def id_collate(batch):
    new_batch = []
    doc_ids = []

    for b in batch:
        new_batch.append(b[:-1])
        doc_ids.append(b[-1])
    return default_collate(new_batch), doc_ids

print("customDataSetGenerator refreshed")