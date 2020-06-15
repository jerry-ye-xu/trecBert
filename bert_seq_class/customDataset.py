import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

class DatasetWithStrID(Dataset):
    def __init__(
        self,
        X, X_mask, X_token_ids, y,
        attrib_seq, doc_ids):

        """

        We utilise a custom DataSet class to enough that we can pass IDs
        in and out of the batch at evaluation time.

        Furthermore, we make sure to pass data samples that have already been tokenized.

        """

        self.X = X
        self.X_mask = X_mask
        self.X_token_ids = X_token_ids
        self.y = y
        # This should be none if loading training data
        self.attrib_seq = attrib_seq
        self.doc_ids = doc_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """ Returns one sample of data"""

        if (self.attrib_seq is None) and (self.doc_ids is None):
            ret_arr = [
                torch.tensor(self.X[index]),
                torch.tensor(self.X_mask[index]),
                torch.tensor(self.X_token_ids[index]),
                torch.tensor(self.y[index]),
            ]
        else:
            ret_arr = [
                torch.tensor(self.X[index]),
                torch.tensor(self.X_mask[index]),
                torch.tensor(self.X_token_ids[index]),
                torch.tensor(self.y[index]),
                torch.tensor(self.attrib_seq[index]),
                self.doc_ids[index]
            ]

        return ret_arr

def id_collate(batch):
    new_batch = []
    doc_ids = []

    for b in batch:
        new_batch.append(b[:-1])
        doc_ids.append(b[-1])
    return default_collate(new_batch), doc_ids
