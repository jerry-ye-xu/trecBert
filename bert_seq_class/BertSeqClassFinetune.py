import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle

from collections import defaultdict

# from matplotlib import pyplot as plt

import random
from tqdm import tqdm, trange

from keras.preprocessing.sequence import pad_sequences

import torch

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from torch.utils.data.dataloader import default_collate

from sklearn.model_selection import train_test_split

# from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

from transformers import load_tf_weights_in_bert
from transformers import (
    BertConfig, BertTokenizer, BertForSequenceClassification,
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
)
from transformers import AdamW, get_linear_schedule_with_warmup

from bert_seq_class.customDataset import DatasetWithStrID
from bert_seq_class.evalFunc import eval_pr_per_topics

class BertForSeqFinetune():
    def __init__(
        self,
        model_name, config,
        num_labels, model_desc,
        save_ckpt_path, save_flag,
        hf_model_class,
        hf_token_class,
        model_class_type,
        vocab_file=None,
        model_weights=None,
        from_tf=False):
        """

        Params
        ------

        model_name: model_name e.g. "bert-base-uncased" or path
        config: BertConfig object that is initialised with the same model_name
        num_labels: number of classes for finetuning categorical data
        model_desc: description of model used to name checkpoints that are saved with training
        save_ckpt_path: base path for saving checkpoints.
        save_flag: Whether to save, not used for model evaluation, i.e. using validation data.
        hf_model_class: HuggingFace model class
        hf_model_class: HuggingFace token class
        vocab_file: The vocabulary from a pretrained model
        model_weights: The Pytorch binary file from a pretrained model
        from_tf: Whether the model_name is a path pointing to a model pre-trained in Tensorflow. (Not tested)

        """

        # super(BertForSeqFinetune, self).__init__(config)

        ###   MODEL VARIABLES   ###

        # self.args_loaded = False
        self.device = None

        self.model_name = model_name
        self.config = config # initialised outside of class
        self.model = None
        self.tokenizer = None

        self.hf_model_class = hf_model_class
        self.hf_token_class = hf_token_class
        self.model_class_type = model_class_type

        ###   DATA VARIABLES   ###

        self.training_data_loader = None
        self.testing_data_loader = None
        self.validating_data_loader = None

        ###   TRAINING VARIABLES   ###

        self.num_labels = num_labels
        self.max_token_len = 128
        self.lr = 2e-5
        # self.TEST_SIZE = 0.2
        # self.EPOCHS = 3
        # self.BATCH_SIZE = 8

        # Save file names
        self.optimizer_pt = "optimizer.pt"
        self.scheduler_pt = "scheduler.pt"

        self.save_steps = 10
        self.warmup_steps = None
        self.total_steps = None
        self.gradient_accumulation_steps = 1
        self.logging_steps = 50
        self.max_grad_norm = 1.0
        self.loss_over_time = []
        self.random_state = 2018

        ###   EVALUATION VARIABLES   ###

        self.validation_accuracy = None
        # Precision-recall by topic
        # self.pr_dict = defaultdict(lambda: defaultdict(int))

        self.preds_arr = None
        self.labels_arr = None
        self.topics_eval_arr = None
        self.doc_id_eval_arr = None

        ###   SAVE PATH VARIABLES   ###

        self.cache_dir = None
        self.save_flag = save_flag
        if self.save_flag:
            self.output_dir = f"./{save_ckpt_path}/{model_desc}"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self._specify_model(
            self.model_name, self.config, self.num_labels,
            vocab_file=vocab_file, model_weights=model_weights
        )

    def load_dataloader_train_and_test(
        self,
        training_data_loader,
        testing_data_loader):
        self.training_data_loader = training_data_loader
        self.testing_data_loader = testing_data_loader

    def load_dataloader_validate(self, validating_data_loader):
        self.validating_data_loader = validating_data_loader

    def _specify_model(
        self,
        model_name, config, num_labels,
        vocab_file=None, model_weights=None,
        from_tf=False):
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

        if (model_weights is not None) and (from_tf == False):
            self.model = self.hf_model_class.from_pretrained(
                f"{model_name}",
                config=self.config
            )
        elif from_tf:
            self.model = self.hf_model_class.from_pretrained(
                f"{model_name}",
                from_tf=from_tf,
                config=self.config
            )
        else:
            self.model = self.hf_model_class.from_pretrained(
                f"{model_name}",
                config=self.config
            )
        if vocab_file is not None:
            self.tokenizer = self.hf_token_class.from_pretrained(
                f"{model_name}/{vocab_file}",
                do_lower_case=True
            )
        else:
            self.tokenizer = self.hf_token_class.from_pretrained(
                f"{model_name}",
                do_lower_case=True
            )

    def train(self, epochs, batch_size, use_gpu):
        """
        use_gpu: int
        """

        if self.model is None:
            raise ValueError("Model has not been specified!")

        if torch.cuda.is_available() and use_gpu:
            print("Using GPU")
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
            if use_gpu > 1:
                self.model = nn.DataParallel(self.model)
        else:
            print("CUDA not available. Using CPU")
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.total_steps = len(self.training_data_loader) // (self.gradient_accumulation_steps * epochs)
        self.warmup_steps = int(self.total_steps / 10)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        print(f"{self.output_dir}/optimizer.pt")
        if os.path.isfile(f"{self.output_dir}/optimizer.pt") and os.path.isfile(f"{self.output_dir}/scheduler.pt"):
            print("loading saved optimiser and scheduler")
            self.optimizer.load_state_dict(
                torch.load(f"{self.output_dir}/{self.optimizer_pt}")
            )
            self.scheduler.load_state_dict(
                torch.load(f"{self.output_dir}/{self.scheduler_pt}")
            )

        global_steps = 0
        tr_loss, tr_loss_prev = 0.0, 0.0
        nb_tr_examples = 0

        for epoch in trange(epochs, desc="EPOCHS"):
            epoch_iterator = tqdm(self.training_data_loader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                self.model.zero_grad()

                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[3]
                }
                inputs['token_type_ids'] = (batch[2] if self.model_class_type in ["bert", "xlnet", "albert"] else None)
            # Rewrite this code to check for model_type more easily.
            # if args.model_type != 'distilbert':
            #     inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None

                outputs = self.model(**inputs)
                loss = outputs[0]
                print(f"loss: {loss}")
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                tr_loss += loss.item()
                self.loss_over_time.append(tr_loss)
                nb_tr_examples += inputs["input_ids"].size(0)
                global_steps += 1

                # @TODO: Find suitable way to record this information
                if global_steps % self.logging_steps == 0:
                    avg_loss = (tr_loss - tr_loss_prev)/self.logging_steps
                    tr_loss_prev = tr_loss
                    print(f"Statistics over the last {self.logging_steps} steps:")
                    print(f"\t global_steps: {global_steps}")
                    print(f"\t average loss: {avg_loss}")
                    print(f"\t loss.item(): {loss.item()}")
                    print(f"\t tr_loss: {tr_loss}")
                    print(f"\t nb_tr_examples: {nb_tr_examples}")

            if self.save_flag:
                output_dir = os.path.join(self.output_dir, 'checkpoint-{}'.format(global_steps))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if (epoch % 1) == 0:
                    tmp_eval = self.evaluate(
                        {"precision_recall_by_topic": eval_pr_per_topics},
                        use_ids=True,
                        validate=False,
                        use_gpu=True
                    )

                    # Convert the defaultdict to dict, for JSON
                    # print(tmp_eval[4])
                    for key in tmp_eval[4].keys():
                        tmp_eval[4][key] = dict(tmp_eval[4][key])
                    # print(tmp_eval[4])
                    pr_dict_tmp = dict(tmp_eval[4])
                    # print(pr_dict_tmp)

                    output_dict = {
                        "y_truth": tmp_eval[0].tolist(),
                        "y_pred": tmp_eval[1].tolist(),
                        "topics_arr": tmp_eval[2].tolist(),
                        "doc_ids_arr": tmp_eval[3].tolist(),
                        "pr_dict": pr_dict_tmp,
                    }
                    # print(output_dict["pr_dict"])
                    # print(type(output_dict["pr_dict"]))

                    pd.to_pickle(output_dict, f"{output_dir}/ckpt_eval.pickle")
                    # with open(f"{output_dir}/ckpt_eval.json", "w") as f:
                    #     pickle.dump(pr_dict_tmp, f)

                    self.save_model(output_dir)

            # Take care of distributed/parallel training
            # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            # model_to_save.save_pretrained(output_dir)

            # @TODO: Do we want to implement a way to save the arguments?
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        return global_steps, tr_loss/global_steps

    def evaluate(self, eval_metrics_dict, use_ids, validate, use_gpu):
        """

        Format of the batch is different depending on use_ids:

        IF use_ids IS True:
        (
          [
            tensor([
              [a1], ..., [an]
            ])
            tensor([
              [b1], ..., [bn]
            ])
            ...
            tensor([
              [j1], ..., [jn]
            ])
          ],
          [id1, ..., idn]
        )
        ELSE:
          (
            tensor([
              [a1], ..., [an]
            ])
            ...
            tensor([
              [j1], ..., [jn]
            ])
          )

        Params
        ------
        eval_metric_dict: {
                "accuracy": num_correctly_classified,
                "precision_recall_by_topic": eval_pr_per_topics,
                "roc_curve": calc_roc
            }
        use_ids: If true, we use unique IDs are used to track the individual data points
        validate: If true, then this is a validation set with no labels.

        """
        eval_loss = 0.0
        nb_eval_steps = 0

        acc_test_loss = 0.0
        self.pr_dict = defaultdict(lambda: defaultdict(int))

        if torch.cuda.is_available() and use_gpu:
            print("Using GPU")
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
            if use_gpu > 1:
                self.model = nn.DataParallel(self.model)
        else:
            print("CUDA not available. Using CPU")
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.model.eval()

        if validate:
            test_data = self.validating_data_loader
        else:
            test_data = self.testing_data_loader

        for batch in tqdm(test_data, desc="EVALUATING"):
            with torch.no_grad():
                # print(batch)
                if use_ids:
                    doc_ids_batch = batch[1]
                    batch = tuple(t.to(self.device) for t in batch[0])

                    topics_batch = batch[4].detach().cpu().numpy()
                else:
                    batch = tuple(t.to(self.device) for t in batch)
                    topics_batch = batch[4].detach().cpu().numpy()
                    doc_ids_batch = batch[5].detach().cpu().numpy()
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[3]
                }
                inputs['token_type_ids'] = (batch[2] if self.model_class_type in ["bert", "xlnet", "albert"] else None)
            # if args.model_type != 'distilbert':
            #     inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                # What does this do?

                if validate:
                    inputs.pop("labels")
                    outputs = self.model(**inputs)
                    logits = outputs[0]
                    # print(logits)
                else:
                    outputs = self.model(**inputs)
                    tmp_test_loss, logits = outputs[:2]

                    # What does this do?
                    eval_loss += tmp_test_loss.mean().item()
            nb_eval_steps += 1

            ################     UPDATE TOTAL LOSS     ################

            logits_batch = logits.detach().cpu().numpy()
            if not validate:
                labels_batch = inputs["labels"].cpu().numpy()

            if "accuracy" in eval_metrics_dict.keys():
                batch_test_loss = eval_metrics_dict["accuracy"](
                    logits_batch,
                    labels_batch
                )
                acc_test_loss += batch_test_loss

            if "precision_recall_by_topic" in eval_metrics_dict.keys():
                eval_metrics_dict["precision_recall_by_topic"](
                    logits_batch,
                    labels_batch,
                    topics_batch,
                    self.pr_dict
                )

            # We're going to save this and return it later
            if self.preds_arr is None:
                self.preds_arr = logits_batch

                if not validate:
                    self.labels_arr = labels_batch

                self.topics_eval_arr = topics_batch
                self.doc_id_eval_arr = doc_ids_batch
            else:
                self.preds_arr = np.append(
                    self.preds_arr,
                    logits_batch, axis=0
                )

                if not validate:
                  # print(inputs["labels"])
                  self.labels_arr = np.append(
                      self.labels_arr,
                      labels_batch, axis=0
                  )
                self.topics_eval_arr = np.append(
                    self.topics_eval_arr,
                    topics_batch, axis=0
                )
                self.doc_id_eval_arr = np.append(
                    self.doc_id_eval_arr, doc_ids_batch,
                    axis=0
                )

        ################     DISPLAY RESULTS     ################

        # previous metric_function function accuracy percentage for each batch
        # self.validation_accuracy = acc_test_loss/nb_eval_steps

        if not validate:
            eval_loss = eval_loss/nb_eval_steps
            print(f"eval_loss: {eval_loss}")

        if validate:
            num_test_points = len(self.validating_data_loader.dataset)
        else:
            num_test_points = len(self.testing_data_loader.dataset)

        print(f"acc_test_loss: {acc_test_loss}")
        print(f"num_test_points: {num_test_points}")

        if "accuracy" in eval_metrics_dict.keys():
            self.validation_accuracy = acc_test_loss/num_test_points
            print("Validation Accuracy: {}".format(self.validation_accuracy))

        if "precision_recall_by_topic" in eval_metrics_dict.keys():
            for topic in self.pr_dict.keys():

                if (self.pr_dict[topic]["false_positive"] + self.pr_dict[topic]["true_positive"]) == 0:
                    print(f"FP + TP = 0")
                    precision = 0
                else:
                    precision = self.pr_dict[topic]["true_positive"]/(self.pr_dict[topic]["false_positive"] + self.pr_dict[topic]["true_positive"])

                if (self.pr_dict[topic]["false_negative"] + self.pr_dict[topic]["true_positive"]) == 0:
                    print(f"FN + TP = 0")
                    recall = 0
                else:
                    recall = self.pr_dict[topic]["true_positive"]/(self.pr_dict[topic]["false_negative"] + self.pr_dict[topic]["true_positive"])

                self.pr_dict[topic]["precision"] = precision
                self.pr_dict[topic]["recall"] = recall

        if "roc_curve" in eval_metrics_dict.keys():
            eval_metrics_dict["roc_curve"](
                self.preds_arr,
                self.labels_arr,
                num_classes=self.num_labels
            )

        # self.labels_arr is None if we are evaluating with validation data.

        return self.labels_arr, self.preds_arr, self.topics_eval_arr, self.doc_id_eval_arr, self.pr_dict, self.validation_accuracy

    def save_model(self, output_dir):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(
            self.optimizer.state_dict(),
            f"{output_dir}/{self.optimizer_pt}"
        )
        torch.save(
            self.scheduler.state_dict(),
            f"{output_dir}/{self.scheduler_pt}"
        )

        # @TODO: Implement dict of args
        #  torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        self.model = self.hf_model_class.from_pretrained(output_dir)
        self.tokenizer = self.hf_token_class.from_pretrained(output_dir)
        self.model.to(self.device)

print("BertSeqClassFinetune refreshed")
