class BertForSeqFinetune():
    def __init__(self, 
                 model_name, config, num_labels,
                 hf_model_class=BertForSequenceClassification,
                 hf_token_class=BertTokenizer,
                 vocab_file=None,
                 model_weights=None,
                 from_tf=False
                 ):
        # super(BertForSeqFinetune, self).__init__(config)

        # self.args_loaded = False
        self.device = None

        self.model_name = model_name
        self.config = config # initialised outside of class
        self.model = None
        self.tokenizer = None
        self.hf_model_class = hf_model_class
        self.hf_token_class = hf_token_class

        # Only used if you need to split data into train and test
        # by a specific attribute to avoid the model data snooping
        self.attribute_split_ratio = 0.3

        # If split_by_attribute is used then each variable will
        # hold the [train, test] split data.

        # Normally it would just be an array. Yes, this might be bad design since
        # the structure of variable changes.

        self.input_ids = None
        self.token_type_ids = None
        self.attention_masks = None
        self.labels = None

        self.training_data_loader = None
        self.test_data_loader = None
        self.validation_accuracy = None

        # Precision-recall by topic
        self.pr_dict = defaultdict(lambda: defaultdict(int))

        # For ROC curve
        self.preds_arr = None
        self.labels_arr = None

        self.NUM_LABELS = num_labels
        self.MAX_TOKEN_LEN = 128
        self.LR = 2e-5
        # self.TEST_SIZE = 0.2
        # self.EPOCHS = 3
        # self.BATCH_SIZE = 8
        self.SAVE_STEPS = 10
        self.WARMUP_STEPS = 100
        self.TOTAL_STEPS = 1000
        self.LOGGING_STEPS = 50
        self.MAX_GRAD_NORM = 1.0
        self.LOSS_OVER_TIME = []
        self.RANDOM_STATE = 2018

        self.cache_dir = None
        self.output_dir = "./save_files"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._specify_model(
            self.model_name, self.config, self.NUM_LABELS,
            vocab_file=vocab_file, model_weights=model_weights
        )

    def _specify_model(self, 
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
                f"{model_name}/{model_weights}",
                config=self.config
            )
        elif from_tf:
            self.model = self.hf_model_class.from_pretrained(
                model_name,
                from_tf=from_tf,
                config=self.config
            )             
        else:
            self.model = self.hf_model_class.from_pretrained(
                model_name,
                config=self.config
            )            
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

    #######################################################

    #########    AREA OF UNDECIDED CODE DESIGN    #########

    #######################################################

    def _truncate_seq_pair(self, pair_a, pair_b=None):
        """Truncates a sequence pair to the maximum length."""

        if pair_b is None:
            if len(pair_a) > (self.MAX_TOKEN_LEN - 2):
                return pair_a[:(self.MAX_TOKEN_LEN - 2)], pair_b
        else:
            while True:
                total_length = len(pair_a) + len(pair_b)
                if total_length <= (self.MAX_TOKEN_LEN - 3):
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
        while len(pair_ab_token_id) < self.MAX_TOKEN_LEN:
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

    def _split_data_by_attribute(self,
                                 seq_a, seq_b, labels,
                                 attribute_seq,
                                 test_size):
        # random.seed(self.RANDOM_STATE)
        uniq_attrib = set(attribute_seq)

        random.seed(self.RANDOM_STATE)
        attrib_for_test = random.sample(uniq_attrib, int(len(uniq_attrib)*test_size))

        print(f"attrib_for_test: {attrib_for_test}")

        seq_a_test = []
        labels_test = []
        attrib_test = []

        seq_a_train = []
        labels_train = []
        attrib_train = []

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
                if seq_b is not None:
                    seq_b_test.append(seq_b[idx])
            else:
                seq_a_train.append(seq_a[idx])
                labels_train.append(labels[idx])
                attrib_train.append(attrib)
                if seq_b is not None:
                    seq_b_train.append(seq_b[idx])
            idx += 1

        ret_arr = [
            seq_a_train, seq_b_train,
            labels_train, attrib_train,
            seq_a_test, seq_b_test,
            labels_test, attrib_test
        ]

        return ret_arr

    #######################################################

    def create_train_and_test_dataset(self,
                                      seq_a, seq_b, labels,
                                      test_size, batch_size,
                                      split_by_attribute=None,
                                      attribute_seq=None,
                                      attribute_split_ratio=None
                                      ):
        """
        Params
        ------

        seq_a: Array of text strings containing the first sentence pair
        seq_b: Likewise, for the second sentence pair
        labels: Labels of <seq_a, seq_b>
        test_size: Hold out percentage
        batch_size: Number of training samples per backprop
        split_by_attribute: Choose specific attribute to split up training and test set. This is required for TREC PM datasets, where we randomise topics so that during validation the test set contains ONLY topics not
        seen during training time.
        attribute_seq: If attribute is `split_by_attribute`, then an array of attributes corresponding to the data must be passed.

        """

        if split_by_attribute is not None:
            if (attribute_seq is None) and (attribute_split_ratio is not None):
                raise ValueError("Array of attributes must be passed if split_by_attribute is used.")

            seq_a_train, seq_b_train, y_train, attrib_train, seq_a_test, seq_b_test, y_test, attrib_test = self._split_data_by_attribute(seq_a, seq_b, labels, attribute_seq, test_size)

            X_train, X_train_token_ids, X_mask = self._build_tokenised_dataset_seq(seq_a_train, seq_b_train)
            X_test, X_test_token_ids, X_mask_test = self._build_tokenised_dataset_seq(seq_a_test, seq_b_test)

            self.input_ids = [X_train, X_test]
            self.token_type_ids = [X_train_token_ids, X_test_token_ids]
            self.attention_masks = [X_mask, X_mask_test]
            self.labels = [y_train, y_test]
            self.attrib_seq = [attrib_train, attrib_test]

        else:
            # Convert to tokenised_text and save as self.input_ids
            self.input_ids, self.token_type_ids, self.attention_masks = self._build_tokenised_dataset_seq(seq_a, seq_b)
            self.labels = labels
            self.attrib_seq = attribute_seq

            # To ensure stratify goes correctly (actually for any of this to go
            # correctly) we need to set the random_state.
            X_train, X_test, y_train, y_test = train_test_split(
                self.input_ids, self.labels,
                random_state=self.RANDOM_STATE,
                test_size=test_size,
                stratify=self.labels
            )
            X_mask, X_mask_test, _, _ = train_test_split(
                self.attention_masks, self.labels,
                random_state=self.RANDOM_STATE,
                test_size=test_size,
                stratify=self.labels
            )

            X_train_token_ids, X_test_token_ids, _, _ = train_test_split(
                self.token_type_ids, self.labels,
                random_state=self.RANDOM_STATE,
                test_size=test_size,
                stratify=self.labels
            )

            if self.attrib_seq is not None:
                attrib_train, attrib_test, _, _ = train_test_split(
                    self.attrib_seq, self.labels,
                    random_state=self.RANDOM_STATE,
                    test_size=test_size,
                    stratify=self.labels
                )

        self.training_data_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train),
                torch.tensor(X_mask),
                torch.tensor(X_train_token_ids),
                torch.tensor(y_train)
            ),
            shuffle=True,
            batch_size=batch_size
        )

        # `attrib_seq` is used when we want to calculate
        # precision-recall accuracy for each topic
        if self.attrib_seq is None:
            attrib_test = [0b0 for i in range(len(X_test))]

        self.test_data_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test),
                torch.tensor(X_mask_test),
                torch.tensor(X_test_token_ids),
                torch.tensor(y_test),
                torch.tensor(attrib_test)
            ),
            shuffle=False,
            batch_size=batch_size
        )

    def train(self, epochs, batch_size, use_gpu):
        if self.model is None:
            raise ValueError("Model has not been specified!")

        if torch.cuda.is_available() and use_gpu:
            print("Using GPU")
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            print("CUDA not available. Using CPU")
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
            num_warmup_steps=self.WARMUP_STEPS,
            num_training_steps=self.TOTAL_STEPS
        )

        global_steps = 0
        tr_loss, tr_loss_prev = 0.0, 0.0
        nb_tr_examples = 0
        self.model.zero_grad()

        for _ in trange(epochs, desc="EPOCHS"):
            epoch_iterator = tqdm(self.training_data_loader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],      # BERT and XLM does use this, but not strictly necessary.
                    'labels':         batch[3]
                }
            # Rewrite this code to check for model_type more easily.
            # if args.model_type != 'distilbert':
            #     inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None

                self.model.zero_grad()

                outputs = self.model(**inputs)
                loss = outputs[0]
                print(f"loss: {loss}")
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.MAX_GRAD_NORM
                )
                self.optimizer.step()
                self.scheduler.step()

                tr_loss += loss.item()
                self.LOSS_OVER_TIME.append(tr_loss)
                nb_tr_examples += inputs["input_ids"].size(0)
                global_steps += 1

                # @TODO: Find suitable way to record this information
                if global_steps % self.LOGGING_STEPS == 0:
                    avg_loss = (tr_loss - tr_loss_prev)/self.LOGGING_STEPS
                    tr_loss_prev = tr_loss
                    print(f"Statistics over the last {self.LOGGING_STEPS} steps:")
                    print(f"\t global_steps: {global_steps}")
                    print(f"\t average loss: {avg_loss}")
                    print(f"\t loss.item(): {loss.item()}")
                    print(f"\t tr_loss: {tr_loss}")
                    print(f"\t nb_tr_examples: {nb_tr_examples}")

            output_dir = os.path.join(self.output_dir, 'checkpoint-{}'.format(global_steps))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.save_model()
            # Take care of distributed/parallel training
            # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            # model_to_save.save_pretrained(output_dir)

            # @TODO: Do we want to implement a way to save the arguments?
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        return global_steps, tr_loss/global_steps

    def evaluate(self, metric_function, metric):
        """

        Params
        ------
        metric_function: function corresponding to the metric you'll use.
        metric: One of ["accuracy", "roc_curve", "precision_recall_by_topic"]

        """
        test_loss = 0.0 # for accuracy
        eval_loss = 0.0
        nb_eval_steps = 0

        self.model.to(self.device)

        self.model.eval()

        for batch in tqdm(self.test_data_loader, desc="EVALUATING"):
            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3],
                    'topics':         batch[4]
                }
            # if args.model_type != 'distilbert':
            #     inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                # What does this do?
                topics = inputs["topics"]
                inputs.pop("topics")

                outputs = self.model(**inputs)
                tmp_test_loss, logits = outputs[:2]
                eval_loss += tmp_test_loss.mean().item()

                # What does this do?
                eval_loss += tmp_test_loss.mean().item()
            nb_eval_steps += 1 

            ################     UPDATE TOTAL LOSS     ################

            if metric == "accuracy":
                batch_test_loss = metric_function(
                    logits.detach().cpu().numpy(),
                    inputs["labels"].cpu().numpy()
                )
                test_loss += batch_test_loss

            if metric == "precision_recall_by_topic":
                metric_function(
                    logits.detach().cpu().numpy(),
                    inputs["labels"].cpu().numpy(),
                    topics.detach().cpu().numpy(),
                    self.pr_dict
                )

            # We're going to save this and return it later
            if self.preds_arr is None:
                self.preds_arr = logits.detach().cpu().numpy()
                self.labels_arr = inputs['labels'].detach().cpu().numpy()
            else:
                self.preds_arr = np.append(
                    self.preds_arr,
                    logits.detach().cpu().numpy(),
                    axis=0
                )
                self.labels_arr = np.append(
                    self.labels_arr,
                    inputs['labels'].detach().cpu().numpy(),
                    axis=0
                )

        ################     DISPLAY RESULTS     ################

        # previous metric_function function accuracy percentage for each batch
        # self.validation_accuracy = test_loss/nb_eval_steps

        eval_loss = eval_loss/nb_eval_steps

        num_test_points = len(self.test_data_loader.dataset)

        print(f"eval_loss: {eval_loss}")
        print(f"test_loss: {test_loss}")
        print(f"num_test_points: {num_test_points}")

        if metric == "accuracy":
            self.validation_accuracy = test_loss/num_test_points
            print("Validation Accuracy: {}".format(self.validation_accuracy))

        if metric == "precision_recall_by_topic":
            print(self.pr_dict)

            for topic in self.pr_dict.keys():

                if (self.pr_dict[topic]["false_positive"] + self.pr_dict[topic]["true_positive"]) == 0:
                    print(f"FP + TP = 0")
                    precision = 0
                else:
                    precision = self.pr_dict[topic]["true_positive"]/(self.pr_dict[topic]["false_positive"] + 
                                                                      self.pr_dict[topic]["true_positive"])
                
                if (self.pr_dict[topic]["false_negative"] + self.pr_dict[topic]["true_positive"]) == 0:
                    print(f"FN + TP = 0")
                    recall = 0
                else:
                    recall = self.pr_dict[topic]["true_positive"]/(self.pr_dict[topic]["false_negative"] + 
                                                                   self.pr_dict[topic]["true_positive"])

                self.pr_dict[topic]["precision"] = precision
                self.pr_dict[topic]["recall"] = recall

            return self.pr_dict

        if metric == "roc_curve":
            metric_function(
                self.preds_arr, 
                self.labels_arr,
                num_classes=self.NUM_LABELS
            )
        
        return self.labels_arr, self.preds_arr


    def save_model(self):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # @TODO: Implement dict of args
        #  torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        self.model = self.hf_model_class.from_pretrained(self.output_dir)
        self.tokenizer = self.hf_token_class.from_pretrained(self.output_dir)
        self.model.to(self.device)

print("function refreshed")