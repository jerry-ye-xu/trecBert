NAME    := TREC_PM
SHELL   := /bin/bash
VERSION := $(shell cat VERSION)

PARSER := ./local_parser
QE     := ./query_expansion
BERT   := ./bert_seq_class
LTR    := ./ltr_models
A2A    := ./A2A4UMA

LM     := ./bert_finetune_lm

DATA          := ./data
INDEX         := ./A2A4UMA/indices
LABELLED_DATA := $(DATA)/pm_labels_2017 $(DATA)/pm_labels_2018 $(DATA)/pm_labels_2019

TRIAL_TOPIC       := $(DATA)/trials_topics_combined_all_years.pickle
TRIAL_TOPIC_QE    := $(DATA)/trials_topics_combined_all_years_qe.pickle
TRIAL_TOPIC_QE_PAPER := $(DATA)/trials_topics_combined_all_years_qe_paper.pickle

BIOASQ_TRAIN := $(DATA)/bioasq_df_train.pickle
BIOASQ_TEST := $(DATA)/bioasq_df_test.pickle

FINETUNED_MODELS  := $(DATA)/finetuned_models
PRETRAINED_MODELS := $(DATA)/pretrained_models

# MODEL_NAME ?= roberta-base
# MODEL_CLASS_TYPE ?= roberta
# MODEL_DESC ?= roberta
# MODEL_NAME_EVAL ?= roberta

MODEL_NAME ?= bert-base-uncased
MODEL_CLASS_TYPE ?= bert
MODEL_DESC ?= basic_bert
MODEL_NAME_EVAL ?= basic_bert

MODEL_DESC_SENT ?= basic_bert_sent
MODEL_NAME_EVAL_SENT ?= basic_bert_sent
DF_BERT_PATH ?= $(DATA)/df_bert_tmp.pickle

BASE_RANKER_TYPE ?= BM25
LTR_MODEL ?= bert

SAVE_TREC_NAME ?= ct_data

CKPT_NUM_FINETUNE ?= -1
# CKPT_NUM ?= 599
CKPT_NUM ?= 3398

ifeq ($(MODEL_NAME), bert-base-uncased)
	LOAD_MODEL_TYPE := "preloaded"
else
	ifneq ($(CKPT_NUM_FINETUNE), -1)
		LOAD_MODEL_TYPE := "checkpoint"
	else
		LOAD_MODEL_TYPE := "pretrained"
	endif
endif

ifeq ($(MODEL_NAME), roberta-base)
	LOAD_MODEL_TYPE := "preloaded"
endif

TEST_YEAR ?= 2017
TRAIN_YEAR ?= 2018 2019

MAX_TOKEN_LEN ?= 256
START_N ?= 0
TOP_N ?= 50
N_CHARS_TRIM ?= 200
USE_GPU ?= 1
BATCH_SIZE = 8
EPOCHS = 1

TEST_PIPELINE ?= 0
SAVE_FLAG ?= 1
SAVE_ARGPARSER ?= 1

NUM_FOLDS ?= 5
SEED ?= 1

.PHONY: all clean tags info

update_venv: requirements.txt
	rm -rf venv/
	test -f venv/bin/activate || virtualenv -p $(shell which python3) venv
	. venv/bin/activate; \
	pip3 install -Ur requirements.txt; \
	pip3 freeze > requirements.txt
	touch venv/bin/activate

update_req: requirements.txt
	pip3 freeze > requirements.txt

# Incorrect.
# local_module: $(BERT) $(QE) $(PARSER)
# 	. venv/bin/activate; \
# 	pip3 install -e $(BERT) && \
# 	pip3 install -e $(QE) && \
# 	pip3 install -e $(PARSER)

continue_pretrain: $(PRETRAINED_MODELS)/BLUE_BERT $(LM)
	python3 $(LM)/BertFinetuneLm.py \
		--model_name_or_path $(PRETRAINED_MODELS)/BLUE_BERT_HF \
		--model_type "BERT" \
		--train_data_file $(DATA)/trials_raw_sample.txt \
		--line_by_line \
		--mlm \
		--mlm_probability "0.15" \
		--block_size 256 \
		--output_dir $(FINETUNED_MODELS) \
		--do_train \
		--per_device_train_batch_size 8 \
		--num_train_epochs 3 \
		--logging_steps 50 \
		--save_steps 20 \
		--n_gpu 1
# 		--config_name $(PRETRAINED_MODELS)/BLUE_BERT/config.json \
# 		--tokenizer_name $(PRETRAINED_MODELS)/BLUE_BERT/vocab.txt \
# 		-- eval_data_file \
#		--overwrite_cache


build_training: $(PARSER) $(LABELLED_DATA) $(INDEX)
	python3 $(PARSER)/initial_build.py

build_qe: $(QE) $(TRIAL_TOPIC) $(INDEX)
	python3 $(QE)/expand_raw_baseline.py

finetune_model: $(TRIAL_TOPIC)
	echo $(LOAD_MODEL_TYPE)
	python3 run_bert_finetune.py \
		--test_pipeline $(TEST_PIPELINE) \
		--training_data_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--model_desc $(MODEL_DESC) \
		--load_pretrained_path $(PRETRAINED_MODELS) \
		--load_model_type $(LOAD_MODEL_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--max_token_len $(MAX_TOKEN_LEN) \
		--eval_metrics accuracy precision_recall_by_topic \
		--use_gpu $(USE_GPU) \
		--use_ids \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--num_labels 2 \
		--save_flag $(SAVE_FLAG)\
		--save_argparser_filename $(SAVE_ARGPARSER) \
# 		--n_chars_trim $(N_CHARS_TRIM) \
# 		--use_qe \
# 		--ckpt_num $(CKPT_NUM_FINETUNE) \

finetune_model_qe: $(TRIAL_TOPIC_QE_PAPER)
	echo $(LOAD_MODEL_TYPE)
	python3 run_bert_finetune.py \
		--test_pipeline $(TEST_PIPELINE) \
		--training_data_path $(TRIAL_TOPIC_QE_PAPER) \
		--model_name $(MODEL_NAME) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--model_desc $(MODEL_DESC) \
		--load_pretrained_path $(PRETRAINED_MODELS) \
		--load_model_type $(LOAD_MODEL_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--max_token_len $(MAX_TOKEN_LEN) \
		--eval_metrics accuracy precision_recall_by_topic \
		--use_gpu $(USE_GPU) \
		--use_ids \
		--use_qe \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--num_labels 2 \
		--n_chars_trim $(N_CHARS_TRIM) \
		--save_flag $(SAVE_FLAG)\
		--save_argparser_filename $(SAVE_ARGPARSER) \
# 		--ckpt_num $(CKPT_NUM_FINETUNE) \

finetune_model_kfold_cv: $(TRIAL_TOPIC)
	echo $(LOAD_MODEL_TYPE)
	python3 run_bert_finetune.py \
		--test_pipeline $(TEST_PIPELINE) \
		--training_data_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--model_desc $(MODEL_DESC) \
		--load_pretrained_path $(PRETRAINED_MODELS) \
		--load_model_type $(LOAD_MODEL_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--max_token_len $(MAX_TOKEN_LEN) \
		--eval_metrics accuracy precision_recall_by_topic \
		--use_gpu $(USE_GPU) \
		--use_ids \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--num_labels 2 \
		--n_chars_trim $(N_CHARS_TRIM) \
		--save_flag $(SAVE_FLAG) \
		--save_argparser_filename $(SAVE_ARGPARSER) \
		--kfold_cv \
		--num_folds $(NUM_FOLDS) \
		--seed $(SEED) \

finetune_model_qe_kfold_cv: $(TRIAL_TOPIC_QE_PAPER)
	echo $(LOAD_MODEL_TYPE)
	python3 run_bert_finetune.py \
		--test_pipeline $(TEST_PIPELINE) \
		--training_data_path $(TRIAL_TOPIC_QE_PAPER) \
		--model_name $(MODEL_NAME) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--model_desc $(MODEL_DESC) \
		--load_pretrained_path $(PRETRAINED_MODELS) \
		--load_model_type $(LOAD_MODEL_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--max_token_len $(MAX_TOKEN_LEN) \
		--eval_metrics accuracy precision_recall_by_topic \
		--use_gpu $(USE_GPU) \
		--use_ids \
		--use_qe \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--num_labels 2 \
		--n_chars_trim $(N_CHARS_TRIM) \
		--save_flag $(SAVE_FLAG) \
		--save_argparser_filename $(SAVE_ARGPARSER) \
		--kfold_cv \
		--num_folds $(NUM_FOLDS) \
		--seed $(SEED) \

eval_model: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline $(TEST_PIPELINE) \
		--ltr_model $(LTR_MODEL) \
		--trial_topic_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME_EVAL) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--ckpt_num $(CKPT_NUM) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--start_n $(START_N) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE)
# 		--use_qe \

eval_model_test: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline 1 \
		--ltr_model $(LTR_MODEL) \
		--trial_topic_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME_EVAL) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--ckpt_num 3401 \
		--test_year 2019 \
		--train_year 2017 2018 \
		--start_n $(START_N) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE)

eval_model_kfold_cv: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline $(TEST_PIPELINE) \
		--ltr_model $(LTR_MODEL) \
		--trial_topic_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME_EVAL) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS)/$(MODEL_NAME_EVAL)_folds$(NUM_FOLDS)_seed$(SEED) \
		--start_n $(START_N) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE) \
		--kfold_cv \
		--num_folds $(NUM_FOLDS) \
		--seed $(SEED) \
		--topic_year_breakpoints ${TOPIC_YEAR_BREAKPOINTS} \
		--list_ckpt_num ${LIST_CKPT_NUM} \
		--list_years ${LIST_YEARS} \
		# --ckpt_num $(CKPT_NUM) \
		# --test_year $(TEST_YEAR) \
		# --train_year $(TRAIN_YEAR) \

eval_model_kfold_cv_test: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline 1 \
		--ltr_model $(LTR_MODEL) \
		--trial_topic_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME_EVAL) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS)/$(MODEL_NAME_EVAL)_folds5_seed1 \
		--ckpt_num 2 \
		--test_year 2019 \
		--train_year 2017 2018 \
		--start_n $(START_N) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE) \
		--kfold_cv \
		--num_folds $(NUM_FOLDS) \
		--seed $(SEED) \
		--topic_year_breakpoints 30 80 \
		--list_years 2017 2018 2019 \


eval_model_qe: $(TRIAL_TOPIC_QE_PAPER) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline $(TEST_PIPELINE) \
		--ltr_model $(LTR_MODEL) \
		--trial_topic_path $(TRIAL_TOPIC_QE_PAPER) \
		--model_name $(MODEL_NAME_EVAL) \
		--model_class_type $(MODEL_CLASS_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--ckpt_num $(CKPT_NUM) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--start_n $(START_N) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--use_qe \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE)

finetune_sent_model: $(TRIAL_TOPIC)
	echo $(LOAD_MODEL_TYPE)
	python3 run_sent_bert_finetune.py \
		--test_pipeline $(TEST_PIPELINE) \
		--training_data_path $(BIOASQ_TRAIN) \
		--testing_data_path $(BIOASQ_TEST) \
		--model_name $(MODEL_NAME) \
		--model_desc $(MODEL_DESC_SENT) \
		--load_pretrained_path $(PRETRAINED_MODELS) \
		--load_model_type $(LOAD_MODEL_TYPE) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--max_token_len $(MAX_TOKEN_LEN) \
		--eval_metrics accuracy precision_recall_by_topic \
		--use_gpu $(USE_GPU) \
		--use_ids \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--num_labels 2 \
		--save_flag $(SAVE_FLAG)\
		--save_argparser_filename $(SAVE_ARGPARSER) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_qe \
# 		--ckpt_num $(CKPT_NUM_FINETUNE) \

eval_sent_model_test: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline 1 \
		--ltr_model "bert_sent" \
		--df_bert_path $(DF_BERT_PATH) \
		--trial_topic_path $(TRIAL_TOPIC) \
		--model_name basic_bert_sent\
		--save_ckpt_path $(FINETUNED_MODELS) \
		--ckpt_num 4440 \
		--test_year 2018 \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name max_pool \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE)

eval_sent_model: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline $(TEST_PIPELINE) \
		--ltr_model "bert_sent" \
		--df_bert_path $(DF_BERT_PATH) \
		--trial_topic_path $(TRIAL_TOPIC) \
		--model_name $(MODEL_NAME_EVAL_SENT) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--ckpt_num $(CKPT_NUM) \
		--test_year $(TEST_YEAR) \
		--train_year $(TRAIN_YEAR) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE)
# 		--use_qe \

eval_sent_model_qe: $(TRIAL_TOPIC) $(FINETUNED_MODELS)
	python3 $(A2A)/eval.py \
		--test_pipeline $(TEST_PIPELINE) \
		--ltr_model "bert_sent" \
		--trial_topic_path $(TRIAL_TOPIC_QE_PAPER) \
		--model_name $(MODEL_NAME_EVAL_SENT) \
		--save_ckpt_path $(FINETUNED_MODELS) \
		--ckpt_num $(CKPT_NUM) \
		--test_year $(TEST_YEAR) \
		--top_n $(TOP_N) \
		--n_chars_trim $(N_CHARS_TRIM) \
		--use_gpu $(USE_GPU) \
		--save_trec_name $(SAVE_TREC_NAME) \
		--use_solr \
		--base_ranker_type $(BASE_RANKER_TYPE)
# 		--use_qe \

finetune: $(BERT) $(PRETRAINED_MODELS) build_qe build_training
	python3 $(LTR)/not_implemented.py

clean: $(A2A)/logs $(A2A)/results
	rm -rf $(A2A)/results/*
	rm -rf $(A2A)/logs/*

# clean: $(A2A)/logs $(A2A)/results $(DATA)/missing_cache
# 	rm -rf $(A2A)/results/*
# 	rm -rf $(A2A)/logs/*
# 	rm -rf $(DATA)/missing_cache/*

