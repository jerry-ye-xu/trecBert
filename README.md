# Table of Contents

- [Introduction](#introduction)
- [Setting up](#setting-up)
- [Building the Datasets](#building_the_datasets)
- [Loading the data](#loading-the-data)
- [Finetuning the Models](#finetuning-the-models)
- [Evaluation End-to-end](#evaluation-end-to-end)
- [Local Parser](#local-parser)
- [Query Expansions](#query-expansions)
- [BERT Models](#bert-models)
- [BERT Sentence Models](#bert-sentence-models)
- [BERT-as-a-service](#bert-as-a-service)
- [Cross Validation](#cross-validation)
- [Multi-task Learning](#multi-task-learning)
- [Worklog](#worklog)

---
<br>

## Introduction

This repository contains Jerry's code for building, training and evaluating various BERT models on the `Clinical Trials` task in the Precision Medicine track of TREC.

This project is part of the Language and Social Computing Team of Data61, and runs over summer of 2019-20.

The published paper, Clinical trial search: Using biomedical language understanding models for re-ranking, can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S1532046420301581)

---
<br>

## Setting up

### VirtualEnv

You can install and update the virtual environment with
```{bash}
make update_venv
```
be sure to activate your virtual environment with
```{bash}
source activate.sh
```
before you run any code.

To install the required libraries, run
```{bash}
pip3 install -r requirements.txt
```

To use the same venv on Jupyter, install a new kernel with
```{bash}
ipython kernel install --name "local-venv" --user
```
You can check if Jupyter is pointing to the right Python location with
```{python}
import sys

print(sys.executable)
```
### Changing relative paths

You may need to change a few relative/absolute paths in A2A4UMA. Search for `defaults.json` and neighbours for more clarity.

## Miscellaneous

You'll need to create some directories for debugging.

Just create
```{bash}
mkdir debug_data
```
in the root directory

### Manually Installing Local Modules

Also, you may want to install all the local modules so that everything is up to date. I've spent wasted minutes debugging an error because my local modules were out of date after installing `requirements.txt`

```{bash}
source activate.sh
pip3 install -e local_parser
pip3 install -e query_expansion
pip3 install -e bert_seq_class
```
and anything else I might have missed.

---
<br>

## Building the Datasets

### Downloading the Data from TREC

You will need the clinical trials XML file which can be found [here](http://www.trec-cds.org/2018.html). This is the 2018 clinical trials data, which is a snapshot up to of April 2017.

Download, extract and save all of this inside the `./data` folder.

__The `trec_eval` data and `relevant_judgement` data is already inside the repo, but if you want to download it, follow the instructions below.__

To download the `trec_eval` data and `relevant_judgement` files for 2017-8, preceed to
'[past data](https://trec.nist.gov/data/precmed.html)' of the TREC webpage.

Save the files into separate directories, naming them by year.
```{bash}
pm_labels_<year>
```

Rename all of the files with their year attached, e.g.
```{bash}
abstracts_judgments_<year>.csv
clinical_trials_judgments_<year>.csv
qrels_sample_abstracts_<year>.txt
qrels_sample_trials_<year>.txt
qrels_treceval_abstracts_<year>.txt
qrels_treceval_clinical_trials_-<year>.txt
topics2018.xml
```
with all underscores and remove the "version numbers".

### Whoosh Storage

You will need the trials and abstract documents that are stored by Whoosh.

Please ask Maciek to give you the required files (it is quite big, \~4GB).

Store these in
```{python}
./A2A4UMA/indices
```

### Solr

Put the indexes in
```
${PATH_TO_DIR}/server/solr/<index>
```

To run the solr server, you'll need to kickstart the server with
```{bash}
A2A4UMA/solr-8.2.0/bin/solr start
```
Check that it's running by going to
```{bash}
localhost:<port_number>
```
```{bash}
A2A4UMA/solr-8.2.0/bin/solr stop -all
```

Note: You won't be able to multi-process with solr index, but it's fast enough so that this shouldn't be a worry.
Note 2: If you want to make it work, `extract_trial_data.py` holds the code that needs to be changed.

__DFR and BM25__

Inside `A2A4UMA/solr-8.2.0/server/solr/ct2017/conf/managed-schema`, you want to add one of the two similarity metrics to use.

You will also need to change relevant file for `ct2019` as well.

__DFR__
```
<similarity class="solr.DFRSimilarityFactory">
  <str name="basicModel">I(n)</str>
  <str name="afterEffect">L</str>
  <str name="normalization">H2</str>
  <float name="c">1</float>
</similarity>
```
__BM25__
```
<similarity class="solr.BM25SimilarityFactory">
   <float name="k1">1.2</float>
   <float name="b">0.75</float>
</similarity>
```

__Note: Remember to RESTART Solr after you change the parameters!__

### Building the pickle files

Now that you have the data required,

Create the training data for finetuning the BERT models
```{bash}
make build_training
```
which creates `trials_topics_combined_all_years.pickle` files.

Note: We can build 2 years worth of data at a time before we run out of memory on my personal computer, so the code is altered slightly to reflect that.
<br>
Note2: This takes a few hours to run, so feel free to get a cup of coffee

__Query Expansion__

With this you can run
```{bash}
make build_qe
```
to build query expansion (QE) on top of the training data. `trials_topics_combined_all_years_qe.pickle` contains all query expansions possible. `trials_topics_combined_all_years_qe_paper.pickle` contains only the QE done in this [paper](/papers/CCNL.PM.pdf).

---
<br>

## Finetuning the Models

There are two places where you can finetune the models

1) Colab
2) GCP VM Instance

I'll briefly run through the steps.

### Colab

Put the relevant data onto Colab and run the `4_finetune_tmp.ipynb` inside `trials_finetune` directory.

Be sure to authenticate your drive directories!

### GCP VM

Once you spin up a VM, be sure to set `enable-oslogin` and if you want to, put the data in Cloud Storage to transfer the data over.

SSH into the VM, set up your virtual environment, and run the following code
```{bash}
nohup bash run_finetune_makefile.sh &
```
This will run the process in the background and ignore the SIGHUP (signal hangup) that kills your processes when you disconnect from the terminal.

If you don't do this, you will lose your training session if you ever accidently disconnect from the VM!!

Be sure to watch the process with `top` and `nvidia-smi`. You can also check the checkpoints being churned out at `./data/finetuned_models/`.

---
<br>

## Evaluation End-to-end

The pipeline is as follows; we take bm25f (amongst other) parameters to produce a base ranking.

This then fed into the learning-to-rank (LTR) model (in this case BERT and it's variants) to rerank the top `n` documents.

The reranked results sent to be evaluated and the results are saved.

We refer to bm25f as the base ranker and LTR as the reranker.

In the `utils` directory, you will need to change the absolute path the TREC eval scripts in `defaults.json`.

Paths provided in the `defaults.json` file have to be relative for MacOS users. Leaving in numbers and dots may result in errors. I re-cloned the `trec_eval` repo (see below) and renamed it to have no space and no dots.
```{bash}
./A2A4UMA/utils/eval/trec_eval/trec_eval
```
This file is required for running the evaluations.

Modules e.g. code for LTR models plug-in should stored in the modules inside `core`, and contain a function called `run`.

You'll need to specify a "rerank_dict" that holds the parameters you need.

### Segmented Version

Note: This is NOT the recommended way to do this.

If you are not inside a VM or using a GPU, then the time it takes to run an evaluation will be fairly long.

Finetuning is out of the question - it'll be 3 hours to compute one epoch...

Hence we produce a work-aroound such that the base ranker's results are outputted and sent to Colab. The LTR model reranks and then sends the results back down for evaluation.

Right before calling `mod.run()` (i.e. LTR models), we save the query results to `pre_rerank_data_files` with which we take to `Colab 9` after wrangling.

So after running `eval.py` you'll obtain an intermediate pickle file.

To create a DataFrame for input into BERT models, use

```{bash}
python3 ./local_parser/ingest_preranking.py
```

You can then load this into `Colab 9` for evaluation, and output the reranking scores with which to add to the original scores. Then bring the reranking scores back to your local computer and write a run function to return this file.

### Full Version

Well, in this case you'll specify what to do with the Makefile...

To evaluate, run something along the lines of
```{bash}
make eval_model MODEL_NAME="basic_bert" MODEL_DESC="basic_bert" TRAIN_YEAR='2018 2019' TEST_YEAR=2017 CKPT_NUM=3398 TEST_PIPELINE=1
```
See `Makefile_scripts.txt`.

I have also created a full script that runs every single evaluation and finetune. This works wonders if combined with the GCP VM.

### Retrieving Missing Documents

The base ranker may retrieve a set of documents different to what we have in the training data - this is to be expected.

These missing documents are noted by ID and then queries to retrieve their information for evaluation.

In order to speed up this process, we save a copy of the documents in `./data/missing_cache` by year and topics.

Anytime the documents retrieved by the base ranker changes, you will need to wipe the cache and restart.

There may be a tedious way to map the parameters of the base_ranker to the cache but that is an exercise for another day.

If you are getting an Attribute error like below, it's likely that there's NaN values in your dataframe because your `missing cache` does not contain some of the doc ids.

```
Traceback (most recent call last):
  File "./A2A4UMA/eval.py", line 542, in <module>
    proc=1, tids={})
  File "./A2A4UMA/eval.py", line 197, in eval_bm25_edismax
    proc=proc, tids=tids)
  File "./A2A4UMA/core/pipe_conductor.py", line 256, in run_full_process
    rankings = run_ranking_pipeline(rankings, topics, reranking_steps)
  File "./A2A4UMA/core/pipe_conductor.py", line 88, in run_ranking_pipeline
    results = mod.run(result_sets, topics, **step['params'])
  File "./A2A4UMA/core/modules/ranking/bert_model_eval/bert_model_eval.py", line 83, in run
    global_var=global_var
  File "./A2A4UMA/../bert_seq_class/BertSeqClassProcessData.py", line 194, in split_into_bert_input
    seq_a = [process_raw_array(x) for x in list(df_input[seq_a_col_name])]
  File "./A2A4UMA/../bert_seq_class/BertSeqClassProcessData.py", line 194, in <listcomp>
    seq_a = [process_raw_array(x) for x in list(df_input[seq_a_col_name])]
  File "./A2A4UMA/../bert_seq_class/BertSeqClassProcessData.py", line 236, in process_raw_array
    text = text.replace("-", "")
AttributeError: 'float' object has no attribute 'replace'
```
Just remove the files and rerun the code again.

### Evaluating with infNDCG

Note: This cannot be done for 2017 data because topic 10 is missing (removed because it contains no relevant documents...

---
<br>

## BERT Models

You can read the original paper [here](https://arxiv.org/abs/1810.04805).

We utilise HuggingFace's repo __Transformers__, which can be found [here](http://github.com/huggingface/transformers/).

### Miscellaneous Details

[Here](https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch) is a good explanation about why we need `model.zero_grad()`

[Here](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/7) is a quick explanation on `torch.no_grad()` and `model.eval()`.

### Loading bioBERT TF into Pytorch (HF-transformers)

Note: This needs to be done on Tensorflow!

For the docs see [here](https://huggingface.co/transformers/main_classes/model.html)

The template of adding new models, one of which includes a script to convert from TF to Pytorch might be useful, see [here](https://github.com/huggingface/transformers/blob/e92bcb7eb6c5b9b6ed313cc74abaab50b3dc674f/templates/adding_a_new_model/convert_xxx_original_tf_checkpoint_to_pytorch.py)

Note 2: A kind person has managed to convert the weights. See the Github issues [here](https://github.com/dmis-lab/biobert/issues/26).

### BlueBERT

Check out blueBERT [here](https://github.com/ncbi-nlp/bluebert), yet another BERT variant trained on preprocessed PubMed texts and MIMIC-III.

### Evaluation Functions

__Binary accuracy__: As you would expect

__Precision-recall by topic__: Confusion matrix, but for each individual topic.

__ROC Curve__: As you would expect.

---
<br>

---
<br>

## BERT Sentence Models

We use `BioASQ-training8b` dataset to build a sentence-level classifier.

The negative samples are constructed from using a different sentence in the same document (hard negatives) and a random sentence not relevant to other questions from another document (easy negatives).

Since we using a fundamentally different distribution of text and applying the model to our clinical trials data, we can argue that this is a type of zero-shot learning.

See `bert_seq_sent` and `local_parser/build_bioasq_df.py` for more information.

---
<br>

## BERT-as-a-service
```{bash}
bert-serving-start -model_dir ./data/pretrained_models/scibert_scivocab_uncased_tf/ -num_worker=4
```
To encode text, simply
```{python}
from bert_serving.client import BertClient
bc = BertClient()
bc.encode(['First do it', 'then do it right', 'then do it better'])
```
See [here](https://github.com/hanxiao/bert-as-service) for repo and details.

If you want to run this on Colab see this issue [here]. I couldn't get it to work.

For this use case specifically, we use
```{bash}
bert-serving-start -model_dir ./data/pretrained_models/scibert_scivocab_uncased_tf/ -num_worker=4 -max_seq_len=128
```

---
<br>

## Cross Validation

There were a few changes we needed to apply in order to make cross-validation work.

__Finetuning__

A new function was created to wrap the original `run_finetune` function around, and we apply the split using sklearn's `KFold` function. We ensure consistency by setting `seed` and `num_folds`.

A few downstream functions were adjusted to not use `train_year` and `test_year` when loading in the data.

The naming of the saved models follows "model_desc_batch" in the `run_bert_finetune.py` file.

During evaluation, the first part is called directly in Makefile with "--save_ckpt_path" flag and the batch number directory is called internally inside `bert_model_eval.py`.

__Evaluating__

We needed to create a combined `qrels` and `topics*.xml` file such that the evaluation library we utilise reads in those single files to evaluate the rerankers.

Using these combined models, we'll need to dynamically adjust the `qrels` file at run-time s.t. the evaluation isn't skewed because it's missing topics.

Another thing to be wary of is to

---
<br>

## Multi-task Learning

### One-hot Vectors for Fields

For each field, create a one-hot vector representing each unique word e.g. gene and disease, and take a count of how many times these words appear in each document.

The labels for these vectors are constructed using labels from the relevance judgement files. See the `aspect_parser` directory for more details.

Lemmatisation was done using Spacy's lemmatiser, with hyphens being removed from the infixes s.t. words like `fine-tuning` will be kept as one word.

### Building NER Parser for Genes

We use `BertTokenClassification` model from HuggingFace and the `BC2GM-IOB` dataset. The data is trained on Colab in `trec_pm_classifier/2_gene_ner.ipynb`.

The plan is to use this model to identify genes in text to be used for downstream tasks.

## Formatting Markdown

See Bitbucket's example README [here](https://bitbucket.org/tutorials/markdowndemo/src/master/README.md)

---
<br>

## Worklog
- 14/03/20 | 0.0.42-rc - Re-running sciBERT for year-by-year evaluation due to preliminary results with cross-validation.
- 09/03/20 | 0.0.41-rc - Rewriting the `dynamic_file_generation_for_kfold_cv_eval.py` script to remove unwanted topic labels. Update the same file to automatically remove topics `(10, 2017), (32, 2019), (33, 2019)` as the qrels files does not contain any positive labels. Finally, adjust subsetting of topics with index, as KFold subsets via index. 
- 06/03/20 | 0.0.40-rc - testing evaluation, writing `run_eval_kfold_cv.sh` script. 
- 24/02/20 | 0.0.39-rc - k-fold cross validation added for evaluation. Locally tested, but not yet fully run on remote VM.
- 21/02/20 | 0.0.38-rc - k-fold cross validation added for finetuning. `run_bert_finetune.py` and others adjusted to accommodate for this change.
- 18/02/20 | 0.0.37-rc - Update code to include `roBERTa` as part of finetuning and evaluation. Successful finetune and evaluation of `blueBERT`.
- 17/02/20 | 0.0.36-rc - Update missing cache to differentiated by `base_ranker_type`.
- 14/02/20 | 0.0.35-rc - Successfully test bioBERT on all data. Results are clearly better than the Lucene base ranker.
- 12/02/20 | 0.0.34-rc - Concatenate all representations together and establish baseline on fully connected layer, no dropout no nothing.
- 06/02/20 | 0.0.33-rc - Build word vector representations using BaaS with sciBERT, specifically for judgement files.
- 04/02/20 | 0.0.32-rc - Built one-hot vector encoding a single query field using Spacy's lemmatiser. Chose to have no hyphen ("-") split for tokenisation.
- 03/02/20 | 0.0.31-rc - Fix issue with outputting reranking scores.
- 02/02/20 | 0.0.30-rc - Add new `bert_token_class`, for training NER model to recognise genes. New parser for data format also introduced. This was done on Colab.
- 30/01/20 | 0.0.29-rc - Change indexing engine to Lucene. The results are significantly better, both in performance and speed.
- 23/01/20 | 0.0.28-rc - Debug `treceval` issues on GCP VM, run evaluation for `max pool` and `avg pool` for all 3 years, with one-third of the data.
- 22/01/20 | 0.0.27-rc - Implement pipeline for evaluation, debugged the 'index issue' for longer than necessary, did test run with pooling function and `epoch = 1` finished for `Colab`.
- 21/01/20 | 0.0.26-rc - Added finetuning code to Colab notebook, added the flag to Makefile and components of the A2A4UMA. Testing finetuning and evaluation pipeline on Makefile scripts.
- 20/01/20 | 0.0.25-rc - Added in script to produce negative samples, both weak and strong labels.
- 15/01/20 | 0.0.24-rc - Updated eval script to receive bert and base ranker. Evaluation conducted on all 3 years with all 3 models.
- 13/01/20 | 0.0.23-rc - Add optimiser and scheduler save files, minor tweaks to code and testing epoch training on subsets of data.
- 09/01/20 | 0.0.22-rc - Revamp code for training on GCP instance. Code is now running remotely and in `Colab 4_finetune` at the same time. Finished evaluation script to run on VM.
- 07/01/20 | 0.0.21-rc - Rewrite query expansion options to include specifically the TREC 2019 paper. Update warmup and total steps for `get_linear_warmup_scheduler`. Added optimal parameters and rebuilt `missing cache`. Added `argparser.json` save file and `ckpt_eval.pickle` files for every 3rd checkpoint. For a subset of the data (8 topics x 2 years), finetuning results are fine for scibert when training more epochs. `qe_paper` models are fine.
- 06/01/20 | 0.0.20-rc - A day of finetuning, rewrote the Makefile to be more concise and have variables to be set on the CLI. Check all 6 scripts in `Makefile_scripts`. Debugging finetuning results.
- 02/01/20 | 0.0.19-rc - Introduced `BertSeqClassGlobalVar.py` file to store `df_col_names`, `subset_columns` as well as easy to passing around global variables across multiple files. Tested the finetuning scripts on a brand new notebook that runs everything on `BertSeqClass` functions.
- 01/01/20 | 0.0.18-rc - Changed `bert_model.py` to `bert_model_eval.py` and added `run_bert_finetune.py` to have a modularised script for finetuning models. Tested this by adding in a shortcut in the Makefile and not waiting 50 minutes for the finetuning to run on CPU. Continue to remove use of `CAPITAL_VAR` in code.
- 31/12/19 | 0.0.17-rc - Add `bert_model.py` "module" to `A2A4UMA`. Upgraded BERT model dir to be installable, connected the pipe end-to-end with `basic_bert` ckpt as test case. Successful test run end-to-end.
- 30/12/19 | 0.0.16-rc - Tried to use TF2.0 + Keras API to build a BERT model. Have yet to find a satisfactory solution. I may have to resort to using TF1.15 as show in previous guides to make it happen.
- 26/12/19 | 0.0.15-rc - Added some extra processing steps before tokenising the input for BERT models. Output QE for training data and finetuned the models on this dataset.
- 26/12/19 | 0.0.14-rc - Created a `BertSeqClassRun.py` file for running finetuning and validation sets. This will be part of the final E2E workflow. Created the full version with all ~30 topics with 2017 data ran a training session in `Colab 9`. Built the query expansion (QE) version of full datasets, `df_for_bert` as well as `trials_topics_combined_all_years`. Changed path files to run from root of repo instead of inside the respective folders.
- 24/12/19 | 0.0.13-rc - Moved BERT into 3 separate files, the `model` itself, `DataLoader` and `DataProcessing` stage. Created 2 new notebooks `Colab 8 & 9` in lieu of this, one that trains all BERT models and outputs checkpoints, and the other to test the plug and play of the pipeline.
- 24/12/19 | 0.0.12-rc - Refactor BERT to work with non-labelled test (previously we just used validation data and they had labels) data.
- 23/12/19 | 0.0.11-rc - Connect ingested data to be the same format as what BERT model requires.
- 23/12/19 | 0.0.10-rc - Introduce new Dataset class s.t. unique IDs to each testing sample can be passed in and returned in the same order. Change `parser` to `local_parser` and make it an installable module.
- 19/12/19 | 0.0.9-rc - Implement ROC curve, set seed on all random functions to ensure reproducibility.
- 18/12/19 | 0.0.8-rc - Convert TF bioBERT to Pytorch file. Unsuccessful test run of bioBERT on Colab. The TF weights were not being loaded in correctly.
- 17/12/19 | 0.0.7-rc - Write a `main.py` function for obtaining the pickle files from raw TREC data, requires Whoosh indexed data.
- 16/12/19 | 0.0.6-rc - Implement bio, sci and clinical BERT and check results.
- 15/12/19 | 0.0.5-rc - Implement precision-recall by topic evaluation metric for the results
- 12/12/19 | 0.0.4-rc - First test run of BERT with clinical trials, corrected mistake with evaluation metric that was giving incorrect binary accuracies.
- 09/12/19 | 0.0.3-rc - First test run of BERT with BBCsports, obtained the ent
- 06/12/19 | 0.0.2-rc - Implemented extracting all queries by ID,
- 04/12/19 | 0.0.1-rc - Re-implement BERT example with minor formatting changes.

