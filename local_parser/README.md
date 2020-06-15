## Introduction

This module contains code that is used to populate the `trial_topics_combined_*.pickle` files and `trials_pickle_<year>` directories.

We require this to build a labelled dataset for the TREC Precision Medicine Track, Clinical Trials task.

This is dependent on the existence of a Whoosh storage (ask Maciek) with the appropriate index.

## Usage

1. To build all the files needed for BERT Model

From the root directory of the repo run
```{bash}
python3 parser/main.py
```

2. To import the module into a file/notebook

First, add the path of the module to the `PATH` environment variable.

```{bash}
export PATH=$PATH:/Users/xu081/Documents/trec_t2/local_parser
```

## Installing the Package

Install the package with no changes
```{bash}
python3 -m pip install ./local_parser
```

Install package in develop mode
```{bash}
pip install -e ./
```

## Using the package inside Jupyter Notebook

As a quick hack, you can as it to your path inside the notebook and then import it.

To check the path, run `python3 -i` and then

```{python}
import os
import sys

import local_parser

path = os.path.dirname(local_parser.__file__)

sys.path.append(path)
```