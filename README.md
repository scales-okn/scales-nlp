# SCALES NLP

This library provides access to the SCALES Inference Pipelines, in addition to a general-purpose API for training and using transformer models.

Try out the demo!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yVu3kMvvQj0JhBUySLpOyTULbq0q7UNT?usp=sharing)

## How to Install

This library is [available on pip](https://pypi.org/project/scales-nlp/) and can be installed as follows:

    > pip install scales-nlp
    
If you would like to use this library with private models on the Hugging Face Model Hub or if you would like to use the Label Studio integration, you can store your credentials and keys by running the following from command line:

    > scales-okn configure

## SCALES Inference Pipelines

All pipelines can be loaded with the `pipeline` function:

    from scales_nlp import pipeline

### Docket Classification

(coming soon)

### Entity Resolution

Load this pipeline as 'entity-resolution'.  This model predicts whether a pair of names corefer to the same entity.  The model expects a list of pairs of names when calling it.

    nlp = pipeline('entity-resolution')
    results = nlp([
        ['American Airlines Corporation', 'American Airlines Inc.'], 
        ['American Airlines Corporation', 'United Airlines'],
    ])


The scales inference module can be used to load general purpose, task-specific pipelines for any model.  To do so pass the Hugging Face Model Identifier to the `pipeline` function and include the `task` argument.

## Task Pipelines

### Classification

Supply the model ID and provide `task='classification'` to load any classification model.

    nlp = pipeline('distilbert-base-uncased-finetuned-sst-2-english', task='classification')
    results = nlp(['I like you.'])

### Multi-Label Classification

For multi-label classification use `task='classification'` and supply `multi_label=True`.

    nlp = pipeline('docketanalyzer/distilroberta-base-ddcl', task='classification', multi_label=True)
    results = nlp([
         "MOTION to Dismiss by Hanson Alaska Professional Services, Inc.. (Attachments: # 1 Proposed Order)(Scanlan, Terence) (Entered: 07/20/2016)", 
         "ORDER granting 7 Motion to Dismiss for Failure to State a Claim. Plaintiff's complaint is dismissed with prejudice. Signed by Judge H. Russel Holland on 8/31/17. (JLH, COURT STAFF) (Entered: 08/31/2017)",
    ])

### Named Entity Recognition

(coming soon)

# Developer Guide

Developers should take the following steps to set up their access.

1.   Request access to the [SCALES-OKN Hugging Face Organization](https://huggingface.co/scales-okn) and generate a Hugging Face API Token.
2.   Configure your access using the developer flag by running `scales-nlp configure --dev` on command line (if you run this on Delilah, all of your access tokens should be pre-populated except for the Hugging Face Token which you can use from step 1).

This will allow you to use our private pipelines and models, as well as the dataloaders for the training data available in the Docket Viewer and SCALES Label Studio applications.

## Developer Pipelines

### Ontology Single Label Classification

Load these pipelines as 'ontology-' + the name of the label you wish to classify (spaces replaces with hyphens).  These are binary classifiers for individual labels.  For example:

    nlp = pipeline('ontology-motion-to-dismiss')
    results = nlp([
         "MOTION to Dismiss by Hanson Alaska Professional Services, Inc.. (Attachments: # 1 Proposed Order)(Scanlan, Terence) (Entered: 07/20/2016)", 
         "ORDER granting 7 Motion to Dismiss for Failure to State a Claim. Plaintiff's complaint is dismissed with prejudice. Signed by Judge H. Russel Holland on 8/31/17. (JLH, COURT STAFF) (Entered: 08/31/2017)",
    ])


## Developer Dataloaders

### Docket Viewer Dataloader

You can pull the consolidated traning data from the Docket Viewer application using the `from_docket_viewer` function and supplying the name of a label.

    data = scales_nlp.data.from_docket_viewer('motion to dismiss')

    
