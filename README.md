# SCALES NLP

**SCALES NLP** is an AI toolkit for legal research.  

[Check out the docs!](https://docs.scales-okn.org/nlp/)

## Quickstart

### Installation

This module can be installed with pip:
```
$ pip install scales-nlp
```

You should also have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed for your system.


### Configuration

To get the most out of this module, run the following to set module-wide configuration variables. These variables can also be set or overriden by your environment.

```
$ scales-nlp configure
```

- `PACER_DIR` The folder where all of your PACER data will be saved and managed. This should be the same top-level directory that you use with the scraper and parser.
- `PACER_USERNAME` The username to your PACER account.
- `PACER_PASSWORD` The password to your PACER account.
- `HUGGING_FACE_TOKEN` You only need to include this if you want to use the pipelines API with private models on Hugging Face.

In addition to these variables, you can also configure the default values that are used by the training routines API. These can be set by running `scales-nlp configure train-args`.


## Collect PACER Data

This module includes simplified wrappera for the SCALES scraper and parser. This version eschews much of the functionality in order to make it easy to download a single case.  For downloading bulk PACER data or taking advantage of the wide-range of functionality available in the original implementation please reference the [PACER Tools](https://github.com/scales-okn/PACER-tools) package that this is based on.

### Simplified Scraper

To download a single case provide a UCID to following command. The UCID consists of the court abrreviation and the docket number, seperated by ';;'.  For example: `azd;;3:18-cv-08134`


```
$ scales-nlp download [UCID]
```

### Simplified Parser

Run the following to run the parser across all of your downloaded cases.  You may also provide a court abbreviation as an argument if you only want to apply the parser to cases within a single court.

```
$ scales-nlp parse
```

## Apply SCALES Models

### Update Classifier Labels

The following command can be used to compute and update computed classifier labels from the SCALES litigation ontology to new data. By default the model outputs will be cached in your PACER_DIR and the model will only be applied to new cases that do not already have saved predictions.  You can override saved predictions by passing the `--reset` flag.  For optimal performance it is recommended that you only perform inference using the SCALES models on a device with a GPU.  If you run into memory errors, try adjusting the `--batch-size` to your needs.

```
$ scales-nlp update-labels --batch-size 4
```

### Update Named-entity Extraction

SCALES will release several NER models in the near future.

## Loading Tagged Dockets

Downloaded cases can be loaded using the SCALES NLP `Docket` object as follows.  If classifier labels or entity spans have been computed for the case these will accessible as well.  Furthermore, the `Docket` object will consolidate all of the information available in the case to infer the specific pathway events.  To learn more about labels, entities, and pathway events, check out the [Litigation Ontology Guide](https://docs.scales-okn.org/guide/ontology/).

```
import scales_nlp

docket = scales_nlp.Docket.from_ucid("CASE UCID")

print(docket.ucid)
print(docket.case_name)
print(docket.header.keys())

for entry in docket:
    print(entry.row_number, entry.entry_number, entry.date_filed)
    print(entry.text)
    print(entry.event)
    print(entry.labels)
    print(entry.spans)
    print()
```
    
