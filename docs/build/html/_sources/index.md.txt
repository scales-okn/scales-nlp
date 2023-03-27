SCALES-NLP Documentation
========================

```{eval-rst}

**SCALES-NLP** is an AI toolkit for legal research that includes a collection 
of tools and deep learning models that can be used for the following:

- Download dockets from PACER and parse their contents
- Classify the text of docket entries with 70+ labels
- Extract a range of named entites from docket entries
- Apply a case-level logic to the extracted data to identify key opening and dispositive events according the the **SCALES-OKN Litigation Ontology**


This work is part of `SCALES-OKN <https://scales-okn.org/>`_ and builds on the `PACER-Tools <https://github.com/scales-okn/PACER-tools>`_ module.  To learn more about this
and related projects check out the `SCALES-OKN Documentation <https://docs.scales-okn.org/>`_.


### Get Started

.. code-block:: console

   $ pip install scales-nlp torch


.. code-block:: console

   $ scales-nlp configure


.. code-block:: console

   $ scales-nlp download 'azd;;3:18-cv-08134'


.. code-block:: console

   $ scales-nlp parse


.. code-block:: console

   $ scales-nlp predict


.. code-block:: console

   import scales_nlp

   docket = scales_nlp.Docket.from_ucid('azd;;3:18-cv-08134')
   
   print(docket.ucid)
   print(docket.court)
   print(docket.header.keys())
   for entry in docket:
      print()
      print(entry.row_number, entry.entry_number, entry.date_filed)
      print(entry.text)
      print(entry.event)
      print(entry.labels)
```


```{toctree}
---
maxdepth: 2
caption: Overview
---
pages/overview/start
pages/overview/data
pages/overview/inference
pages/overview/research
```

```{toctree}
---
maxdepth: 2
caption: Litigation Ontology
---
pages/ontology/events
pages/ontology/labels
pages/ontology/entities
```


```{toctree}
---
maxdepth: 2
caption: Package Reference
---
pages/api/pipelines
pages/api/routines
pages/api/cli
```

