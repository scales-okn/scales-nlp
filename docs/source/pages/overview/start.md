Getting Started
===============

Installation
-------------

```{eval-rst}
You can install this library with pip:

.. code-block:: console

   $ pip install scales-nlp

.. note::
   To use a GPU (recommended) your PyTorch installation must be compatible with your version of CUDA.  
   See the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_ for more information.
```


Configuration
-------------

```{eval-rst}
Some features use package wide **configuration variables** that can be set using the CLI.
These configuration variables are bundled into groups.  To set their values run the following:

.. code-block:: console

   $ scales-nlp configure
```

To get the most out of this library, set values for the following variables:

**PACER_DIR:** The path to a directory where your PACER data will be stored.  Any data downloaded using the crawler and model updates using the CLI will automatically store and manage your data here.

**PACER_USERNAME:** Username for your PACER account.  Only needed if using the crawler.

**PACER_PASSWORD:** Password for your PACER account.

**HUGGING_FACE_TOKEN:** Only needed if using the Pipelines API with private models.

Additionally you can run `scales-nlp configure train-args` to modify the default training hyperparameters for the TrainingRoutines API.

```{eval-rst}
.. note::
   **Environment variables** can be used to override the values of any configuration variables
   with the same name.  Use your environment to create localized, custom configurations that deviate 
   from the package-wide configuration set using the CLI.
```

Access configuration variables from within the module using the `scales_nlp.config` object.

```{eval-rst}
.. code-block::

    from scales_nlp import config

    print(config['PACER_DIR'])
```