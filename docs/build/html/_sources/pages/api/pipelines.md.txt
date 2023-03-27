Pipelines
=========

Load a model for inference
----------------------------

You can use the `pipeline` function to load a model for inference by its name.  If you wish to generic Hugging Face model provide the name of a task as the pipeline name along with the `model_name` argument for the model you wish to load.  If the model is private, provide your authentication token with the `auth_token` argument or, alternatively, you can cache your token by running the `scales-nlp configure` from command line and instead just use `auth_token=True`.


```{eval-rst}
.. automethod:: scales_nlp.pipelines.pipeline
```

Create custom pipelines
--------------------------

You can extend the `BasePipeline` class to create your own inference pipelines.  See the `ClassificationPipeline`, `MultiLabelClassificationPipeline`, `TokenClassificationPipeline`, and `SentenceEncodingPipeline` for examples.

```{eval-rst}
.. autoclass:: scales_nlp.pipelines.BasePipeline
    :members:
```