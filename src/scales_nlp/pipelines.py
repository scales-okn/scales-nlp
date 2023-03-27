from typing import Union, Optional, List, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch
import numpy as np
from toolz import partition_all
from tqdm import tqdm
from scales_nlp.utils import convert_default_binary_outputs
from scales_nlp import config


class BasePipeline(object):
    """Base class for all pipelines."""
    def __init__(self, model_name: str, max_length: int = 512, device: int = None, use_auth_token: bool = False, **kwargs):
        """Initialize the pipeline.

        :param model_name: Name of the Hugging Face model to use
        :param max_length: Maximum length of input tokens
        :param device: Device to use for inference. Use -1 for CPU, or the index of the GPU, if None, then will use GPU if available.
        :param use_auth_token: Whether to use the Hugging Face auth token to download the model
        """

        use_auth_token = use_auth_token if not self.require_auth_token else True
        if use_auth_token == True and config['HUGGING_FACE_TOKEN'] is not None:
            use_auth_token = config['HUGGING_FACE_TOKEN']

        self.device = device if device is not None else -1 if not torch.cuda.is_available() else torch.cuda.current_device()
        self.max_length = max_length
        self.use_auth_token = use_auth_token
        self.model_name = self.get_model_name(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)
        self.model = self.load_model()

    def place_on_device(self, obj: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Method for placing models and tensors on the correct device.

        :param obj: Object to place on device
        :return: Object on device
        """
        if self.device != -1:
            return obj.to(self.device)
        else:
            return obj.to('cpu')

    def get_model_name(self, model_name: str, **kwargs) -> str:
        """Trivial method for getting the model name. Pipelines that rely on a specific model should override this method.
        
        :param model_name: Name of the model
        :return: Updated of the model
        """
        return model_name

    def process_inputs(self, examples: List[str]) -> List[str]:
        """Hook for processing inputs before tokenization.
        
        :param examples: Input examples
        :return: Processed inputs
        """
        return examples
    
    def process_predictions(self, examples, predictions):
        """Hook for processing predictions before returning them.
        
        :param examples: Input examples
        :param predictions: Predictions from model
        :return: Processed predictions
        """
        return predictions
    
    def tokenize(self, texts: List[str]):
        """Tokenize method.
        
        :param texts: List of texts to tokenize
        :return: Tokenized inputs
        """
        return self.tokenizer(texts, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
    
    def generate_batches(self, examples: List, batch_size: int) -> List[List]:
        """Generate batches of examples.
        
        :param examples: List of examples
        :param batch_size: Batch size
        :return: List of batched examples
        """
        return [list(batch) for batch in partition_all(batch_size, examples)]
        
    def load_model(self):
        """Load the model. Must be implemented by all pipelines.
        
        :return: Loaded Hugging Face model
        """
        raise NotImplementedError('load_model not implemented')
    
    def batch_predict(self, inputs, **kwargs):
        """Predict on a batch of inputs. Must be implemented by all pipelines.
        
        :param inputs: Batch of inputs
        :return: Predictions
        """
        raise NotImplementedError('batch_predict not implemented')

    def __call__(self, examples: Any, batch_size: int = 4, verbose: bool = True, **kwargs):
        """Main inference method, which returns predictions for a list of example texts.
        
        :param examples: List of example texts
        :param batch_size: Batch size
        :param verbose: Whether to show a progress bar
        :return: Predictions
        """
        examples = self.process_inputs(examples)
        batches = self.generate_batches(examples, batch_size)
        predictions = []
        for batch in tqdm(batches, disable=not verbose):
            inputs = self.place_on_device(self.tokenize(batch))
            predictions += self.batch_predict(inputs, **kwargs)
        return self.process_predictions(examples, predictions)

    @property
    def require_auth_token(self):
        """Override this property to force pipeline to require an auth token."""
        return False


# TASK PIPELINES
class ClassificationPipeline(BasePipeline):
    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)
        model = self.place_on_device(model)
        return model
    
    def batch_predict(self, inputs, return_scores=False, **kwargs):
        outputs = self.model(**inputs)
        logits = outputs.logits.detach().cpu()
        scores = torch.softmax(logits, dim=-1)
        predictions = []
        for i in range(len(scores)):
            prediction = {self.model.config.id2label[label_id]: score.item() for label_id, score in enumerate(scores[i])}
            if not return_scores:
                prediction = max(prediction.items(), key=lambda x: x[1])[0]
            predictions.append(prediction)
        return predictions


class MultiLabelClassificationPipeline(ClassificationPipeline):
    def batch_predict(self, inputs, return_scores=False, prediction_threshold=0.5, **kwargs):
        outputs = self.model(**inputs)
        logits = outputs.logits.detach().cpu()
        scores = torch.sigmoid(logits)
        predictions = []
        for i in range(len(scores)):
            prediction = {self.model.config.id2label[label_id]: score.item() for label_id, score in enumerate(scores[i])}
            if not return_scores:
                prediction = [label for label, score in prediction.items() if score > prediction_threshold]
            predictions.append(prediction)
        return predictions


class TokenClassificationPipeline(BasePipeline):
    def tokenize(self, texts):
        return self.tokenizer(texts, 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True, 
            return_offsets_mapping=True,
            return_tensors='pt'
        )

    def load_model(self):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)
        model = self.place_on_device(model)
        return model
    
    def batch_predict(self, inputs, **kwargs):
        offsets = inputs.pop('offset_mapping')
        outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask'].detach().cpu()
        input_ids = inputs['input_ids'].detach().cpu()
        logits = outputs.logits.detach().cpu()
        scores = torch.softmax(logits, dim=-1)
        labels = torch.argmax(scores, dim=-1)

        predictions = []
        for i, example_scores in enumerate(scores):
            example_offsets = offsets[i][attention_mask[i] == 1].tolist()
            if len(example_offsets) > 1:
                example_offsets[-1] = [example_offsets[-2][1], example_offsets[-2][1]]
            example_labels = labels[i][attention_mask[i] == 1].tolist()
            prediction = []
            for token_idx, label in enumerate(example_labels):
                label_name = self.model.config.id2label[label]
                entity = label_name.split('-')[-1]
                position = label_name.split('-')[0]

                if position == 'B' or \
                        (position == 'I' and (not prediction or prediction[-1]['entity'] != entity)):
                    prediction.append({
                        'entity': entity,
                        'start': example_offsets[token_idx][0],
                        'end': example_offsets[token_idx][1],
                        'score': [example_scores[token_idx][label].item()],
                        'text': [input_ids[i][token_idx].item()],
                    })
                elif position == 'I':
                    prediction[-1]['end'] = example_offsets[token_idx][1]
                    prediction[-1]['score'].append(example_scores[token_idx][label].item())
                    prediction[-1]['text'].append(input_ids[i][token_idx].item())

            for span in prediction:
                span['score'] = np.mean(span['score'])
                span['text'] = self.tokenizer.decode(span['text'])
            predictions.append(prediction)
        return predictions


class SentenceEncodingPipeline(BasePipeline):
    def load_model(self):
        model = AutoModel.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)
        model = self.place_on_device(model)
        return model
    
    def batch_predict(self, inputs, **kwargs):
        outputs = self.model(**inputs)
        return [outputs[0].detach().cpu().numpy()[:,0,:]]
    
    def process_predictions(self, examples, predictions):
        return np.concatenate(predictions, axis=0)


# CUSTOM PIPELINES
class OntologySingleLabelPipeline(ClassificationPipeline):
    def get_model_name(self, model_name, **kwargs):
        return 'scales-okn/ontology-' + kwargs['label_name'].replace(' ', '-')

    def process_predictions(self, examples, predictions):
        return convert_default_binary_outputs(predictions)
    
    @property
    def require_auth_token(self):
        return True


class DocketClassificationPipeline(MultiLabelClassificationPipeline):
    def get_model_name(self, model_name, **kwargs):
        return 'scales-okn/docket-classification'

    def process_predictions(self, examples, predictions):

        updated_predictions = []
        for example, pred in zip(examples, predictions):
            
            updated_predictions.append(pred)

        return updated_predictions


class DocketEncoderPipeline(SentenceEncodingPipeline):
    def get_model_name(self, model_name, **kwargs):
        return 'scales-okn/docket-encoder'
    
    @property
    def require_auth_token(self):
        return True


def pipeline(pipeline_name: str, **kwargs) -> BasePipeline:
    """Load a pipeline for a given task.

    :param pipeline_name: Name of the pipeline to load
    :param model_name: (str, optional) If using a generic task pipeline, must specify a Hugging Face model name
    """

    pipelines = {
        # Generic pipelines
        'classification': ClassificationPipeline,
        'multi-label-classification': MultiLabelClassificationPipeline,
        'ner': TokenClassificationPipeline,
        'token-classification': TokenClassificationPipeline,
        'sentence-encoding': SentenceEncodingPipeline,

        # SCALES-NLP pipelines
        'ontology': OntologySingleLabelPipeline,
        'docket-classifier': DocketClassificationPipeline,
        'docket-encoder': DocketEncoderPipeline,
    }

    model_name = kwargs.pop('model_name', None)

    if pipeline_name in pipelines:
        return pipelines[pipeline_name](model_name, **kwargs)
    else:
        raise Exception("'%s' is not a valid pipeline name." % pipeline_name)