from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers.integrations import TensorBoardCallback
from datasets import Dataset
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score as sk_f1_score
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from scales_nlp import config
import scales_nlp


class BaseRoutine():
    def __init__(
        self, model_name=config['MODEL_NAME'], max_length=config['MAX_LENGTH'],
        eval_split=config['EVAL_SPLIT'], epochs=config['EPOCHS'], shuffle=config['SHUFFLE'],
        train_batch_size=config['TRAIN_BATCH_SIZE'], eval_batch_size=config['EVAL_BATCH_SIZE'],
        gradient_accumulation_steps=config['GRADIENT_ACCUMULATION_STEPS'],
        learning_rate=config['LEARNING_RATE'], warmup_ratio=config['WARMUP_RATIO'],
        weight_decay=config['WEIGHT_DECAY'], save_steps=config['SAVE_STEPS'],
        callbacks=None, **kwargs
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.eval_split = eval_split
        self.epochs = epochs
        self.shuffle = shuffle
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.callbacks = callbacks
        self.kwargs = kwargs

    def process_labels(self, labels):
        raise NotImplementedError('process_labels not implemented')

    def load_model(self, model_name):
        raise NotImplementedError('load_model not implemented')

    def create_dataset(self, texts, labels, name):
        inputs = self.tokenizer(texts, padding="max_length", max_length=self.max_length, truncation=True)
        inputs['labels'] = torch.Tensor(labels).reshape(len(texts), -1)
        return Dataset.from_dict(inputs)
    
    def load_trainer_class(self):
        if hasattr(self, 'compute_loss'):
            compute_loss_fn = self.compute_loss
            class CustomTrainer(Trainer):
                def compute_loss(self, *args, **kwargs):
                    return compute_loss_fn(*args, **kwargs)
            return CustomTrainer
        else:
            return Trainer

    def train(self, output_dir, texts, labels, push=None, overwrite=False):
        labels, label_names = self.process_labels(labels)
        self.label_names = label_names

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = self.load_model(self.model_name)

        if self.shuffle:
            data = list(zip(texts, labels))
            np.random.shuffle(data)
            texts, labels = zip(*data)
            texts, labels = list(texts), list(labels)
        
        split = len(texts) - int(self.eval_split * len(texts))
        
        train_dataset = self.create_dataset(texts[:split], labels[:split], 'train')
        eval_dataset = self.create_dataset(texts[split:], labels[split:], 'eval')

        trainer_class = self.load_trainer_class()

        output_dir = Path(output_dir)
        args = TrainingArguments(
            output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            eval_steps=self.save_steps,
            save_steps=self.save_steps,
            evaluation_strategy='steps',
            load_best_model_at_end=True,
            save_total_limit=5,
            logging_steps=5,
            logging_dir=output_dir / 'runs',
            hub_strategy='end',
            hub_model_id=push,
            hub_token=config['HUGGING_FACE_TOKEN'],
            push_to_hub=push is not None,
        )

        if output_dir.exists():
            if overwrite:
                shutil.rmtree(output_dir)
            else:
                raise Exception("Output directory already exists, please use the overwrite argument to overwrite it")

        trainer_class_args = dict(
            model=model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if hasattr(self, 'compute_metrics'):
            trainer_class_args['compute_metrics'] = self.compute_metrics
        
        if hasattr(self, 'data_collator'):
            trainer_class_args['data_collator'] = self.data_collator

        self.trainer = trainer_class(**trainer_class_args)

        if self.callbacks is None:
            self.callbacks = [TensorBoardCallback()]

        for callback in self.callbacks:
            self.trainer.add_callback(callback)

        self.trainer.train()
        results = self.trainer.evaluate()
        print(results)

        self.trainer.save_model(output_dir)
        print("Model saved to", output_dir.resolve())
        return results
        

class ClassificationRoutine(BaseRoutine):    
    def process_labels(self, labels):
        label_names = list(sorted(list(set(labels))))
        labels = [[int(label_names[i] == label) for i in range(len(label_names))] for label in labels]
        return labels, label_names

    def load_model(self, model_name):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.label_names))
        model.config.label2id = {self.label_names[i]: i for i in range(len(self.label_names))}
        model.config.id2label = {i: self.label_names[i] for i in range(len(self.label_names))}
        return model
    
    def compute_metrics(self, eval_pred):
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        logits, labels = eval_pred
        logits = torch.softmax(torch.Tensor(logits), dim=-1).numpy()
        predictions = np.argmax(logits, axis=-1)
        labels = np.argmax(labels, axis=-1)
        return {
            'accuracy': acc_metric.compute(predictions=predictions, references=labels)['accuracy'],
            'f1': f1_metric.compute(predictions=predictions, references=labels)['f1'],
            'precision': precision_metric.compute(predictions=predictions, references=labels)['precision'],
            'recall': recall_metric.compute(predictions=predictions, references=labels)['recall'],
        }


class MultiLabelClassificationRoutine(ClassificationRoutine):
    def process_labels(self, labels):
        label_names = list(sorted(list(set([label for label_set in labels for label in label_set]))))
        labels = [[int(label_names[i] in label_set) for i in range(len(label_names))] for label_set in labels]
        return labels, label_names

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        logits = torch.sigmoid(torch.Tensor(logits)).numpy()
        predictions = (logits > 0.5).astype(int)
        scores = sk_f1_score(predictions, labels, average=None)
        scores = {'labels': {self.label_names[i]: scores[i] for i in range(len(scores))}}
        scores['f1_macro'] = sk_f1_score(predictions, labels, average='macro')
        return scores


class TokenClassificationRoutine(BaseRoutine):    
    def process_labels(self, labels):
        label_names = []
        for spans in labels:
            for span in spans:
                label_names.append(span['label'])
        label_names = list(sorted(list(set(label_names))))
        return labels, label_names

    def load_model(self, model_name):
        id2label = {0: 'O'}
        for label_name in sorted(self.label_names):
            id2label[len(id2label)] = 'B-' + label_name
            id2label[len(id2label)] = 'I-' + label_name

        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(id2label))
        model.config.id2label = id2label
        model.config.label2id = {v:k for k,v in id2label.items()}
        return model

    def create_dataset(self, texts, labels, name):
        return scales_nlp.datasets.TokenClassificationDataset(self.tokenizer, texts, labels, self.label_names, max_length=self.max_length)

    @property
    def data_collator(self):
        return DataCollatorForTokenClassification(self.tokenizer)


def training_routine(task, **kwargs):
    task2routine = {
        'classification': ClassificationRoutine,
        'multi-label-classification': MultiLabelClassificationRoutine,
        'token-classification': TokenClassificationRoutine,
        'ner': TokenClassificationRoutine,
    }
    return task2routine[task](**kwargs)
