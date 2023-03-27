import requests
import torch
from tqdm import tqdm
from scales_nlp import config




class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, spans, label_names, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.spans = spans
        self.label_names = label_names
        self.label_map = {'O': 0}
        for label_name in sorted(label_names):
            self.label_map['B-' + label_name] = len(self.label_map)
            self.label_map['I-' + label_name] = len(self.label_map)
        self.max_length = max_length

    def __getitem__(self, i):
        text = self.texts[i]
        inputs = self.tokenizer(text, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt", return_offsets_mapping=True)
        example = {k: v[0] for k, v in inputs.items()}
        example['labels'] = self.example_spans_to_labels(example, self.spans[i])
        del example['offset_mapping']
        return example

    def __len__(self):
        return len(self.texts)

    def example_spans_to_labels(self, example, spans):
        input_ids = example['input_ids']
        spans = sorted(spans, key=lambda x: x['start'])

        labels = []
        current_label = None
        for i in range(len(example['input_ids'])):
            offset = example['offset_mapping'][i]
            
            if len(spans) > 0 and offset[0] >= spans[0]['end']:
                spans.pop(0)
                current_label = None

            if offset[1] == 0:
                labels.append(-100)
                current_label = None
            elif len(spans) == 0 or offset[1] <= spans[0]['start']:
                labels.append(0)
                current_label = None
            else:
                if current_label is None:
                    current_label = spans[0]['label']
                    labels.append(self.label_map['B-' + current_label])
                else:
                    labels.append(self.label_map['I-' + current_label])

        return torch.tensor(labels)

