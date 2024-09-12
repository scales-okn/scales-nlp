import json
from pathlib import Path
from typing import Union, List, Tuple, Dict
import os
import pandas as pd
import requests
import tempfile
import time
from tqdm import tqdm
from toolz import partition_all
import scales_nlp
from scales_nlp import config


PACKAGE_DIR = Path(__file__).parent
PACKAGE_DATA_DIR = PACKAGE_DIR / 'data'

LABEL_DATA_DIR = config['LABEL_DATA_DIR'] if config['LABEL_DATA_DIR'] is not None else config['PACER_DIR']
JUDGE_DATA_DIR = config['JUDGE_DATA_DIR'] if config['JUDGE_DATA_DIR'] is not None else config['PACER_DIR']

COURTS = pd.read_csv(PACKAGE_DATA_DIR / 'courts.csv')
STATES = COURTS['state'].dropna().unique()
DIVISIONS = COURTS['cardinal'].dropna().unique()


def courts() -> pd.DataFrame:
    return COURTS.copy()


def states() -> List[str]:
    return STATES


def divisions() -> List[str]:
    return DIVISIONS


def load_json(path: Union[str, Path]) -> Dict:
    with open(str(path), 'r') as f:
        return json.loads(f.read())


def get_ucid_components(ucid: str) -> Tuple[str, str, str]:
    court = ucid.split(';;')[0]
    docket_number = ucid.split(';;')[1]
    office_number = docket_number.split(':')[0]
    year = docket_number.split(':')[1].split('-')[0]
    return court, docket_number, office_number, year


def case_path(ucid: str, html: bool=False) -> Path:
    file_type = 'html' if html else 'json'
    court, docket_number, _, year = get_ucid_components(ucid)
    filename = docket_number.replace(':', '-') + '.' + file_type
    return config['PACER_DIR'] / court / file_type / year / filename


def load_case(ucid: str, html: bool=False) -> Dict:
    path = case_path(ucid, html)
    if html:
        return path.read_text()
    else:
        return load_json(path)


def load_case_classifier_labels(ucid: str) -> List:
    court, docket_number, _, year = get_ucid_components(ucid)
    filename = docket_number.replace(':', '-') + '.json'
    path = LABEL_DATA_DIR / court / 'labels' / year / filename
    if path.exists():
        return load_json(path)
    else:
        print('labels not computed for {}'.format(ucid))
        return []

def load_case_judge_labels(ucid: str) -> pd.DataFrame:
    filename = ucid.replace(';;', '-').replace(':', '-') + '.jsonl'
    path = JUDGE_DATA_DIR / court / year / filename
    if path.exists():
        return pd.read_json(path, lines=True)
    else:
        print('labels not computed for {}'.format(ucid))
        return pd.DataFrame()


def load_court(court: str) -> Dict:
    courts = COURTS[COURTS['abbreviation'] == court]
    if len(courts) == 0:
        print(f'Court {court} not found')
        return None
    return courts.iloc[0].to_dict()


def convert_default_binary_outputs(
    predictions: List[Union[str, Dict[str, float], bool]]
) -> List[Union[float, bool]]:
    if isinstance(predictions[0], str):
        converted_predictions = [prediction == 'LABEL_1' for prediction in predictions]
    elif isinstance(predictions[0], dict):
        converted_predictions = [prediction['LABEL_1'] for prediction in predictions]
    else:
        converted_predictions = predictions
    return converted_predictions


def crawl_pacer(ucid: str):
    print('WARNING: this command can spend up to 3$ per case (depending on the number of pages)')
    court, docket_number, office_number, year = get_ucid_components(ucid)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        auth = {"user": config['PACER_USERNAME'], "pass": config['PACER_PASSWORD']}
        auth_path = tempdir / 'auth.json'
        with open(auth_path, 'w') as f:
            f.write(json.dumps(auth))
        query = pd.DataFrame({'ucid': [ucid]})
        query_path = tempdir / 'query.csv'
        query.to_csv(query_path, index=False)
        (config['PACER_DIR'] / court / 'html' / year).mkdir(parents=True, exist_ok=True)
        (config['PACER_DIR'] / court / 'json' / year).mkdir(parents=True, exist_ok=True)
        cmd = f"pacer-tools scraper --headless --override-time --docket-input {query_path} -c {court} -nw 1 -cl 1 -m docket -a {auth_path} {config['PACER_DIR'] / court}"
        os.system(cmd)


def parse_pacer_dir(court: str=None):
    if court is not None:
        paths = [config['PACER_DIR'] / court / 'html']
    else:
        paths = config['PACER_DIR'].glob('*/html')
        
    for path in paths:
        cmd = f"pacer-tools parser {path}"
        os.system(cmd)
        time.sleep(0.5)


def update_classifier_predictions(batch_size=8, reset=False):
    paths = list(config['PACER_DIR'].glob('*/json/*/*.json'))
    if not reset:
        paths = [path for path in paths if not Path(str(path).replace('/json/', '/labels/')).exists()]
    if len(paths) > 0:
        batches = list(partition_all(100, paths))
        nlp = scales_nlp.pipeline('multi-label-classification', model_name='scales-okn/docket-classification')
        for batch in tqdm(batches):
            batch_data = []
            for path in batch:
                case = load_json(path)
                case_data = pd.DataFrame(case['docket'])
                case_data['ucid'] = case['ucid']
                case_data['row_number'] = range(len(case_data))
                case_data['labels_path'] = Path(str(path).replace('/json/', '/labels/'))
                batch_data.append(case_data)
            batch_data = pd.concat(batch_data)
            batch_data['labels'] = nlp(batch_data['docket_text'].tolist(), batch_size=batch_size)
            batch_data =  batch_data[batch_data['labels'].apply(lambda x: len(x) > 0)]
            for path, labels in batch_data.groupby('labels_path'):
                labels = labels[['row_number', 'labels']]
                labels['spans'] = labels['labels'].apply(lambda x: [])
                labels = labels.to_dict(orient='records')
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(labels, f)



def html_spans(texts: List[str], spans: List[Dict], info: List[List[str]]=None, label2color: Dict={}):
    html = f"""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-GLhlTQ8iRABdZLl6O3oVMWSktQOp6b7In1Zl3/Jr59b6EGGoI1aFkw7cmDA6j6gD" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js" integrity="sha384-w76AqPfDkMBDXo30jS1Sgez6pr3x5MlQ1ZAGC+nuZB+EYdgRZgiwxhTBTkF7CXvN" crossorigin="anonymous"></script>
    <div class='row mt-5'>
        <div class='col-10 mx-auto' style='max-width: 600px;'>
            <NEXT>
        </div>
    </div>
    """
    for i in range(len(texts)):
        text = texts[i]
        for span in sorted(spans[i], key=lambda x: -x['start']):
            color = label2color.get(span['entity'], "rgb(205, 255, 255)")
            text = text[:span['start']] + f"<span style='background-color: {color}'>{text[span['start']:span['end']]}</span>" + text[span['end']:]
        entry_html = f"<p class='mb-0 mt-3'>{text}</p>"
        if info is not None:
            info_html = "<div>" + "</div><div>".join(info[i]) + "</div>"
            entry_html += f"<div class='d-flex justify-content-between'>{info_html}</div>"
        entry_html += f"<NEXT>"
        html = html.replace("<NEXT>", entry_html)
    html = html.replace("<NEXT>", '')
    return html


def generate_label_studio_data(texts: str, spans: List[Dict]=None):
    data = []
    for i in range(len(texts)):
        row = {'data': {'text': texts[i]}}
        if spans is not None:
            text_spans = []
            for span in spans[i]:
                text_spans.append({
                    'type': 'labels',
                    'to_name': 'text',
                    'from_name': 'label',
                    'value': {
                        'start': span['start'],
                        'end': span['end'],
                        'text': span['text'],
                        'labels': [span['entity']]
                    },
                })
            row['predictions'] = [{
                'model_version': 'v1',
                'result': text_spans,
            }]
        data.append(row)
    return data
