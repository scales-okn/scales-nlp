import json
from pathlib import Path
from typing import Union, List, Tuple, Dict
import os
import pandas as pd
import requests
import tempfile
import time
import glob
from tqdm import tqdm
from toolz import partition_all
from datetime import datetime

# this will probably break if run out of the repo rather than the pypi package
import scales_nlp
from scales_nlp import config
from scales_nlp import docket as docket_functions
from disambiguation_scripts import extract_new_cases
from disambiguation_scripts import JED_Main_public as jed


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
    court, _, _, year = get_ucid_components(ucid)
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
        # print(f'Court {court} not found') # i don't think anyone uses anything besides 'abbreviation', so this shouldn't matter...
        return {'abbreviation': court}
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


def update_classifier_predictions(indir=None, outdir=None, batch_size=8, reset=False):
    paths = list(Path(indir).glob('*.json') if indir else config['PACER_DIR'].glob('*/json/*/*.json'))
    if not reset:
        length_old = len(paths)
        paths = [path for path in paths if not (
            Path(f"{outdir}/{str(path).split('/')[-1]}") if outdir else Path(str(path).replace('/json/', '/labels/'))).exists()]
        print(f'reset=False; reduced paths list from {length_old} paths to {len(paths)} paths')
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
                case_data['labels_path'] = Path(f"{outdir}/{str(path).split('/')[-1]}" if outdir else str(path).replace('/json/', '/labels/'))
                batch_data.append(case_data)
            batch_data = pd.concat(batch_data)
            batch_data['labels'] = nlp(batch_data['docket_text'].tolist(), batch_size=batch_size)
            batch_data = batch_data[batch_data['labels'].apply(lambda x: len(x) > 0)]
            for path, labels in batch_data.groupby('labels_path'):
                labels = labels[['row_number', 'labels']]
                labels['spans'] = labels['labels'].apply(lambda x: [])
                labels = labels.to_dict(orient='records')
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(labels, f)


def generate_judge_data(indir, outdir):
    extract_new_cases.main_run(indir, outdir)
    # jed.run_jed(use_config_file=False, indir=outdir, outdir=outdir)


# adapted from docket-viewer/ml/cli/update_mongo.py
def apply_rules(indir, outfile, preds_dir, judge_dir, reset=False):
    append_mode = (not reset and Path(outfile).exists())
    if append_mode:
        print(f"{outfile.split('/')[-1]} exists and reset=False; skipping any cases listed in that file")
        fnames_to_skip = set([x.replace(';;','-').replace(':','-')+'.json' for x in pd.read_csv(outfile)['ucid']])
    else:
        fnames_to_skip = set()
    fpath_tuples = []
    judge_df_all = pd.read_csv(Path(judge_dir)/'entries.csv')
    judge_dfs, current_ucid, last_index = {}, list(judge_df_all.ucid)[0], 0
    for i in judge_df_all.index:
        if judge_df_all.at[i,'ucid'] != current_ucid:
            judge_dfs[current_ucid] = judge_df_all.iloc[last_index:i]
            current_ucid, last_index = judge_df_all.at[i,'ucid'], i
        elif i==len(judge_df_all)-1:
            judge_dfs[current_ucid] = judge_df_all.iloc[last_index:]
    for fpath in Path(indir).glob('*.json'):
        fname = os.path.basename(fpath)
        ucid = fname.split('.')[0].replace('-',';;',1).replace('-',':',1)
        court,_,year,_,_ = fname.split('-')
        if fname not in fnames_to_skip:
            fpath_tuples.append((fpath, Path(preds_dir)/fname, judge_dfs[ucid]))
    for batch in tqdm(list(partition_all(1000, fpath_tuples))):
        batch_data = []
        for fpath_tuple in batch:
            with open(fpath_tuple[0]) as f0, open(fpath_tuple[1]) as f1:
                case_json, label_json, judge_df = json.load(f0), json.load(f1), fpath_tuple[2]
            docket = docket_functions.Docket.from_json(case_json, label_json=label_json, judge_df=judge_df)
            for entry in docket:
                for label in entry.labels:
                    batch_data.append({'ucid': case_json['ucid'], 'row_ordinal': entry.row_number, 'label': label})
                if entry.event:
                    label = f'{entry.event.name} ({entry.event.event_type})'
                    batch_data.append({'ucid': case_json['ucid'], 'row_ordinal': entry.row_number, 'label': label})
        batch_data = pd.DataFrame(batch_data)
        # # all this line did in update_mongo.py was pull from a nathan-maintained json file, so i feel ok omitting it & leaving versioning to the end user
        # batch_data['model_version'] = batch_data['label'].apply(lambda x: None if x not in version['labels'] else version['labels'][x])
        batch_data['_updated'] = datetime.now()
        batch_data = batch_data.sort_values(['ucid', 'row_ordinal'])
        if not append_mode:
            batch_data.to_csv(outfile, index=False)
            append_mode = True
        else:
            batch_data.to_csv(outfile, index=False, mode='a', header=False)


def generate_scales_labels(indir, outfile, preds_dir=None, judge_dir=None, batch_size=8, reset=False):
    # set up directories
    if not preds_dir:
        preds_dir = indir.rstrip('/').rsplit('/',1)[0] + '/intermediate_predictions'
    if not judge_dir:
        judge_dir = indir.rstrip('/').rsplit('/',1)[0] + '/intermediate_judge_data'
    outdir = ('./'+outfile if outfile[0]!='/' else outfile).rsplit('/',1)[0]
    for path in (preds_dir, judge_dir, outdir):
        if path and path!='.':
            Path(path).mkdir(parents=True, exist_ok=True)
    # run the labeling pipeline
    update_classifier_predictions(indir=indir, outdir=preds_dir, batch_size=batch_size, reset=reset)
    generate_judge_data(indir, judge_dir)
    apply_rules(indir, outfile, preds_dir, judge_dir, reset=reset)



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
