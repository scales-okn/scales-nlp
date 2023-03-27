import os
from pathlib import Path
import shutil
import click
import pandas as pd
import scales_nlp
from scales_nlp import config


@click.command()
@click.option('--push/--no-push', default=False, help='Push project to PyPI')
def build(push):
    """
    Build and publish the project to PyPI.
    """
    repo_dir = Path(scales_nlp.__file__).parents[2]
    os.system(f'python {repo_dir}/setup.py bdist_wheel --universal')
    if push:
        os.system('twine upload dist/*')


@click.command()
@click.option('--build/--no-build', default=False, help='Rebuild the docs')
def docs(build):
    """
    Start a local server and optionally build the docs with `--build`.
    """
    docs_dir = Path(scales_nlp.__file__).parents[2] / 'docs'
    build_dir = docs_dir / 'build' / 'html'
    if build:
        if build_dir.exists():
            shutil.rmtree(docs_dir / 'build')
        os.system(f'(cd {docs_dir} && make html)')
    os.system(f'(cd {build_dir} && python -m http.server 8003)')


@click.command()
@click.argument('group', default='main', required=False)
@click.option('--reset/--no-reset', default=False, help='Delete existing group keys in config')
def configure(group, reset):
    """Update configuration keys."""

    config.update(group, reset)


@click.command()
@click.argument('ucid')
def download(ucid):
    """Download a case from PACER. This will bill your PACER account.  
    
    Provide the UCID of the case.  The ucid consists of {court};;{docket_number}
    
    Example UCID: ilnd;;1:21-cv-04600
    """
    scales_nlp.utils.crawl_pacer(ucid)


@click.command()
@click.argument('court', default=None, required=False)
def parse(court):
    """Parse downloaded PACER data in the PACER_DIR into JSON format.  Leave `court` blank to parse all courts."""
    scales_nlp.utils.parse_pacer_dir(court)


@click.command()
@click.option('--batch-size', default=8, help='Batch size for model predictons')
@click.option('--reset/-no-reset', default=False, help='Overwrite existing predictions')
def update_labels(batch_size, reset):
    """Apply docket classification model to PACER data in the PACER_DIR."""
    scales_nlp.utils.update_classifier_predictions(batch_size, reset)


@click.command()
@click.argument('data-path')
@click.argument('output-dir')
@click.argument('task')
@click.option('--model-name', default=config['MODEL_NAME'], help='Name of model to finetune')
@click.option('--loss', default=None, help='Name of loss to use for training')
@click.option('--metric', default=None, help='Name of metric to use for evaluation')
@click.option('--max-length', default=config['MAX_LENGTH'], help='Truncate inputs to max token sequence length')
@click.option('--eval-split', default=config['EVAL_SPLIT'], help='Proportion of data to use for evaluation')
@click.option('--epochs', default=config['EPOCHS'], help='Number of training epochs')
@click.option('--train-batch-size', default=config['TRAIN_BATCH_SIZE'], help='Train batch size')
@click.option('--eval-batch-size', default=config['EVAL_BATCH_SIZE'], help='Evaluation batch size')
@click.option('--gradient-accumulation-steps', default=config['GRADIENT_ACCUMULATION_STEPS'], help='Artificially increase the train batch size')
@click.option('--learning-rate', default=config['LEARNING_RATE'], help='Learning rate')
@click.option('--warmup-ratio', default=config['WARMUP_RATIO'], help='Learning rate warmup')
@click.option('--weight-decay', default=config['WEIGHT_DECAY'], help='Weight decay for AdamW')
@click.option('--save-steps', default=config['SAVE_STEPS'], help='Save model checkpoint every n steps')
@click.option('--push', default=None, help='model id to push to hub')
@click.option('--overwrite/--no-overwrite', default=False, help='Overwrite output dir if it exists')
@click.option('--multi-label-delimiter', default='|', help='Delimiter for splitting labels in multi-label-classification task')
@click.option('--text-col', default='text', help='The column with text')
@click.option('--label-col', default='label', help='The column with labels')
def train(
        data_path, output_dir, task, model_name, 
        loss, metric, max_length, eval_split, epochs, 
        train_batch_size, eval_batch_size, gradient_accumulation_steps,
        learning_rate, warmup_ratio, weight_decay,
        save_steps, push, overwrite, 
        multi_label_delimiter, text_col, label_col
    ):
    """
    Run a training routine from a csv file.

    Specify one of the following tasks:
    \n\tclassification
    \n\tmulti-label-classification
    \n\ttoken-classification (ner)
    """
    
    data = pd.read_csv(data_path)
    texts = list(data[text_col].values)
    labels = list(data[label_col].astype(str).values)
    if task == 'multi-label-classification':
        labels = [[x for x in labels.split(multi_label_delimiter) if x != ''] for labels in labels]
    
    routine = scales_nlp.training_routine(task,
        model_name=model_name, loss=loss, metric=metric, 
        max_length=max_length, eval_split=eval_split, epochs=epochs,
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate,
        warmup_ratio=warmup_ratio, weight_decay=weight_decay,
        save_steps=save_steps,
    )

    routine.train(output_dir, texts, labels, push=push, overwrite=overwrite)


@click.group()
def main():
    """SCALES-NLP: An AI Toolkit for Legal Research"""
    pass


main.add_command(configure)
main.add_command(download)
main.add_command(parse)
main.add_command(update_labels)
main.add_command(train)


if config['DEVELOPER_MODE']:
    main.add_command(build)
    main.add_command(docs)


if __name__=='__main__':
    main()