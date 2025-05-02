from glob import glob
from setuptools import setup, find_packages

__version__ = None
exec(open('tmp/scales_nlp/version.py').read())

 
setup(
	name='scales-nlp',
	version=__version__,
	description='',
	url='https://github.com/scales-okn/scales-nlp',
	author='Nathan Dahlberg',
	packages=['scales_nlp', 'disambiguation_scripts', 'scales_nlp_support'],
	package_dir={
		'': 'tmp',
		'disambiguation_scripts': 'tmp/scales_nlp/research_materials/code/research/judge_linking/public_scripts/disambiguation_scripts',
		'scales_nlp_support': 'tmp/scales_nlp/research_materials/code/support'
	},
	install_requires=[
            'cchardet==2.2.0a2',
            'configuration-maker',
            'datasets',
            'evaluate',
            'flashtext',
            'numpy',
            'pacer-tools',
            'pandas',
            'pathlib',
            'protobuf<3.21.0',
            'sentencepiece',
            'scikit-learn',
            'spacy',
            'toolz',
            'tqdm',
            'transformers'
	],
	
	data_files=[
        ('scales_nlp', glob('tmp/scales_nlp/data/*')),
        ('scales_nlp_support', glob('tmp/scales_nlp/research_materials/code/support/core_data/*'))
    ],
    include_package_data = True,

	entry_points={
		'console_scripts': [
			'scales-nlp = scales_nlp:cli',
		],
	},
)
