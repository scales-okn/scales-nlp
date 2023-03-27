from glob import glob
from setuptools import setup, find_packages


__version__ = None
exec(open('src/scales_nlp/version.py').read())

 
setup(
	name='scales-nlp',
	version=__version__,   
	description='',
	url='https://github.com/scales-okn/scales-nlp',
	author='Nathan Dahlberg',
	package_dir={'': 'src'},
	packages=find_packages('src'),
	install_requires=[
		'configuration-maker',
		'datasets',
		'evaluate',
		'numpy<1.23.0',
		'pacer-tools',
		'pandas',
		'pathlib',
		'protobuf<3.21.0',
		'sentencepiece',
		'scikit-learn',
		'toolz',
		'tqdm',
		'transformers',
	],
	
	data_files=[
        ('scales_nlp', glob('src/scales_nlp/data/*')),
    ],
    include_package_data = True,

	entry_points={
		'console_scripts': [
			'scales-nlp = scales_nlp:cli',
		],
	},
)
