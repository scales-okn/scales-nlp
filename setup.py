from glob import glob
from setuptools import setup, find_packages


__version__ = '0.1.0'
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
            'cchardet==2.2.0a2',
            'configuration-maker',
            'datasets',
            'evaluate',
            'numpy',
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
