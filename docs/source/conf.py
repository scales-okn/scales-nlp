# pylint: skip-file
import datetime

project = 'SCALES-NLP'
copyright = str(datetime.datetime.now().year)+', SCALES-OKN'
author = 'Nathan Dahlberg'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx_click',
    'myst_nb',
    'sphinx.ext.autosectionlabel',
]

autodoc_member_order = 'bysource'

nb_execution_mode = "off"

source_suffix = ['.md']

templates_path = ['_templates']
exclude_patterns = ['build']

html_theme = "press"
html_static_path = ['_static']

html_logo = "logo.png"

html_context = {
    'display_github': True,
    'github_user': 'scales-okn',
    'github_repo': 'scales-nlp',
    'github_version': 'master/',
    'conf_py_path': '/docs/',
    'show_sphinx': False,
}

