#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import setup, find_packages

# Package meta-data
NAME = 'geneinsight'
DESCRIPTION = 'Topic modeling pipeline for gene sets with enrichment analysis'
URL = 'https://github.com/yourusername/geneinsight'
EMAIL = 'your.email@example.com'
AUTHOR = 'Your Name'
REQUIRES_PYTHON = '>=3.9,<3.13'
VERSION = '0.1.0'

# Required packages
REQUIRED = [
    'pandas>=2.0.0',
    'bertopic>=0.15.0',
    'scikit-learn>=1.3.0',
    'sentence-transformers>=2.2.2',
    'openai>=1.1.0',
    'instructor>=0.3.0',
    'tqdm>=4.66.0',
    'python-dotenv>=1.0.0',
    'gseapy>=1.1.0',
    'joblib>=1.3.0',
    'optuna>=3.3.0',
    'stringdb>=0.1.3',
    'pydantic>=2.4.0',
    'rich>=13.6.0',
    'typer>=0.9.0',
    'pyyaml>=6.0',
    'seaborn>=0.12.0',  # Added seaborn dependency
    'colorcet>=1.0.0',  # Added colorcet dependency
    'torch>=1.7.0',  # Needed for sentence-transformers in ontology module
]

# Optional packages
EXTRAS = {
    'dev': [
        'pytest>=7.3.1',
        'black>=23.3.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
    ],
}

# The rest of the setup code
here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
    VERSION = about['__version__']
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    entry_points={
        'console_scripts': [
            'geneinsight=geneinsight.cli:main',
            'geneinsight-enrichment=geneinsight.scripts.geneinsight_enrichment:main',
            'geneinsight-topic=geneinsight.scripts.geneinsight_topic_modeling:main',
            'geneinsight-prompt=geneinsight.scripts.geneinsight_prompt_gen:main',
            'geneinsight-api=geneinsight.scripts.geneinsight_api_process:main',
            'geneinsight-hypergeometric=geneinsight.scripts.geneinsight_hypergeometric:main',
            'geneinsight-filter=geneinsight.scripts.geneinsight_filter:main',
            'geneinsight-meta=geneinsight.scripts.geneinsight_meta:main',
            'geneinsight-counter=geneinsight.scripts.geneinsight_counter:main',
            'geneinsight-report=geneinsight.scripts.geneinsight_report_pipeline:main',
            'geneinsight-ontology=geneinsight.ontology.workflow:main'
        ],
    },

    package_data={
        'topicgenes.report': ['assets/*.png'],
        'geneinsight.ontology.ontology_folders': ['*.txt'],  # Include all .txt files in ontology_folders
    }
)