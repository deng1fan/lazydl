#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='lazydl',
    version='0.2.0',
    author='deng1fan',
    author_email='dengyifan@iie.ac.cn',
    url='https://github.com/deng1fan',
    description=u'Deep learning tools',
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'nvitop',
        'redis',
        'transformers',
        'datasets',
        'evaluate',
        'pandas',
        'dingtalkchatbot',
        'psutil',
        'nltk',
        'jsonlines',
        'omegaconf',
        'setproctitle',
        'rich',
        'comet_ml',
        'hydra_colorlog',
        'peft',
        'bitsandbytes',
        'bert_score',
        'rouge',
        'sacrebleu'    
    ],
    exclude=["*.tests", "*.tests.*", "tests"],
    include_package_data=True,
    python_requires='>=3.6',
    keywords=['lazydl', 'deep learning', 'pytorch'],
)
