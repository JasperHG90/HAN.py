from setuptools import setup, find_packages

setup(
    name="HAN",
    version="0.1",
    description="Train a Hierarchical Attention Model (HAN) on textual data.",
    url='https://github.com/FinTxt/HAN.py',
    author='Jasper Ginn',
    author_email='jasperginn@gmail.com',
    license='GPL-4',
    packages=find_packages(exclude="__pycache__"),
    install_requires=[
        "torch>=1.3.1",
        "spacy>=2.2.0,<3.0.0",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz",
        "numpy"
    ],
    zip_safe=False,
    python_requires='>=3.6'
)
