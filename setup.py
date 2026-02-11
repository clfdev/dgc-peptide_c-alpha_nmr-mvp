# setup.py
from setuptools import setup, find_packages

setup(
    name="dgc_nmr_mvp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'requests>=2.28.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'biopython>=1.79',
        'pytest>=7.0.0',
    ],
    python_requires='>=3.8',
)