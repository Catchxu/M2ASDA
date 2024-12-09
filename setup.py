from setuptools import find_packages
from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='m2asda',
    version='1.0.0',
    description='Detecting and subtyping anomalous single cells with M2ASDA',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Kaichen Xu',
    author_email='Kaichenxu@stu.zuel.edu.cn',
    url='https://github.com/Catchxu/M2ASDA',
    license='GPL v3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
    install_requires=[
        'anndata>=0.10.7',
        'numpy>=1.22.4',
        'pandas>=1.5.1',
        'scanpy>=1.10.1',
        'scikit-learn>=1.2.0',
        'scipy>=1.11.4',
        'torch>=2.0.0',
        'tqdm>=4.64.1'
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    zip_safe=False,
)