# Detecting and subtyping anomalous single cells with M2ASDA

Detecting and identifying anomalous single cells from single-cell datasets is crucial for understanding molecular heterogeneity in diseases and promoting precision medicine. No existing method unifies multimodal and multi-sample anomaly detection and identification, involving crucial tasks like anomaly detection, alignment, and annotation. We propose an innovative Generative Adversarial Network-based framework named Multimodal and Multi-sample Anomalous Single-cell Detection and Annotation (M2ASDA), integrating solutions of these crucial tasks into a unified framework. Comprehensive tests on real datasets demonstrate M2ASDA's superior performance in anomaly detection, multi-sample alignment, and identifying common and specific cell types across multiple target datasets.

<br/>
<div align=center>
<img src="/docs/images/framework.png" width="70%">
</div>
<br/>


## Dependencies
- anndata>=0.10.7
- numpy>=1.22.4
- pandas>=1.5.1
- scanpy>=1.10.1
- scikit_learn>=1.2.0
- scipy>=1.11.4
- torch>=2.0.0
- tqdm>=4.64.1


## Installation
M2ASDA is developed as a Python package. You will need to install Python, and the recommended version is Python 3.9.

You can download the package from GitHub and install it locally:

```commandline
git clone https://github.com/Catchxu/M2ASDA.git
cd M2ASDA/
python3 setup.py install
```


## Getting Started
M2ASDA offers a variety of functions for single-cell omics data analysis, and all these functions can be implemented through both python package and terminal commands. Here, we provide the detailed tutorials as follows:
- Detecting anomalous cells with M2ASDA package ([tutorial](https://catchxu.github.io/M2ASDA/tutorial/Anomaly/))
- Running M2ASDA to detect anomalous cells in terminal ([tutorial](https://catchxu.github.io/M2ASDA/tutorial/Anomaly_T/))

Before starting the tutorial, we need to make some preparations, including: installing M2ASDA and its required Python packages, downloading the datasets required for the tutorial, and so on. The preparations is available at [M2ASDA Preparations](https://catchxu.github.io/M2ASDA/start/). Additionally, we strongly recommend using a GPU and pretraining M2ASDA on the public single-cell datasets. More useful and helpful information can be found at the [online documentation](https://Catchxu.github.io/M2ASDA/).


## Datasets
All experimental datasets involved in this paper are available from their respective original sources. The 10x scRNA-seq datasets of healthy human lung tissues (10xG-hHL) and human lung cancer tissues (10xG-hLC-A and -B) are available at [GSE196303](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196303). The 10x scRNA-seq dataset of mouse embryo (10xG-mEmb) is available at [GSE186069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186069). The 10x scRNA-seq datasets of healthy human peripheral blood mononuclear cells (10xG-hHPBMC), and 10x scATAC-seq datasets of healthy and basal cell carcinoma human peripheral blood mononuclear cells (10xC-hHPBMC, and 10xC-hPBMCBCC) are available at [10x Genomics](https://www.10xgenomics.com/datasets). The 10x scRNA-seq datasets of human systemic lupus erythematosus peripheral blood mononuclear cells (10xG-hPBMCSLE-A, -B, and -C) are available at [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583). The Slide-seqV2 datasets of mouse embryo (ssq-mEmb-33) are available at [GSE197353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197353).


## Tested environment
### Environment 1
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- Memory: 256 GB
- System: Ubuntu 20.04.5 LTS
- Python: 3.9.15

### Environment 2
- CPU: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
- Memory: 256 GB
- System: Ubuntu 22.04.3 LTS
- Python: 3.9.18


## Getting help
For any questions or comments, please use the [GitHub issues](https://github.com/Catchxu/M2ASDA/issues) or directly contact Kaichen Xu at the email: kaichenxu358@gmail.com.


## Citation
Coming soon.