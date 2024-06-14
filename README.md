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