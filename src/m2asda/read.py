import numpy as np
import scanpy as sc
import anndata as ad
from math import e
from scipy.sparse import issparse
from typing import Sequence, Optional, Union

from .utils import clear_warnings


@clear_warnings(category=UserWarning)
def preprocess_data(adata: ad.AnnData, gene_list=None):
    # clear the obs &var names
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if gene_list is None:
        adata = filter_gene(adata)
    else:
        adata = adata[:, gene_list]

    # normalization
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata


def filter_gene(adata: ad.AnnData):
    drop_pattern1 = adata.var_names.str.startswith('ERCC')
    drop_pattern2 = adata.var_names.str.startswith('MT-')
    drop_pattern = np.logical_and(~drop_pattern1, ~drop_pattern2)
    adata._inplace_subset_var(drop_pattern)
    sc.pp.filter_genes(adata, min_cells=3)
    return adata


def read_dataset(dir, names):
    def read_single(data_dir: str, data_name: str):
        if not data_name.endswith('.h5ad'):
            data_name += '.h5ad'

        input_dir = data_dir + data_name
        adata = sc.read_h5ad(input_dir)

        if issparse(adata.X):
            adata.X = adata.X.toarray()

        return adata

    if isinstance(names, str):
        adata = read_single(dir, names)
    else:
        adatas = []
        for name in names:
            adatas.append(read_single(dir, name, False))
        adata = ad.concat(adatas)
    return adata


@clear_warnings()
def read(ref_dir: str, ref_name: Union[Sequence[str], str],
         tgt_dir: Optional[str] = None, tgt_name: Optional[Union[Sequence[str], str]] = None,
         preprocess: bool = True):
    if tgt_dir is None:
        tgt_dir = ref_dir

    ref = read_dataset(ref_dir, ref_name)
    tgt = read_dataset(tgt_dir, tgt_name)

    if preprocess:
        ref = preprocess_data(ref)
        tgt = preprocess_data(tgt, ref.var_names)

    return ref, tgt