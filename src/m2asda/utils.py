import warnings
import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Union, Dict, Any
from torch.utils.data import Dataset


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clear_warnings(category=FutureWarning):
    def outwrapper(func):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=category)
                return func(*args, **kwargs)

        return wrapper

    return outwrapper


def select_device(GPU: Union[bool, str] = True):
    if GPU:
        if torch.cuda.is_available():
            if isinstance(GPU, str):
                device = torch.device(GPU)
            else:
                device = torch.device('cuda:0')
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def update_configs_with_args(configs, args_dict: Dict[str, Any], suffix):
    for key, value in args_dict.items():
        if suffix is not None:
            if key.endswith(suffix):
                # Remove the suffix
                config_key = key[:-len(suffix)]
                # Only update if the argument is provided and valid
                if hasattr(configs, config_key) and value is not None:
                    setattr(configs, config_key, value)


@clear_warnings()
def evaluate(y_true, y_score):
    """
    Calculate evaluation metrics
    """
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)

    roc_auc = metrics.roc_auc_score(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)

    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thres = np.percentile(y_score, ratio)
    y_pred = (y_score >= thres).astype(int)
    y_true = y_true.astype(int)
    _, _, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f1


class PairDataset(Dataset):
    def __init__(self, DataA, DataB):
        self.DataA = DataA
        self.DataB = DataB

        if len(self.DataA) != len(self.DataB):
            raise RuntimeError('Input data can not be paired')

    def __len__(self):
        return len(self.DataA)

    def __getitem__(self, index):
        A_sample = self.DataA[index]
        B_sample = self.DataB[index]

        return A_sample, B_sample