import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans

from .generator import GeneratorWithMemory
from .layer import TFBlock


class Subtyper(nn.Module):
    def __init__(self, 
                 generator: GeneratorWithMemory,
                 num_types: int,
                 alpha: float = 1,
                 KMeans_n_init: int = 20,
                 num_layers: int = 3,
                 nheads: int = 4,
                 hidden_dim: int = 512,
                 dropout: float = 0.1
                 ):
        super().__init__()

        self.G = generator
        self.z_dim = self.G.extractor.latent_dim
        self.num_types = num_types
        self.alpha = alpha
        self.KMeans_n_init = KMeans_n_init

        self.fusion = TFBlock(self.z_dim, num_layers, nheads, hidden_dim, dropout)
        self.mu = Parameter(torch.Tensor(self.num_types, self.z_dim))

        # classifer for supervised pre-training
        self.classifer = nn.Linear(self.z_dim, num_types)

        # Additional initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, z, res):
        res_z = self.G.extractor.encoder(res)
        z = self.fusion(z, res_z)
        q = 1.0/((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2)/self.alpha)+1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        self.mu_update(z, q)
        return z, q

    def pretrain(self, z, res):
        res_z = self.G.extractor.encoder(res)
        z = self.fusion(z, res_z)
        return self.classifer(z)

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def mu_init(self, feat):
        kmeans = KMeans(self.num_types, n_init=self.KMeans_n_init)
        y_pred = kmeans.fit_predict(feat)
        feat = pd.DataFrame(feat, index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def mu_update(self, feat, q):
        y_pred = torch.argmax(q, axis=1).cpu().numpy()
        feat = pd.DataFrame(feat.cpu().detach().numpy(), index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name='Group')
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby('Group').mean())

        self.mu.data.copy_(torch.Tensor(centroid))