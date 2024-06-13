import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any, Optional

from .utils import seed_everything, update_configs_with_args
from .configs import AnomalyConfigs
from .model import GeneratorWithMemory, Discriminator, GMMWithPrior


class AnomalyModel:
    # List attributes
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, int]
    device: torch.device
    random_state: int
    n_genes: int
    g_configs: Dict[str, Any]
    d_configs: Dict[str, Any]
    gmm_configs: Dict[str, Any]

    def __init__(self, **kwargs):
        configs = AnomalyConfigs()

        # Update configs with kwargs
        for key, value in kwargs.items():
            if hasattr(configs, key):
                setattr(configs, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Config")
        
        configs.build()

        # Initialize the attributes from configs
        for key, value in configs.__dict__.items():
            setattr(self, key, value)

        self._init_model()

        seed_everything(self.random_state)
    
    def _init_model(self):
        self.G = GeneratorWithMemory(**self.g_configs).to(self.device)
        self.D = Discriminator(**self.d_configs).to(self.device) 

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

        self.loss = nn.L1Loss().to(self.device)

    def train(self, ref: ad.AnnData):
        tqdm.write('Begin to train M2ASDA on the reference dataset...')

        self.gene_names = ref.var_names
        train_data = torch.Tensor(ref.X)
        self.loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

        self.G.train()
        self.D.train()
        
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description('Training Epochs')

                for data in self.loader:
                    data = data.to(self.device)

                    for _ in range(self.n_critic):
                        self.UpdateD(data)

                    self.UpdateG(data)
        
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

                self.sch_G.step()
                self.sch_D.step()
        
        tqdm.write('Training has been finished.\n')

    def predict(self, tgt: ad.AnnData, run_gmm: bool = True):
        self.check(tgt)

        tqdm.write('Begin to detect anomalies on the target dataset...')
        real_data = torch.Tensor(tgt.X)
        loader = DataLoader(real_data, batch_size=self.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)
        
        ref_score = self.score(self.loader)
        tgt_score = self.score(loader)

        tqdm.write('Anomalous spots have been detected.\n')

        if run_gmm:
            gmm = GMMWithPrior(ref_score, **self.gmm_configs)
            threshold = gmm.fit(tgt_score=tgt_score)
            tgt_label = [1 if s >= threshold else 0 for s in tgt_score]
            return tgt_score, tgt_label
        else:
            return tgt_score

    def UpdateG(self, data):
        fake_data, z = self.G(data)
        _, fake_z = self.G(fake_data)

        # Discriminator provides feedback
        d = self.D(fake_data)

        self.G_loss = self.loss_weight['alpha'] * self.loss(data, fake_data) \
                      + self.loss_weight['beta'] * self.loss(z, fake_z) \
                      - self.loss_weight['gamma'] * torch.mean(d)
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

        self.G.memory.update_mem(z)

    def UpdateD(self, data):
        fake_data, _ = self.G(data)

        d1 = torch.mean(self.D(data))
        d2 = torch.mean(self.D(fake_data.detach()))
        gp = self.D.gradient_penalty(data, fake_data.detach())
        self.D_loss = - d1 + d2 + gp * self.loss_weight['lambda']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()

    def check(self, tgt: ad.AnnData):
        if (tgt.var_names != self.gene_names).any():
            raise RuntimeError('Target and reference data have different genes.')

        if (self.G is None or self.D is None):
            raise RuntimeError('Please train the model first.')

    @torch.no_grad()
    def score(self, dataset):
        self.D.eval()
        self.G.eval()
        score = []

        for data in dataset:
            data = data.to(self.device)

            fake_data, z = self.G(data)
            _, fake_z = self.G(fake_data)

            s = self.cosine_similarity(z, fake_z)
            score.append(s.cpu().detach())

        score = torch.cat(score, dim=0).numpy()
        return self.normalize(score)

    def cosine_similarity(self, z, fake_z):
        dot_product = torch.sum(z * fake_z, dim=1)
        norm_z = torch.norm(z, dim=1)
        norm_fake_z = torch.norm(fake_z, dim=1)

        cosine_sim = dot_product / (norm_z * norm_fake_z)
        return cosine_sim.reshape(-1, 1)

    def normalize(self, score: np.ndarray):
        score = (score.max() - score)/(score.max() - score.min())
        return score.reshape(-1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M2ASDA for anomaly detection.")
    configs = AnomalyConfigs()

    # Data path arguments
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--ref_path', type=str, help='Path to read the reference h5ad file')
    data_group.add_argument('--tgt_path', type=str, help='Path to read the target h5ad file')
    data_group.add_argument('--result_path', type=str, default='result.csv', help='Path to save the output csv file')
    data_group.add_argument('--pth_path', type=Optional[str], default=None, help='Path to save the trained generator')

    # AnomalyModel arguments with defaults from AnomalyConfigs
    a_group = parser.add_argument_group('AnomalyModel Parameters')
    a_group.add_argument('--n_epochs', type=int, default=configs.n_epochs, help='Number of epochs')
    a_group.add_argument('--batch_size', type=int, default=configs.batch_size, help='Batch size')
    a_group.add_argument('--learning_rate', type=float, default=configs.learning_rate, help='Learning rate')
    a_group.add_argument('--n_critic', type=int, default=configs.n_critic, help='Number of discriminator iterations per generator iteration')
    a_group.add_argument('--alpha', type=int, default=configs.alpha, help='Loss weight alpha')
    a_group.add_argument('--beta', type=int, default=configs.beta, help='Loss weight beta') 
    a_group.add_argument('--gamma', type=int, default=configs.gamma, help='Loss weight gamma') 
    a_group.add_argument('--lambda', type=int, default=configs.lamb, help='Loss weight lambda') 
    a_group.add_argument('--GPU', type=str, default=configs.GPU, help='GPU ID for training, e.g., cuda:0')
    a_group.add_argument('--random_state', type=int, default=configs.random_state, help='Random seed')
    a_group.add_argument('--n_genes', type=int, default=configs.n_genes, help='Number of genes')
    a_group.add_argument('--run_gmm', type=bool, default=True, help='Run GMM for obtaining binary label')

    args = parser.parse_args()

    # Update the configs with command line argument
    args_dict = vars(args)
    update_configs_with_args(configs, args_dict, None)

    configs.build()
    configs.clear()

    # Print out all configurations to verify they are complete
    print("=============== AnomalyModel Parameters ===============")
    for key, value in configs.__dict__.items():
        print(f"{key} = {value}")

    # Read the preprocessed data
    ref = sc.read_h5ad(args_dict['ref_path'])
    tgt = sc.read_h5ad(args_dict['tgt_path'])

    # Initialize and train AnomalyModel
    model = AnomalyModel(**configs.__dict__)
    model.train(ref)

    if args_dict['run_gmm']:
        score, label = model.predict(tgt, True)
        df = pd.DataFrame({'score': score, 'label': label}, index=tgt.obs_names)
    else:
        score = model.predict(tgt, False)
        df = pd.DataFrame({'score': score}, index=tgt.obs_names)
    
    df.to_csv(args_dict['save_path'])

    if args_dict['pth_path'] is not None:
        torch.save(model.G, args_dict['pth_path'])