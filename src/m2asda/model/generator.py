import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from .layer import Encoder, Decoder, MemoryBlock, StyleBlock


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=[1024, 512, 256], latent_dim=256, **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, **kwargs)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, **kwargs)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class GeneratorWithMemory(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: Union[int, List[int]],
                 latent_dim: int,
                 memory_size: int = 512, 
                 threshold: float = 0.005, 
                 temperature: float = 0.1,
                 **kwargs):
        super().__init__()
        self.extractor = AutoEncoder(input_dim, hidden_dim, latent_dim, **kwargs)
        self.memory = MemoryBlock(
            self.extractor.latent_dim, memory_size, threshold, temperature
        )

        # Additional initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        z = self.extractor.encoder(x)
        memory_z = self.memory(z)
        x = self.extractor.decoder(memory_z)
        return x, z


class GeneratorWithPairs(nn.Module):
    def __init__(self, generator: GeneratorWithMemory, n_ref: int, n_tgt: int):
        super().__init__()
        self.G = generator

        # Freeze the parameters in trained generator
        for param in self.G.parameters():
            param.requires_grad = False
        
        self.P = nn.Parameter(torch.Tensor(n_tgt, n_ref))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.P.size(1))
        self.P.data.uniform_(-stdv, stdv)

    def forward(self, x_ref, x_tgt):
        z_ref = self.G(x_ref)
        z_tgt = self.G(x_tgt)
        fake_z_tgt = torch.mm(F.relu(self.P), z_ref)
        return fake_z_tgt, z_tgt, F.relu(self.P).detach().cpu().numpy()


class GeneratorWithStyle(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: Union[int, List[int]],
                 latent_dim: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()
        self.extractor = AutoEncoder(input_dim, hidden_dim, latent_dim, **kwargs)
        self.style = StyleBlock(num_batches, self.extractor.latent_dim)        

        # Additional initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        z = self.extractor.encoder(x)
        style_z = self.style(z)
        x = self.extractor.decoder(style_z) 
        return x       