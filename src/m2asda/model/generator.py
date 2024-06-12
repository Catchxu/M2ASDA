import torch.nn as nn
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