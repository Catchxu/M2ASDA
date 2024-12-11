import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 normalization: bool = True, 
                 activation: bool = True, 
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        
        # Linear layer
        layers.append(nn.Linear(in_dim, out_dim))

        # Batch Normalization layer
        if normalization:
            layers.append(nn.BatchNorm1d(out_dim))

        # Activation function
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Dropout layer
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, **kwargs):
        super().__init__()
        layers = []

        dims = [input_dim] + hidden_dim
        dim1 = dims[0]
        for dim2 in dims[1:]:
            layers.append(LinearBlock(dim1, dim2, **kwargs))
            dim1 = dim2
        
        layers.append(nn.Linear(dim1, latent_dim))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, **kwargs):
        super().__init__()
        layers = []

        dims = hidden_dim + [latent_dim]
        dim1 = dims[-1]
        for dim2 in reversed(dims[:-1]):
            layers.append(LinearBlock(dim1, dim2, **kwargs))
            dim1 = dim2

        layers.append(nn.Linear(dim1, input_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MemoryBlock(nn.Module):
    def __init__(self, latent_dim, memory_size, threshold, temperature):
        super().__init__()
        self.mem_dim = memory_size
        self.z_dim = latent_dim
        self.shrink_thres = threshold
        self.temperature = temperature
        self.mem = torch.randn(self.mem_dim, self.z_dim)
        self.mem_ptr = torch.zeros(1, dtype=torch.long)

        self._init_parameters()

    def _init_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def hard_shrink_relu(self, x, lambd=0, epsilon=1e-12):
        x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
        return x

    def forward(self, x):
        self.mem = self.mem.to(x.device)
        att_weight = torch.mm(x, self.mem.T)
        att_weight = F.softmax(att_weight/self.temperature, dim=1)

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)
        return output


class StyleBlock(nn.Module):
    def __init__(self, num_batches: int, latent_dim: int):
        super().__init__()
        self.n = num_batches
        self.style = nn.Parameter(torch.Tensor(num_batches, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        stdv = 1. / math.sqrt(self.style.size(1))
        self.style.data.uniform_(-stdv, stdv)

    def forward(self, z, batchid):
        if self.n == 1:
            return z - self.style
        else:
            s = torch.mm(batchid, self.style)
            return z - s


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        assert d_model % nheads == 0

        self.d_k = d_model // nheads
        self.h = nheads
        self.dropout = nn.Dropout(dropout)

        # Produce N identical layers
        self.linears = nn.ModuleList([
            copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)
        ])

    def attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim = -1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v):
        N = q.shape[0]

        q, k, v = [l(x) for l, x in zip(self.linears[:-1], (q, k, v))] # (batch_size, d_model)
        q, k, v = [x.view(N, -1, self.h, self.d_k).transpose(1, 2) for x in (q, k, v)] # (batch_size, h, 1, d_k)

        x, self.attn = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(N, self.h*self.d_k)

        return self.linears[-1](x).squeeze(1)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nheads, hidden_dim=512, dropout=0.3) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, nheads)

        self.norm = nn.ModuleList([
            copy.deepcopy(nn.LayerNorm(d_model)) for _ in range(2)
        ])

        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        attn = self.attention(q, k, v)

        x = self.dropout(self.norm[0](attn + q))
        f = self.fc(x)
        x = self.dropout(self.norm[1](x + f))
        return x


class TFBlock(nn.Module):
    def __init__(self, latent_dim, num_layers=3, nheads=4, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(latent_dim, nheads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, res_z):
        z = self.dropout(z)
        res_z = self.dropout(res_z)

        for layer in self.layers:
            z = layer(z, res_z, res_z)
 
        return z