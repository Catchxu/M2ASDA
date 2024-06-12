import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from .layer import Encoder, LinearBlock


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=[1024, 512, 256], latent_dim=256, **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, **kwargs)
        self.critic = nn.Sequential(
            LinearBlock(latent_dim, 256),
            LinearBlock(256, 64),
            nn.Linear(64, 1)      
        )

        # Additional initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def gradient_penalty(self, real_data, fake_data):
        shapes = [1 if i != 0 else real_data.size(i) for i in range(real_data.dim())]
        cuda = True if torch.cuda.is_available() else False

        eta = torch.FloatTensor(*shapes).uniform_(0, 1)
        eta = eta.cuda() if cuda else eta
        interpolated = eta * real_data + ((1 - eta) * fake_data)
        interpolated = interpolated.cuda() if cuda else interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)

        # calculate gradients of probabilities with respect to examples
        grad = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                             grad_outputs=torch.ones(prob_interpolated.size()).cuda()
                             if cuda else torch.ones(prob_interpolated.size()),
                             create_graph=True, retain_graph=True)[0]

        grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x):
        return self.critic(self.encoder(x))
