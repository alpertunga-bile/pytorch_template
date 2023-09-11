from torch import nn
from torch_utilities.model.generative_helper import (
    get_generator_block,
    get_critic_block,
)


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64, d_dim: int = 16):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.gen = nn.Sequential(
            get_generator_block(z_dim, d_dim * 32, 4, 1, 0),
            get_generator_block(d_dim * 32, d_dim * 16),
            get_generator_block(d_dim * 16, d_dim * 8),
            get_generator_block(d_dim * 8, d_dim * 4),
            get_generator_block(d_dim * 4, d_dim * 2),
            nn.ConvTranspose2d(d_dim * 2, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


class Critic(nn.Module):
    def __init__(self, d_dim=16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.crit = nn.Sequential(
            get_critic_block(3, d_dim),
            get_critic_block(d_dim, d_dim * 2),
            get_critic_block(d_dim * 2, d_dim * 4),
            get_critic_block(d_dim * 4, d_dim * 8),
            get_critic_block(d_dim * 8, d_dim * 16),
            nn.Conv2d(d_dim * 16, 1, 1, 1, 0),
        )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
