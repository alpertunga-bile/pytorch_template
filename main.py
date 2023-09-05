from torch import nn
from torch.optim import Adam
from os.path import join

from torch_utilities.consts import available_device, default_learning_rate
from torch_utilities.model.generative_helper import (
    get_generator_block,
    get_critic_block,
)
from torch_utilities.dataset.gan_dataset import create_dataloader
from torch_utilities.trainer import train_gan
from torch_utilities.plot import plot_gan_model_losses

z_dim = 256
epochs = 5
batch_size = 32
crit_cycles = 10
save_step = 500
show_step = 500
lr = default_learning_rate
device = available_device


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64, d_dim: int = 16):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.gen = nn.Sequential(
            get_generator_block(z_dim, d_dim * 32, 4, 1, 0),
            get_generator_block(d_dim * 32, d_dim * 16),
            get_generator_block(d_dim * 16, d_dim * 8),
            get_generator_block(d_dim * 8, d_dim * 4),
            nn.ConvTranspose2d(d_dim * 4, 3, 4, 2, 1),
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


if __name__ == "__main__":
    data_path = join("dataset", "anime_faces")
    dataloader = create_dataloader(data_path, (64, 64), batch_size)

    gen = Generator(z_dim=z_dim).to(device)
    crit = Critic().to(device)

    gen_opt = Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    crit_opt = Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))

    gen_losses, crit_losses = train_gan(
        epochs,
        crit_cycles,
        z_dim,
        dataloader,
        gen,
        crit,
        gen_opt,
        crit_opt,
        device,
        "outputs",
        "anime_face",
        save_step,
        show_step,
    )

    plot_gan_model_losses(gen_losses, crit_losses)
