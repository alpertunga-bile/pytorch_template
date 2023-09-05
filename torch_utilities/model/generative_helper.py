from torch import nn, device, randn, Tensor, ones_like
from torch.autograd import grad


def get_generator_block(
    in_dim: int, out_dim: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
    )


def get_critic_block(
    in_dim: int, out_dim: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_dim),
        nn.LeakyReLU(0.2),
    )


def gen_noise(num: int, z_dim: int, device: device) -> Tensor:
    return randn(num, z_dim, device=device)


def get_gp(
    real: Tensor, fake: Tensor, crit: nn.Module, alpha: int, gamma: int = 10
) -> Tensor:
    mix_images = real * alpha + fake * (1 - alpha)
    mix_scores = crit(mix_images)

    gradient = grad(
        inputs=mix_images,
        outputs=mix_scores,
        grad_outputs=ones_like(mix_scores),
        retain_graph=True,
        create_graph=True,
    )[0]

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = gamma * ((gradient_norm - 1) ** 2).mean()

    return gp
