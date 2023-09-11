from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple
from os import makedirs
from os.path import join


def show_grid(
    tensor: Tensor, channel: int, size: Tuple[int, int], one_row_size: int = 4
):
    total_grid_size = one_row_size * one_row_size
    data = tensor.detach().cpu().view(-1, channel, *size)
    grid = make_grid(data[:total_grid_size], nrow=one_row_size).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()


def save_grid(
    tensor: Tensor, parent_folder: str, one_row_size: int = 4, filename: str = "test"
):
    total_grid_size = one_row_size * one_row_size

    data = tensor.detach().cpu()
    grid = make_grid(data[:total_grid_size], nrow=one_row_size)

    makedirs(parent_folder, exist_ok=True)
    filepath = join(parent_folder, filename + ".png")

    image = to_pil_image(grid, "RGB")
    image.save(filepath)


def plot_gan_model_losses(gen_losses: list, crit_losses: list) -> None:
    steps = range(len(gen_losses))

    plt.figure(figsize=(15, 7))

    plt.plot(steps, gen_losses, label="gen_loss")
    plt.plot(steps, crit_losses, label="crit_loss")
    plt.title("Loss")
    plt.xlabel("Steps")
    plt.legend()

    plt.show()


def show_single_image(img: Tensor, label: str) -> None:
    plt.imshow(img.permute(1, 2, 0))
    plt.title(label)
    plt.axis(False)
    plt.show()
