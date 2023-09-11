from torch import nn
from torchvision import transforms
import torch
import torchvision

from torch_utilities.dataset.cv_dataset import create_dataloaders
from torch_utilities.plot import show_single_image
from torch_utilities.model.vit.ViT import ViT
from torch_utilities.trainer import train
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch_utilities.consts import available_device
from torch_utilities.predict import pred_and_plot_image


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    set_seed()

    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    PATCH_SIZE = 16
    TRAIN_DIR = "dataset/real_and_fake_face/train"
    TEST_DIR = "dataset/real_and_fake_face/test"

    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(
        available_device
    )

    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    pretrained_vit.heads = nn.Linear(in_features=768, out_features=2).to(
        available_device
    )

    manual_transform = pretrained_vit_weights.transforms()

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=manual_transform,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    optimizer = Adam(params=pretrained_vit.parameters(), lr=1e-3)
    loss_func = CrossEntropyLoss()

    pretrained_vit_results = train(
        model=pretrained_vit,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_func=loss_func,
        epochs=5,
        device=available_device,
        writer=None,
    )

    pred_and_plot_image(
        model=pretrained_vit,
        image_path="dataset/hard_210_1100.jpg",
        device=available_device,
        type=torch.float,
        class_names=class_names,
        transform=manual_transform,
    )
