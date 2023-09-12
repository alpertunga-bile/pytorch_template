from torch import nn
import torch
from torchvision import transforms
from PIL import Image

from torch_utilities.dataset.image_dataset import create_dataloader
from torch_utilities.plot import show_single_image
from torch.optim import SGD
from torch.nn import MSELoss
from torch_utilities.consts import available_device
from torch_utilities.model.info import print_model_info
from torch_utilities.model.save_load import save_model

from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch import autocast

# pretrained ViT architecture example
"""
? TODO -> Look Pytorch Quantiziation

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
"""


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_map_block(s: int):
    return nn.Sequential(nn.Conv2d(s, s, 3, 1, padding="same"), nn.PReLU(s))


class FSRCNN(nn.Module):
    def __init__(self, d: int, s: int, m: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, d, 5, 1, padding="same"), nn.PReLU(56)
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, 1, 1, padding="same"), nn.PReLU(12)
        )

        self.mapping = nn.Sequential(*[get_map_block(s) for _ in range(m)])

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, 1, 1, padding="same"),
            nn.PReLU(56),
        )

        self.upscale = nn.ConvTranspose2d(d, 1, 9, 2, 4, 1, 1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.upscale(x)

        return x


if __name__ == "__main__":
    set_seed()

    D_VALUE = 56
    S_VALUE = 12
    M_VALUE = 4
    EPOCHS = 5
    HALF_IMAGE_SIZE = 300

    model = FSRCNN(D_VALUE, S_VALUE, M_VALUE).to(available_device)

    """

    train_dataloader = create_dataloader(
        image_folder="dataset/real_and_fake_face/train/real",
        batch_size=32,
        wanted_image_size=(HALF_IMAGE_SIZE, HALF_IMAGE_SIZE),
    )

    test_dataloader = create_dataloader(
        image_folder="dataset/real_and_fake_face/test/real",
        batch_size=32,
        wanted_image_size=(HALF_IMAGE_SIZE, HALF_IMAGE_SIZE),
    )

    loss_func = MSELoss()
    optimizer = SGD(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for epoch in tqdm(range(EPOCHS)):
        train_loss = 0.0
        model.train()

        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(available_device), y.to(available_device)

            with autocast(device_type=available_device):
                y_pred = model(x)
                loss = loss_func(y_pred, y)

            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = train_loss / len(train_dataloader)

        test_loss = 0.0

        model.eval()

        with torch.inference_mode():
            for batch, (x, y) in enumerate(test_dataloader):
                x, y = x.to(available_device), y.to(available_device)

                with autocast(device_type=available_device):
                    test_pred_logits = model(x)
                    loss = loss_func(test_pred_logits, y)

                test_loss += loss.item()

        test_loss = test_loss / len(test_dataloader)

        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss} | "
            f"Test Loss: {test_loss}"
        )

    save_model(model, "models", "FSRCNN")

    """

    model.load_state_dict(torch.load("models/FSRCNN.pth"))

    model.eval()

    from cv2 import merge, imwrite
    import numpy as np
    from torch import from_numpy

    test_image_tensor, test_image_cb, test_image_cr = (
        Image.open("dataset/hard_210_1100.jpg").convert("YCbCr").split()
    )
    test_image_tensor = np.array(test_image_tensor, dtype=np.float32) / 255.0
    test_image_cb = np.array(test_image_cb, dtype=np.float32) / 255.0
    test_image_cr = np.array(test_image_cr, dtype=np.float32) / 255.0

    test_image_tensor = from_numpy(test_image_tensor).unsqueeze(0).to(available_device)

    output = model(test_image_tensor)

    y_value = (
        output.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .mul(255)
        .clamp(0, 255)
        .cpu()
        .numpy()
        .astype("uint8")
    ).astype(np.float32) / 255.0

    ycbr_image = merge([y_value, test_image_cb, test_image_cr])

    image_dtype = ycbr_image.dtype
    ycbr_image *= 255.0

    image = np.matmul(
        ycbr_image,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0.00791071, -0.00153632, 0],
            [0, -0.00318811, 0.00625893],
        ],
    ) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.0
    image = image.astype(image_dtype)

    imwrite("test.png", image * 255.0)
