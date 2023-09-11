from torch import nn
from torchvision import transforms
import torch

from torch_utilities.dataset.cv_dataset import create_dataloaders
from torch_utilities.plot import show_single_image
from torch_utilities.model.vit.ViT import ViT


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

    manual_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=manual_transform,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    image_batch, label_batch = next(iter(train_dataloader))

    image, label = image_batch[0], label_batch[0]

    random_image_tensor = torch.randn(1, 3, 224, 224)

    vit = ViT(num_classes=len(class_names))

    print(vit(random_image_tensor))
