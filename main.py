from torchvision import transforms

from torch_utilities.dataset.cv_dataset import create_dataloaders
from torch_utilities.plot import show_single_image
from torch_utilities.model.vit.embedding import PatchEmbedding

if __name__ == "__main__":
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
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

    patchify = PatchEmbedding(3, 16, 768)
    patch_embedded_img = patchify(image.unsqueeze(0))
