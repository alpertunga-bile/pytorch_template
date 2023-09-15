from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch import autocast, nn, inference_mode
from torch_utilities.model.save_load import save_model
from torch_utilities.dataset.fsrcnn_dataset import create_dataloader
from torch.optim import SGD
from torch.nn import MSELoss
from torch_utilities.consts import available_device


def train_fsrcnn(
    model: nn.Module,
    model_name: str,
    train_folderpath: str,
    test_folderpath: str,
    wanted_image_size: int,
    epochs: int,
    batch_size: int = 32,
) -> None:
    train_dataloader = create_dataloader(
        image_folder=train_folderpath,
        batch_size=batch_size,
        wanted_image_size=(wanted_image_size, wanted_image_size),
    )

    test_dataloader = create_dataloader(
        image_folder=test_folderpath,
        batch_size=batch_size,
        wanted_image_size=(wanted_image_size, wanted_image_size),
    )

    loss_func = MSELoss()
    optimizer = SGD(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for epoch in tqdm(range(epochs)):
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

        with inference_mode():
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

    save_model(model, "models", model_name)
