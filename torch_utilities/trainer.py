from tqdm.auto import tqdm
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter

from torch import nn, device, softmax, argmax, inference_mode, Tensor, rand
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from torch_utilities.model.generative_helper import gen_noise, get_gp
from torch_utilities.model.save_load import save_gan_models
from torch_utilities.plot import save_grid


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.Module,
    optimizer: Optimizer,
    device: device,
) -> Tuple[float, float]:
    model.train()

    train_loss = train_acc = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_func(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = argmax(softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(
    model: nn.Module, dataloader: DataLoader, loss_func: nn.Module, device: device
) -> Tuple[float, float]:
    model.eval()

    test_loss = test_acc = 0.0

    with inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_func(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_func: nn.Module,
    epochs: int,
    device: device,
    writer: SummaryWriter,
) -> Dict[str, list]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_func=loss_func,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_func=loss_func, device=device
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer is None:
            continue

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
            global_step=epoch,
        )

        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
            global_step=epoch,
        )

        # Add example image
        # writer.add_graph(model=model, input_to_model=)

    if writer:
        writer.close()

    return results


def crit_step(
    current_batch_size: int,
    cycles: int,
    z_dim: int,
    real: Tensor,
    gen: nn.Module,
    crit: nn.Module,
    crit_opt: Optimizer,
    device: device,
) -> float:
    mean_crit_loss = 0

    for _ in range(cycles):
        crit_opt.zero_grad()

        noise = gen_noise(current_batch_size, z_dim, device)
        fake = gen(noise)
        crit_fake_pred = crit(fake.detach())
        crit_real_pred = crit(real)

        alpha = rand(len(real), 1, 1, 1, device=device, requires_grad=True)
        gp = get_gp(real, fake.detach(), crit, alpha)

        crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp
        mean_crit_loss += crit_loss.item() / cycles
        crit_loss.backward(retain_graph=True)
        crit_opt.step()

    return mean_crit_loss


def gen_step(
    current_batch_size: int,
    z_dim: int,
    gen: nn.Module,
    crit: nn.Module,
    gen_opt: nn.Module,
    device: device,
) -> Tuple[float, Tensor]:
    gen_opt.zero_grad()

    noise = gen_noise(current_batch_size, z_dim, device)
    fake = gen(noise)
    crit_fake_pred = crit(fake)

    gen_loss = -crit_fake_pred.mean()
    gen_loss.backward()
    gen_opt.step()

    return gen_loss.item(), fake


def train_gan(
    epochs: int,
    crit_cycles: int,
    z_dim: int,
    dataloader: DataLoader,
    gen: nn.Module,
    crit: nn.Module,
    gen_opt: Optimizer,
    crit_opt: Optimizer,
    device: device,
    foldername: str,
    modelname: str,
    writer: SummaryWriter = None,
    save_step: int = 100,
    show_step: int = 100,
) -> Tuple[list, list]:
    gen_losses = []
    crit_losses = []
    current_step = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        for real, _ in dataloader:
            current_batch_size = len(real)
            real = real.to(device)

            mean_crit_loss = crit_step(
                current_batch_size,
                crit_cycles,
                z_dim,
                real,
                gen,
                crit,
                crit_opt,
                device,
            )
            crit_losses.append(mean_crit_loss)

            gen_loss, fake = gen_step(
                current_batch_size, z_dim, gen, crit, gen_opt, device
            )
            gen_losses.append(gen_loss)

            if current_step % save_step == 0 and current_step > 0:
                save_gan_models(
                    modelname, epochs, gen, gen_opt, crit, crit_opt, foldername
                )

            if current_step % show_step == 0 and current_step > 0:
                save_grid(fake, foldername, filename=f"{modelname}-{current_step}")

                gen_mean = sum(gen_losses[-show_step:]) / show_step
                crit_mean = sum(crit_losses[-show_step:]) / show_step

                print(
                    f"Epoch : {epoch + 1} | Step : {current_step} | Gen Loss : {gen_mean} | Disc Loss : {crit_mean}"
                )

            current_step += 1

            if writer is None:
                continue

            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"gan_loss": gen_loss, "crit_loss": mean_crit_loss},
                global_step=epoch,
            )

    if writer:
        writer.close()

    return gen_losses, crit_losses
