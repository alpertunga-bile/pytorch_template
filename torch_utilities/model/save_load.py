from torch import nn, save, load
from torch.optim import Optimizer
from os import makedirs
from os.path import join


def save_model(model: nn.Module, target_dir: str, model_name: str):
    makedirs(target_dir, exist_ok=True)

    model_name = (
        model_name
        if model_name.endswith(".pt") or model_name.endswith(".pth")
        else model_name + ".pth"
    )

    path = join(target_dir, model_name)

    print(f"Saving {path} model")
    save(obj=model.state_dict(), f=path)


def save_gan_models(
    name: str,
    epochs: int,
    gen: nn.Module,
    gen_opt: Optimizer,
    crit: nn.Module,
    crit_opt: Optimizer,
    root: str,
) -> None:
    makedirs(root, exist_ok=True)

    gen_path = join(root, f"G-{name}.pkl")
    save(
        {
            "epoch": epochs,
            "model_state_dict": gen.state_dict(),
            "optimizer_state_dict": gen_opt.state_dict(),
        },
        gen_path,
    )
    print(f"Saved {gen_path}")

    crit_path = join(root, f"C-{name}.pkl")
    save(
        {
            "epoch": epochs,
            "model_state_dict": crit.state_dict(),
            "optimizer_state_dict": crit_opt.state_dict(),
        },
        crit_path,
    )

    print(f"Saved {crit_path}")


def load_gan_models(
    name: str,
    gen: nn.Module,
    gen_opt: Optimizer,
    crit: nn.Module,
    crit_opt: Optimizer,
    root: str,
) -> None:
    gen_path = join(root, f"G-{name}.pkl")

    checkpoint = load(gen_path)
    gen.load_state_dict(checkpoint["model_state_dict"])
    gen_opt.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded {gen_path}")

    crit_path = join(root, f"C-{name}.pkl")
    checkpoint = load(crit_path)
    crit.load_state_dict(checkpoint["model_state_dict"])
    crit_opt.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded {crit_path}")
