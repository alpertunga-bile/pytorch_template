from torch import nn, save
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
