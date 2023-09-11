from torchinfo import summary
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from os.path import join
from os import makedirs


def print_model_info(
    model: nn.Module,
    input_size,
    verbose=0,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
):
    summary(
        model,
        input_size=input_size,
        verbose=verbose,
        col_names=col_names,
        col_width=col_width,
        row_settings=row_settings,
    )


def create_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> SummaryWriter():
    timestamp = datetime.now().strftime("%d-%m-%Y")

    makedirs("runs", exist_ok=True)

    if extra:
        log_dir = join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = join("runs", timestamp, experiment_name, model_name)

    print(f"Saving to {log_dir}")

    return SummaryWriter(log_dir=log_dir)
