from os import cpu_count
from torch.cuda import is_available

NUM_WORKERS = cpu_count()
available_device = "cuda" if is_available() else "cpu"
default_epochs = 20
default_batch_size = 32
default_hidden_units = 16
default_learning_rate = 1e-3
