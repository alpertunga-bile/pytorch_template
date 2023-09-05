from torch import nn, device, dtype, inference_mode, softmax, argmax
from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt


def pred_and_plot_image(
    model: nn.Module,
    image_path: str,
    device: device,
    type: dtype,
    class_names: list[str] = None,
    transform: transforms.Compose = None,
):
    image = read_image(image_path).type(type)
    image = image / 255.0

    if transform:
        image = transform(image)

    model = model.to(device)

    model.eval()
    with inference_mode():
        image = image.unsqueeze(dim=0)
        image_pred = model(image.to(device))

    image_pred_probs = softmax(image_pred, dim=1)
    image_pred_label = argmax(image_pred_probs, dim=1)

    plt.imshow(image.squeeze().permute(1, 2, 0))

    if class_names:
        title = f"Pred: {class_names[image_pred_label.cpu()]} | Prob: {image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {image_pred_label} | Prob: {image_pred_probs.max().cpu():.3f}"

    plt.title(title)
    plt.axis(False)
