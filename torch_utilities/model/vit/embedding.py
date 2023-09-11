from torch import nn, Tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embedding_dim: int = 768,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Flat the end of the 2 variables
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: Tensor) -> Tensor:
        image_res = x.shape[-1]

        assert (
            image_res % self.patch_size == 0
        ), f"Image size {image_res} | Patch Size {self.patch_size} Not divisible!!!"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)
