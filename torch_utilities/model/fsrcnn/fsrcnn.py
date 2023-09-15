from torch import nn


def get_map_block(s: int):
    return nn.Sequential(nn.Conv2d(s, s, 3, 1, padding="same"), nn.PReLU(s))


class FSRCNN(nn.Module):
    def __init__(self, d: int, s: int, m: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, d, 5, 1, padding="same"), nn.PReLU(56)
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, 1, 1, padding="same"), nn.PReLU(12)
        )

        self.mapping = nn.Sequential(*[get_map_block(s) for _ in range(m)])

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, 1, 1, padding="same"),
            nn.PReLU(56),
        )

        self.upscale = nn.ConvTranspose2d(d, 1, 9, 2, 4, 1, 1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.upscale(x)

        return x
