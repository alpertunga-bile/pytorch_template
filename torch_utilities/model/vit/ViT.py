from torch import nn, randn
import torch

from .transformer_encoder import TransformerEncoderBlock
from torch_utilities.model.vit.embedding import PatchEmbedding


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert (
            image_size % patch_size == 0
        ), f"Image size is not divisible by patch size"

        self.number_of_patches = int((image_size * image_size) / patch_size**2)

        self.class_embedding = nn.Parameter(
            data=randn(1, 1, embedding_dim), requires_grad=True
        )

        self.position_embedding = nn.Parameter(
            data=randn(1, self.number_of_patches + 1, embedding_dim),
            requires_grad=True,
        )

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_layer = PatchEmbedding(in_channels, patch_size, embedding_dim)

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim, num_heads, mlp_size, mlp_dropout, attn_dropout
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_layer(x)

        x = torch.cat((class_token, x), dim=1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x
