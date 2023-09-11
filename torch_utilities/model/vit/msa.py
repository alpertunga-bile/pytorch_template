from torch import nn


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )

        return attn_output
