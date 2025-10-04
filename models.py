import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

__all__ = [
    "SqueezeExcitation",
    "ResidualBlock",
    "ChessNNWithResiduals",
]


class SqueezeExcitation(nn.Module):
    """Lightweight squeeze-excitation block for channel-wise reweighting."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    """Residual block with optional squeeze-excitation and dropout."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.activation(out)
        return out


class ChessNNWithResiduals(nn.Module):
    """Enhanced AlphaZero-style policy-value network for chess."""

    def __init__(
        self,
        in_channels: int = 15,
        channels: int = 512,
        num_residual_blocks: int = 8,
        policy_channels: int = 512,
        num_policy_outputs: int = 1968,
        value_hidden_dim: int = 512,
        dropout: float = 0.1,
        use_squeeze_excitation: bool = True,
        squeeze_excitation_every: Optional[int] = 3,
        use_transformer: bool = True,
        transformer_depth: int = 8,
        transformer_heads: int = 12,
        transformer_dim: int = 384,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        board_height: int = 8,
        board_width: int = 8,
    ) -> None:
        super().__init__()

        if squeeze_excitation_every is not None and squeeze_excitation_every <= 0:
            raise ValueError(
                "squeeze_excitation_every must be None or a positive integer"
            )

        if board_height <= 0 or board_width <= 0:
            raise ValueError("board dimensions must be positive")

        self.board_height = board_height
        self.board_width = board_width
        self.num_tokens = board_height * board_width

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        residual_blocks = []
        for block_idx in range(num_residual_blocks):
            use_se = use_squeeze_excitation and (
                squeeze_excitation_every is not None
                and (block_idx + 1) % squeeze_excitation_every == 0
            )
            residual_blocks.append(
                ResidualBlock(
                    channels=channels,
                    kernel_size=3,
                    dropout=dropout,
                    use_se=use_se,
                )
            )
        self.backbone = (
            nn.Sequential(*residual_blocks) if residual_blocks else nn.Identity()
        )

        self.use_transformer = use_transformer
        if self.use_transformer:
            if transformer_dim % transformer_heads != 0:
                raise ValueError(
                    "transformer_dim must be divisible by transformer_heads"
                )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=int(transformer_dim * transformer_mlp_ratio),
                dropout=transformer_dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_depth,
            )
            self.token_proj = nn.Linear(channels, transformer_dim)
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.num_tokens, transformer_dim)
            )
            self.transformer_dropout = (
                nn.Dropout(transformer_dropout)
                if transformer_dropout > 0
                else nn.Identity()
            )
            self.transformer_out_proj = nn.Linear(transformer_dim, channels)
        else:
            self.transformer = None
            self.token_proj = None
            self.pos_embedding = None
            self.transformer_dropout = None
            self.transformer_out_proj = None

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=policy_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(policy_channels * 8 * 8, num_policy_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=policy_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(policy_channels, value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(value_hidden_dim, 1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.use_transformer and self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        features = self.stem(x)
        features = self.backbone(features)

        if self.use_transformer:
            batch_size, channels, height, width = features.shape
            if height != self.board_height or width != self.board_width:
                raise ValueError(
                    "Feature map spatial size does not match configured board size"
                )

            tokens = (
                features.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, self.num_tokens, channels)
            )
            tokens = self.token_proj(tokens)
            tokens = tokens + self.pos_embedding[:, : self.num_tokens]
            tokens = self.transformer(tokens)
            tokens = self.transformer_dropout(tokens)
            tokens = self.transformer_out_proj(tokens)
            features = (
                tokens.view(batch_size, height, width, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        if return_intermediate:
            return policy_logits, value, features

        return policy_logits, value
