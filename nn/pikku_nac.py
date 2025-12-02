import math
import torch
import torch.nn as nn
from typing import (
    List,
    Tuple
)
from .snake1d import Snake1d
from .rsvq import ResidualVectorQuantizer


class ResidualBlock(nn.Module):
    def __init__(
            self,
            num_channels: int,
            kernel_size: int = 5,
            dilation: int = 1,
            bias: bool = True
    ) -> None:
        super().__init__()

        # Params
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        # Modules
        self.conv1d_0 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                bias=self.bias
            )
        )
        self.activation_0 = Snake1d(num_channels)
        self.conv1d_1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=1,
                bias=self.bias
            )
        )
        self.activation_1 = Snake1d(num_channels)

    @property
    def padding(self) -> int:
        return ((self.kernel_size - 1) * self.dilation) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.conv1d_0(self.activation_0(x))
        x = self.conv1d_1(self.activation_1(x))
        pad = (x_skip.shape[-1] - x.shape[-1]) // 2

        if pad > 0:
            x_skip = x_skip[..., pad:-pad]

        return x + x_skip


class EncoderBlock(nn.Module):
    def __init__(
            self,
            num_channels: int,
            stride: int = 1,
            residual_block_kernel_size: int = 5,
            bias: bool = True
    ) -> None:
        super().__init__()

        # Params
        self.num_channels = num_channels
        self.stride = stride
        self.residual_block_kernel_size = residual_block_kernel_size
        self.bias = bias

        # Modules
        self.resblock_0 = ResidualBlock(
            num_channels=self.num_channels // 2,
            kernel_size=self.residual_block_kernel_size,
            dilation=1,
            bias=self.bias
        )
        self.resblock_1 = ResidualBlock(
            num_channels=self.num_channels // 2,
            kernel_size=self.residual_block_kernel_size,
            dilation=3,
            bias=self.bias
        )
        self.resblock_2 = ResidualBlock(
            num_channels=self.num_channels // 2,
            kernel_size=self.residual_block_kernel_size,
            dilation=9,
            bias=self.bias
        )

        self.activation = Snake1d(num_channels // 2)
        self.conv1d = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=self.num_channels // 2,
                out_channels=self.num_channels,
                kernel_size=2 * self.stride,
                stride=self.stride,
                padding=math.ceil(self.stride / 2),
                bias=self.bias
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resblock_0(x)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.activation(x)
        x = self.conv1d(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            num_channels: int,
            stride: int = 1,
            residual_block_kernel_size: int = 5,
            bias: bool = True
    ) -> None:
        super().__init__()

        # Params
        self.num_channels = num_channels
        self.stride = stride
        self.residual_block_kernel_size = residual_block_kernel_size
        self.bias = bias

        # Modules
        self.activation = Snake1d(self.num_channels * 2)
        self.convtranspose1d = nn.utils.parametrizations.weight_norm(
            nn.ConvTranspose1d(
                in_channels=self.num_channels * 2,
                out_channels=self.num_channels,
                kernel_size=2 * self.stride,
                stride=self.stride,
                padding=math.ceil(self.stride / 2),
                bias=self.bias,
            )
        )
        self.resblock_0 = ResidualBlock(
            num_channels=self.num_channels,
            kernel_size=self.residual_block_kernel_size,
            dilation=1,
            bias=self.bias
        )
        self.resblock_1 = ResidualBlock(
            num_channels=self.num_channels,
            kernel_size=self.residual_block_kernel_size,
            dilation=3,
            bias=self.bias
        )
        self.resblock_2 = ResidualBlock(
            num_channels=self.num_channels,
            kernel_size=self.residual_block_kernel_size,
            dilation=9,
            bias=self.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.convtranspose1d(x)
        x = self.resblock_0(x)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        return x


class PikkuNAC(nn.Module):
    def __init__(
            self,
            encoder_channels: int = 32,
            encoder_strides_seq: Tuple = (2, 4, 8, 8),
            quantizer_in_channels: int | None = None,
            num_quantizer_layers: int = 9,
            num_embeddings: int = 2 ** 10,
            embedding_size: int | List[int] = 8,
            quantizer_dropout: float = 0.5,
            decoder_channels: int = 256,
            decoder_strides_seq: Tuple = (8, 8, 4, 2),
            residual_block_kernel_size: int = 5,  # 5 for PikkuNAC, 7 for DAC
            commitment_loss_weight: float = 1.0,
            codebook_loss_weight: float = 1.0,
            bias: bool = True
    ) -> None:
        super().__init__()

        # Assertions
        if len(encoder_strides_seq) !=  len(decoder_strides_seq):
            raise ValueError(
                "encoder_strides_seq and decoder_strides_seq should have the "
                "same number of items"
            )

        # Params
        self.encoder_channels = encoder_channels
        self.encoder_strides_seq = encoder_strides_seq
        self.quantizer_in_channels = (
            self.encoder_channels * 2 ** len(self.encoder_strides_seq)
            if quantizer_in_channels is None else quantizer_in_channels
        )
        self.num_quantizer_layers = num_quantizer_layers
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.quantizer_dropout = quantizer_dropout
        self.decoder_channels = decoder_channels
        self.decoder_strides_seq = decoder_strides_seq
        self.residual_block_kernel_size = residual_block_kernel_size
        self.commitment_loss_weight = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.bias = bias

        # Modules
        self.encoder = self._build_encoder()
        self.quantizer = ResidualVectorQuantizer(
            input_size=self.quantizer_in_channels,
            num_layers=self.num_quantizer_layers,
            num_embeddings=self.num_embeddings,
            embedding_size=self.embedding_size,
            codebook_dropout=self.quantizer_dropout,
            proj_weight_norm=True
        )
        self.decoder = self._build_decoder()
    
    def _build_encoder(self) -> nn.ModuleDict:
        encoder = nn.ModuleDict()
        block_num_channels = self.encoder_channels

        # Encoder input block
        encoder.add_module(
            "in_conv1d",
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=self.encoder_channels,
                    kernel_size=7,
                    padding=3,
                    bias=self.bias
                )
            )
        )

        # Add encoder blocks
        for idx, stride in enumerate(self.encoder_strides_seq):
            block_num_channels = 2 * block_num_channels
            encoder.add_module(
                f"block_{idx}",
                EncoderBlock(
                    num_channels=block_num_channels,
                    stride=stride,
                    residual_block_kernel_size=self.residual_block_kernel_size,
                    bias=self.bias
                )
            )
        
        # Add output block
        encoder.add_module("out_activation", Snake1d(block_num_channels))
        encoder.add_module(
            "out_conv1d",
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    in_channels=block_num_channels,
                    out_channels=self.quantizer_in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=self.bias
                )
            )
        )

        return encoder
 
    def _build_decoder(self) -> nn.ModuleDict:
        decoder = nn.ModuleDict()

        # Encoder input block
        decoder.add_module(
            "in_conv1d",
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    in_channels=self.quantizer_in_channels,
                    out_channels=self.decoder_channels,
                    kernel_size=7,
                    padding=3,
                    bias=self.bias
                )
            )
        )

        for idx, stride in enumerate(self.decoder_strides_seq):
            block_num_channels = self.decoder_channels // 2 ** (idx + 1)

            decoder.add_module(
                f"block_{idx}",
                DecoderBlock(
                    num_channels=block_num_channels,
                    stride=stride,
                    residual_block_kernel_size=\
                        self.residual_block_kernel_size,
                    bias=self.bias
                )
            )
                
        # Add output block
        decoder.add_module("out_activation", Snake1d(block_num_channels))
        decoder.add_module(
            "out_conv1d",
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    in_channels=block_num_channels,
                    out_channels=1,
                    kernel_size=7,
                    padding=3,
                    bias=self.bias
                )
            )
        )
        decoder.add_module("out", nn.Tanh())

        return decoder
    
    @property
    def frame_size(self) -> int:
        return math.prod(self.encoder_strides_seq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Encoder
        for block in self.encoder.values():
            x = block(x)
        
        # Quantizer
        x, x_idxs, x_logits, commitment_loss, codebook_loss = self.quantizer(x)

        # Decoder
        for block in self.decoder.values():
            x = block(x)
        
        # Apply loss weights
        commitment_loss = self.commitment_loss_weight * commitment_loss
        codebook_loss = self.codebook_loss_weight * codebook_loss

        return x, x_idxs, x_logits, commitment_loss, codebook_loss
