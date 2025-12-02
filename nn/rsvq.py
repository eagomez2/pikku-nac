import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple
)


class VectorQuantizer(nn.Module):
    # Based on https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py
    def __init__(
            self,
            input_size: int,
            embedding_size: int,
            num_embeddings: int,
            proj_weight_norm: bool = True,
            tau: float = 1.0
    ) -> None:
        super().__init__()

        # Params
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        self.tau = tau

        # Modules
        self.in_proj = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.embedding_size,
            kernel_size=1
        )
        self.out_proj = nn.Conv1d(
            in_channels=self.embedding_size,
            out_channels=self.input_size,
            kernel_size=1
        )

        if proj_weight_norm:
            self.in_proj = nn.utils.parametrizations.weight_norm(self.in_proj)
            self.out_proj =\
                nn.utils.parametrizations.weight_norm(self.out_proj)

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project embeddings into low-dimensional space
        x_proj = self.in_proj(x)  # (batch_size, embedding_size, seq_len)

        # Obtain quantized x and codebook indices
        x_quantized, x_idxs, x_logits = self._quantize(x_proj)

        # Calculate losses
        # NOTE: This losses have to be calculated here to facilitate collecting
        # all losses form all quantizers
        commitment_loss = F.mse_loss(
            input=x_proj,
            target=x_quantized.detach(),
            reduction="none"
        ).mean([1, 2])
        codebook_loss = F.mse_loss(
            input=x_quantized,
            target=x_proj.detach(),
            reduction="none"
        ).mean([1, 2])

        # noop in forward, straight-through gradient estimator in backward
        x_quantized = x_proj + (x_quantized - x_proj).detach()
        x_quantized = self.out_proj(x_quantized)

        return (
            x_quantized,
            x_logits,
            x_idxs,
            commitment_loss,
            codebook_loss
        )
    
    def _quantize(self, x_proj: torch.Tensor) -> Tuple[torch.Tensor]:
        # Reshape to (batch_size * seq_len, embedding_size)
        batch_size, embedding_size, seq_len = x_proj.size() 
        x_proj = x_proj.permute(0, 2, 1).reshape(
            batch_size * seq_len,
            embedding_size
        )

        # Apply L2 normalization to obtain cosine distance
        codebook = self.codebook.weight
        x_proj = F.normalize(x_proj, p=2.0)
        codebook = F.normalize(codebook, p=2.0)

        # Compute euclidean distance with codebook
        distance = (
            x_proj.pow(2).sum(1, keepdim=True)
            - 2 * x_proj @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )

        # Get logits
        x_logits = - distance / self.tau

        # Get quantized values
        x_idxs = (-distance).max(1)[1].view(batch_size, seq_len)
        x_quantized = F.embedding(
            x_idxs,
            self.codebook.weight
        ).swapaxes(1, 2)
        return x_quantized, x_idxs, x_logits


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_embeddings: int,
        embedding_size: int | List[int],
        codebook_dropout: float = 0.0,
        proj_weight_norm: bool = True
    ) -> None:
        super().__init__()

        # Pre-processing
        if isinstance(embedding_size, int):
            embedding_size = [embedding_size for _ in range(num_layers)]

        # Params
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.codebook_dropout = codebook_dropout

        # Modules
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    input_size=input_size,
                    embedding_size=embedding_size[idx],
                    num_embeddings=num_embeddings,
                    proj_weight_norm=proj_weight_norm
                )
                for idx in range(num_layers)
            ]
        )
    
    def forward(self, x: torch.Tensor, num_active_layers: int = None):
        # Setup
        batch_size = x.size(0)
        x_quantized = 0
        x_residual = x
        x_idxs = []
        x_logits = []  # Can have different weights per layer
        commitment_loss = 0.0
        codebook_loss = 0.0

        # Determine number of layers to use
        num_active_layers = torch.tensor(
            (
                self.num_layers if num_active_layers is None
                else num_active_layers
            ),
            dtype=torch.float32,
            device=x.device
        )
        
        if self.training:
            # NOTE: +1 for samples having all layers enabled
            num_active_layers = (
                torch.ones((batch_size,), device=x.device)
                * (self.num_layers + 1)
            )

            # Generates the full dropout pattern
            dropout = torch.randint(
                1, self.num_layers + 1, (batch_size,),
                device=x.device
            )

            # Generates the dropout pattern that is always applied to the first
            # elements of the batch
            num_dropped = int(batch_size * self.codebook_dropout)
            num_active_layers[:num_dropped] = dropout[:num_dropped]

        # Iterate through quantizers
        for idx, quantizer in enumerate(self.quantizers):
            if not self.training and idx >= num_active_layers:
                break

            # Apply layer quantizer
            (
                x_quantized_layer,
                x_logits_layer,
                x_idxs_layer,
                commit_loss_layer,
                codebook_loss_layer
            ) = quantizer(x_residual)

            # Create mask for dropout
            # NOTE: Values are false when the previous tensors contains a layer
            # value below the current layer
            mask = (
                torch.full((batch_size,), fill_value=idx, device=x.device)
                < num_active_layers
            )

            # Accumulate quantized output
            x_quantized = x_quantized + x_quantized_layer * mask[:, None, None]
            x_residual = x_residual - x_quantized_layer

            # Accumulate losses averaging only over active elements
            commitment_loss += (commit_loss_layer * mask).sum() / mask.sum()
            codebook_loss += (codebook_loss_layer * mask).sum() / mask.sum()

            # Save intermediate outputs
            x_idxs.append(x_idxs_layer)  # (batch_size, seq_len)
            x_logits.append(x_logits_layer)  # (batch_size, num_embeddings)

        # Concat all idxs
        # (batch_size, num_layers, seq_len)
        x_idxs = torch.stack(x_idxs, dim=1)

        # (batch_size, num_layers, num_embeddings)
        x_logits = torch.stack(x_logits, dim=1)

        return x_quantized, x_idxs, x_logits, commitment_loss, codebook_loss
