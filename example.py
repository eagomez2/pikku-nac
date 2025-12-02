import torch
from nn.pikku_nac import PikkuNAC


def main() -> None:
    # The input needs a minimum length of frame size of the model. The frame
    # size is simply the multiplication of all encoder strides in
    # encoder_strides_seq or decoder strides in decoder_strides_seq.
    x = torch.rand((1, 1, 512), dtype=torch.float32)
    net = PikkuNAC(
        encoder_strides_seq=(2, 4, 8, 8),
        decoder_strides_seq=(8, 8, 4, 2)
    )

    # The model returns the following outputs:
    #
    # 1. x_quantized: Quantized input.
    # 2. x_idxs: Codebook indices of each quantizer layer.
    # 3. x_logits: Logits corresponding to each quantizer layer.
    # 4. commitment_loss: Commitment loss term commonly used with
    # straight-through estimator.
    # 5. codebook_loss: Codebook loss term commonly used with straight-through
    # estimator.
    x_quantized, x_idxs, x_logits, commitment_loss, codebook_loss = net(x)


if __name__ == "__main__":
    main()
