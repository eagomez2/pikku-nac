# Pikku NAC
Pikku NAC is a lightweight neural audio codec. "Pikku," meaning "small" in Finnish, emphasizes the codec's efficient design. "NAC" stands for Neural Audio Codec, which utilizes neural networks for compressing audio data. The model architecture is based on the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec), which corresponds to a convolutional encoder, a residual vector quantizer, and a convolutional decoder. However, Pikku NAC adopts a significantly streamlined configuration, with an encoder having only 5M parameters, a decoder with 2M parameters, and a quantizer with just 156k parameters. This compact design makes it ideal for applications where computational resources are limited.

# Structure of this repository
The structure of this repository is as follows:
- `nn`: Neural network layers that constitute the model architecture.
- `example.py`: Example script demonstrating how to instantiate and use the model, and the description of the values it outputs.

# Cite
If this repository contributes to your research, please consider citing it:

```
@misc{pikkunac2025,
  author = {Esteban Gómez},
  title  = {PikkuNAC: A lightweight neural audio codec},
  year   = 2025,
  url    = {https://github.com/eagomez2/pikku-nac}
}
```

This repository was developed by <a href="https://estebangomez.me/" target="_blank">Esteban Gómez</a>, member of the <a href="https://www.aalto.fi/en/department-of-information-and-communications-engineering/speech-interaction-technology" target="_blank">Speech Interaction Technology group from Aalto University</a>.

# License
For further details about the license of this package, please see [LICENSE](LICENSE).
