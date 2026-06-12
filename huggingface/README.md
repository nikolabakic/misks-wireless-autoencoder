---
license: mit
library_name: pytorch
tags:
- wireless-communications
- autoencoder
- channel-coding
- communications
- signal-processing
pipeline_tag: other
---

# Wireless Autoencoder (7,4) — End-to-End Learned Communication System

PyTorch reimplementacija MATLAB primera
[Autoencoders for Wireless Communications](https://www.mathworks.com/help/comm/ug/autoencoders-for-wireless-communications.html)
po radu O'Shea & Hoydis (2017). Encoder i decoder su neuronske mreže trenirane
zajedno (end-to-end) kroz diferencijabilan AWGN kanal, tako da mreža sama nauči
kodiranje i modulaciju bez ijedne ručno dizajnirane komponente.

Projekat iz predmeta Modelovanje i simulacija komunikacionih sistema (MiSKS),
Fakultet tehničkih nauka, Univerzitet u Novom Sadu.

## Arhitektura

```
ENCODER:  one-hot(M=16) -> FC(16) -> ReLU -> FC(7) -> Energy norm (||x||^2 = n)
KANAL:    AWGN, treniran na Eb/No = 3 dB, sigma^2 = 1/(2*R*EbNo)
DECODER:  FC(16) -> ReLU -> FC(16) -> softmax (argmax -> poruka)
```

- k = 4 informaciona bita, n = 7 channel uses, M = 16 poruka, R = 4/7
- Loss: categorical cross-entropy, optimizer: Adam
- Enkoder daje n **realnih** vrednosti; po dva channel use-a čine jedan
  kompleksni simbol (ista konvencija kao MATLAB primer)

## Rezultati (BLER, AWGN)

| Eb/No (dB) | Autoencoder (7,4) | Hamming(7,4) MLD | Hamming(7,4) HDD |
|-----------:|------------------:|-----------------:|-----------------:|
| 0          | 0.183             | 0.179            | 0.262            |
| 2          | 0.067             | 0.063            | 0.123            |
| 4          | 0.013             | 0.012            | 0.037            |
| 6          | 0.0010            | 0.0008           | 0.0053           |
| 8          | 2e-5              | 1e-5             | 2.8e-4           |

- Na BLER = 10⁻²: autoencoder 4.22 dB, MLD Hamming 4.15 dB (gap **0.07 dB**)
- Coding gain nad uncoded (4,4) QPSK: **≈ 1.7 dB** na BLER = 10⁻²
- d_min naučenog koda: 3.28 (Hamming BPSK: 3.46) uz iste performanse —
  BLER zavisi od celog spektra distanci, ne samo od minimalne

## Upotreba

```python
import torch
from huggingface_hub import hf_hub_download

# Klase Encoder/Decoder/AWGNChannel/WirelessAutoencoder: vidi app.py u
# pratećem Space-u ili GitHub repo ispod.
path = hf_hub_download(
    repo_id="nikolabakic/wireless-autoencoder-7-4",
    filename="wireless_autoencoder_7_4.pt",
)
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model = WirelessAutoencoder(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

Interaktivni demo: [Gradio Space](https://huggingface.co/spaces/nikolabakic/wireless-autoencoder-demo)

## Linkovi

- GitHub repo: https://github.com/nikolabakic/misks-wireless-autoencoder
- MATLAB primer: https://www.mathworks.com/help/comm/ug/autoencoders-for-wireless-communications.html

## Citation

```bibtex
@article{oshea2017introduction,
  title={An Introduction to Deep Learning for the Physical Layer},
  author={O'Shea, Timothy and Hoydis, Jakob},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  volume={3}, number={4}, pages={563--575}, year={2017},
  doi={10.1109/TCCN.2017.2758370}
}
```
