# misks-wireless-autoencoder

End-to-end neuronski autoencoder za bezicni prenos preko AWGN kanala.
Encoder i decoder se treniraju zajedno kroz diferencijabilan kanal — mreza
sama nauci kodiranje i modulaciju, bez ijedne rucno dizajnirane komponente.

Projekat iz predmeta **Modelovanje i simulacija komunikacionih sistema (MiSKS)**,
Fakultet tehnickih nauka, Univerzitet u Novom Sadu.
Autor: **Nikola Bakic** (EE 58/2022).

Reimplementacija MATLAB primera
[Autoencoders for Wireless Communications](https://www.mathworks.com/help/comm/ug/autoencoders-for-wireless-communications.html)
po radu **O'Shea & Hoydis, 2017** — *An Introduction to Deep Learning for the Physical Layer*.

---

## Arhitektura

```
ENCODER:  one-hot(M=16) -> FC(16) -> ReLU -> FC(7) -> Energy norm (||x||^2 = n)
KANAL:    AWGN, treniran na Eb/No = 3 dB, sigma^2 = 1/(2*R*EbNo)
DECODER:  FC(16) -> ReLU -> FC(16) -> softmax (argmax -> poruka)
```

- `k = 4` informaciona bita, `n = 7` channel uses, `M = 16` poruka, `R = 4/7`
- Loss: categorical cross-entropy, optimizer: Adam
- Enkoder emituje `n` **realnih** vrednosti; po dva channel use-a daju
  jedan kompleksni simbol (ista konvencija kao u MATLAB primeru).

---

## Struktura repoa

```
misks-wireless-autoencoder/
├── README.md
├── .gitignore
├── pytorch/
│   └── wireless_autoencoder.ipynb     # trening + evaluacija (Colab/Jupyter)
├── matlab/
│   ├── autoencoder_wireless.m         # MathWorks primer (referenca)
│   └── matlab_bler.csv                # MATLAB AE BLER vs Eb/No (za cross-val)
├── huggingface/
│   ├── app.py                         # Gradio Space demo
│   ├── requirements.txt
│   ├── README.md                      # HF model card
│   └── upload_model.py                # upload .pt + card na HF Hub
├── results/
│   ├── loss_curve.png                 # cross-entropy tokom treninga
│   ├── bler_4_systems.png             # AE vs Hamming MLD/HDD vs Uncoded QPSK
│   ├── bler_matlab.png                # MATLAB referentni BLER grafik
│   ├── crossval_matlab_pytorch.png    # PyTorch AE vs MATLAB AE (slaganje)
│   ├── constellation_pca.png          # naucena konstelacija (PCA 7D->2D)
│   ├── awgn_vs_rayleigh.png           # generalizacija na fading bez retreninga
│   └── hf_space_screenshot.png        # snimak Gradio demo-a
└── docs/
    ├── izvestaj.pdf                   # pisani izvestaj
    └── prezentacija.pdf               # slajdovi
```

---

## Kako pokrenuti

### PyTorch notebook (Colab ili lokalno)

```bash
pip install torch numpy scipy matplotlib scikit-learn
jupyter notebook pytorch/wireless_autoencoder.ipynb
```

Notebook prolazi sve faze: trening (Eb/No = 3 dB, 30 epoha), evaluacija BLER-a,
poredjenje sa Hamming(7,4) MLD/HDD i uncoded QPSK baseline-om, vizuelizacija
konstelacije i Rayleigh test bez retreninga.

### MATLAB referenca

```matlab
% otvori matlab/autoencoder_wireless.m u MATLAB R2020a+
% potrebni toolboxi: Deep Learning Toolbox, Communications Toolbox
```

Pokretanje generise sopstveni `matlab_bler.csv` koji moze da zameni
verziju u repou (vrednosti ovde su iz autorovog ranijeg run-a).

### Hugging Face Space (interaktivni demo)

- **Live demo:** https://huggingface.co/spaces/nikolabakic/wireless-autoencoder-demo
- **Model:** https://huggingface.co/nikolabakic/wireless-autoencoder-7-4

Lokalno pokretanje Space-a:

```bash
cd huggingface
pip install -r requirements.txt
python app.py
```

Slider-i podesavaju opseg Eb/No-a, broj test poruka i prikaz baseline krivih.
Demo plota BLER vs Eb/No i naucenu konstelaciju (PCA projekcija).

---

## Rezultati (BLER, AWGN)

| Eb/No (dB) | Autoencoder (7,4) | Hamming(7,4) MLD | Hamming(7,4) HDD |
|-----------:|------------------:|-----------------:|-----------------:|
| 0          | 0.183             | 0.179            | 0.262            |
| 2          | 0.067             | 0.063            | 0.123            |
| 4          | 0.013             | 0.012            | 0.037            |
| 6          | 0.0010            | 0.0008           | 0.0053           |
| 8          | 2e-5              | 1e-5             | 2.8e-4           |

- Na BLER = 10⁻²: autoencoder **4.22 dB**, MLD Hamming **4.15 dB** (gap ~ **0.07 dB**)
- Coding gain nad uncoded (4,4) QPSK: **~ 1.7 dB** na BLER = 10⁻²
- d_min naucenog koda: **3.28** (Hamming BPSK: 3.46) — istovetne performanse
  jer BLER zavisi od celog spektra distanci, ne samo od minimalne
- Cross-validacija PyTorch <-> MATLAB: krive se preklapaju u celom opsegu
  (vidi `results/crossval_matlab_pytorch.png`)
- Generalizacija na Rayleigh fading bez retreninga: BLER plato ~10⁻¹
  (vidi `results/awgn_vs_rayleigh.png`) — pokazuje da AE treniran na AWGN
  ne resava fading bez ponovnog treninga sa fading-aware kanalom

---

## Linkovi

- **GitHub:** https://github.com/nikolabakic/misks-wireless-autoencoder
- **HF model:** https://huggingface.co/nikolabakic/wireless-autoencoder-7-4
- **HF Space:** https://huggingface.co/spaces/nikolabakic/wireless-autoencoder-demo
- **MATLAB primer:** https://www.mathworks.com/help/comm/ug/autoencoders-for-wireless-communications.html

---

## Citation

```bibtex
@article{oshea2017introduction,
  title  = {An Introduction to Deep Learning for the Physical Layer},
  author = {O'Shea, Timothy and Hoydis, Jakob},
  journal= {IEEE Transactions on Cognitive Communications and Networking},
  volume = {3}, number = {4}, pages = {563--575}, year = {2017},
  doi    = {10.1109/TCCN.2017.2758370}
}
```

## License

MIT — vidi `huggingface/README.md` za model licencu.
