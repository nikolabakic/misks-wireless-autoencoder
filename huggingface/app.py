"""Wireless Autoencoder (7,4) - interaktivni Gradio demo.

MiSKS projekat, FTN UNS. Model treniran u PyTorch-u po uzoru na MATLAB primer
"Autoencoders for Wireless Communications" (O'Shea & Hoydis, 2017).
"""

import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from scipy.special import erfc
from sklearn.decomposition import PCA

REPO_ID = os.environ.get("MODEL_REPO", "nikolabakic/wireless-autoencoder-7-4")
CKPT_NAME = "wireless_autoencoder_7_4.pt"


# ----------------------------- model (isti kao u trening notebooku) ---------

class AWGNChannel(nn.Module):
    """AWGN kanal: sigma^2 = 1 / (2 * R * EbNo_lin) po realnoj komponenti."""

    def __init__(self, k: int, n: int, train_ebno_db: float = 3.0):
        super().__init__()
        self.k, self.n = k, n
        self.train_ebno_db = train_ebno_db

    def forward(self, x, ebno_db=None):
        if ebno_db is None:
            ebno_db = self.train_ebno_db
        R = self.k / self.n
        sigma = (1.0 / (2.0 * R * 10.0 ** (ebno_db / 10.0))) ** 0.5
        return x + torch.randn_like(x) * sigma


class Encoder(nn.Module):
    """Tx: one-hot (M) -> n realnih vrednosti, Energy normalizacija po uzorku."""

    def __init__(self, M: int, n: int):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(nn.Linear(M, M), nn.ReLU(), nn.Linear(M, n))

    def forward(self, x):
        out = self.net(x)
        norm = torch.linalg.vector_norm(out, dim=1, keepdim=True)
        return out * (self.n ** 0.5) / (norm + 1e-8)


class Decoder(nn.Module):
    """Rx: n realnih vrednosti -> logiti za M poruka."""

    def __init__(self, M: int, n: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, M), nn.ReLU(), nn.Linear(M, M))

    def forward(self, x):
        return self.net(x)


class WirelessAutoencoder(nn.Module):
    def __init__(self, k: int = 4, n: int = 7, train_ebno_db: float = 3.0):
        super().__init__()
        self.k, self.n = k, n
        self.M = 2 ** k
        self.encoder = Encoder(self.M, n)
        self.channel = AWGNChannel(k, n, train_ebno_db)
        self.decoder = Decoder(self.M, n)

    def forward(self, x, ebno_db=None):
        return self.decoder(self.channel(self.encoder(x), ebno_db))


def load_model() -> WirelessAutoencoder:
    # Lokalni fajl u Space repou ima prioritet, inace skini sa model Hub-a
    path = CKPT_NAME if os.path.exists(CKPT_NAME) else hf_hub_download(
        repo_id=REPO_ID, filename=CKPT_NAME)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WirelessAutoencoder(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


MODEL = load_model()


# ----------------------------- baseline sistemi -----------------------------

def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


def bler_uncoded_qpsk(ebno_db_vec, k: int = 4):
    ebno = 10.0 ** (np.asarray(ebno_db_vec) / 10.0)
    return 1.0 - (1.0 - q_func(np.sqrt(2.0 * ebno))) ** k


P = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int64)
G = np.hstack([np.eye(4, dtype=np.int64), P])
MESSAGES_4BIT = np.array(
    [[(i >> b) & 1 for b in range(3, -1, -1)] for i in range(16)], dtype=np.int64)
CODEBOOK_BPSK = 1.0 - 2.0 * ((MESSAGES_4BIT @ G) % 2)


def bler_hamming_mld(ebno_db_vec, num_blocks: int = 20_000, seed: int = 42):
    """Simulirani Hamming(7,4) + QPSK sa ML dekodiranjem."""
    rng = np.random.default_rng(seed)
    R = 4.0 / 7.0
    out = []
    for ebno_db in ebno_db_vec:
        sigma = np.sqrt(1.0 / (2.0 * R * 10.0 ** (ebno_db / 10.0)))
        msg = rng.integers(0, 16, size=num_blocks)
        y = CODEBOOK_BPSK[msg] + rng.normal(0.0, sigma, size=(num_blocks, 7))
        d2 = ((y[:, None, :] - CODEBOOK_BPSK[None, :, :]) ** 2).sum(axis=2)
        out.append(float((d2.argmin(axis=1) != msg).mean()))
    return np.array(out)


# ----------------------------- evaluacija AE --------------------------------

@torch.no_grad()
def bler_autoencoder(model, ebno_db_vec, num_samples: int, batch_size: int = 5_000):
    out = []
    for ebno_db in ebno_db_vec:
        errors, total = 0, 0
        for _ in range(max(1, num_samples // batch_size)):
            messages = torch.randint(0, model.M, (batch_size,))
            oh = torch.zeros(batch_size, model.M)
            oh.scatter_(1, messages.unsqueeze(1), 1.0)
            pred = torch.argmax(model(oh, ebno_db=float(ebno_db)), dim=1)
            errors += (pred != messages).sum().item()
            total += batch_size
        out.append(errors / total)
    return np.array(out)


# ----------------------------- Gradio logika --------------------------------

def compute_demo(min_snr, max_snr, num_points, num_samples, show_baselines):
    ebno = np.linspace(float(min_snr), float(max_snr), int(num_points))
    bler_ae = bler_autoencoder(MODEL, ebno, int(num_samples))

    fig_bler, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(ebno, np.maximum(bler_ae, 1e-7), "b-o", lw=2,
                label="Autoencoder (7,4)")
    if show_baselines:
        ax.semilogy(ebno, bler_uncoded_qpsk(ebno), "k:",
                    label="Uncoded (4,4) QPSK (teorijski)")
        ax.semilogy(ebno, np.maximum(bler_hamming_mld(ebno), 1e-7), "r--s",
                    label="Hamming(7,4) + MLD (simulacija)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel("BLER")
    ax.set_title("BLER vs Eb/No")
    ax.legend()
    fig_bler.tight_layout()

    # Naucena konstelacija: PCA projekcija 7D -> 2D
    identity = torch.eye(MODEL.M)
    with torch.no_grad():
        cw = MODEL.encoder(identity).numpy()
    pca = PCA(n_components=2)
    pts = pca.fit_transform(cw)
    ev = pca.explained_variance_ratio_

    fig_const, ax2 = plt.subplots(figsize=(7, 7))
    for i in range(MODEL.M):
        ax2.scatter(*pts[i], s=160, c="steelblue", edgecolors="black", zorder=3)
        ax2.annotate(format(i, f"0{MODEL.k}b"), pts[i],
                     textcoords="offset points", xytext=(7, 7), fontsize=9)
    ax2.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax2.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% varijanse)")
    ax2.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% varijanse)")
    ax2.set_title("Naucena konstelacija (PCA projekcija 7D -> 2D)")
    fig_const.tight_layout()

    return fig_bler, fig_const


demo = gr.Interface(
    fn=compute_demo,
    inputs=[
        gr.Slider(-4, 2, value=0, step=0.5, label="Min Eb/No (dB)"),
        gr.Slider(4, 12, value=8, step=0.5, label="Max Eb/No (dB)"),
        gr.Slider(5, 25, value=17, step=1, label="Broj SNR tacaka"),
        gr.Slider(5_000, 50_000, value=20_000, step=5_000,
                  label="Broj test poruka po tacki"),
        gr.Checkbox(value=True, label="Prikazi baseline krive"),
    ],
    outputs=[
        gr.Plot(label="BLER vs Eb/No"),
        gr.Plot(label="Naucena konstelacija"),
    ],
    title="Wireless Autoencoder (7,4) — Interaktivni demo",
    description=(
        "End-to-end autoencoder za bezicni prenos preko AWGN kanala "
        "(MiSKS projekat, FTN UNS). Mreza je naucila kodiranje i modulaciju "
        "zajedno; BLER joj je na ~0.1 dB od ML dekodiranog Hamming(7,4) koda. "
        "Podesi opseg SNR-a i broj uzoraka pa pokreni."
    ),
)

if __name__ == "__main__":
    demo.launch()
