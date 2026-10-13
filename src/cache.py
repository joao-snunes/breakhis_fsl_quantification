"""Cache das imagens na resolução original.

Cada magnificação vira um arquivo data/cache/{mag}.npz com:
    imgs   array uint8 (N, 456, 700, 3)
    paths  os caminhos (iguais ao data/breakhis.csv), na mesma ordem das imagens

O BreakHis tem 7.384 imagens de 700×460 e 74 de 700×456. Todas são salvas como 700×456.


Gerado uma vez (uv run python -m src.cache); depois, train e evaluate só carregam.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path("data")
CACHE = DATA / "cache"
LARGURA, ALTURA = 700, 456
MAGS = [40, 100, 200, 400]


def _leitura(path):
    img = np.asarray(Image.open(DATA / path).convert("RGB"))
    corte = (img.shape[0] - ALTURA) // 2  # 2 nas imagens de 460, 0 nas de 456
    return img[corte:corte + ALTURA]


def _geracao(mag):
    df = pd.read_csv(DATA / "breakhis.csv")
    paths = sorted(df.path[df.magnificacao == mag])
    imgs = np.stack([_leitura(p) for p in paths])
    np.savez(CACHE / f"{mag}.npz", imgs=imgs, paths=np.array(paths))
    return imgs


def carrega(mag):
    """Devolve (imgs, linha): o array uint8 e um dicionário path → linha do array."""
    arq = np.load(CACHE / f"{mag}.npz")
    return arq["imgs"], {p: i for i, p in enumerate(arq["paths"])}


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    for mag in MAGS:
        if (CACHE / f"{mag}.npz").exists():
            print(f"{mag}×: já existe")
            continue
        imgs = _geracao(mag)
        assert imgs.shape[1:] == (ALTURA, LARGURA, 3) and imgs.dtype == np.uint8
        print(f"{mag}×: {imgs.shape[0]} imagens, {imgs.nbytes / 1e9:.2f} GB")
