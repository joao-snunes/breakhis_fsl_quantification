"""Etapa 5a: o cache tem todas as imagens da magnificação, no formato certo e alinhadas aos caminhos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cache import ALTURA, CACHE, LARGURA, MAGS, _leitura, carrega

pytestmark = pytest.mark.skipif(not (CACHE / "40.npz").exists(), reason="rode python -m src.cache antes")


def test_cache():
    df = pd.read_csv(Path("data/breakhis.csv"))
    for mag in MAGS:
        imgs, linha = carrega(mag)
        assert imgs.shape == (sum(df.magnificacao == mag), ALTURA, LARGURA, 3) and imgs.dtype == np.uint8
        assert set(linha) == set(df.path[df.magnificacao == mag])  # nenhuma imagem faltando ou de outra magnificação
        for p in list(linha)[::500]:  # o array está na ordem dos caminhos
            assert np.array_equal(imgs[linha[p]], _leitura(p))
