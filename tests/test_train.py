"""Etapa 5c: testes do meta-treino. Os marcados com slow treinam de verdade: uv run pytest -m slow."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from src import train
from src.cache import CACHE, carrega
from src.device import get_device
from src.model import D, cria_backbone
from src.splits import SEEN

SPLIT = Path("data/splits/seed0.csv")
pytestmark = pytest.mark.skipif(not SPLIT.exists() or not (CACHE / "40.npz").exists(),
                                reason="rode python -m src.splits e python -m src.cache antes")


def test_mtst_nunca_carregado():  # regra 2: o treino só enxerga mtrn e mval
    split = pd.read_csv(SPLIT)
    for coluna, _ in train.DIVISOES.values():
        for mag in [40, 100, 200, 400]:
            mtrn, _, mval = train._dados(split, coluna, mag)
            etapas = split.set_index("path").loc[mtrn + mval, coluna]
            assert set(etapas) == {"mtrn", "mval"}


@pytest.mark.slow
def test_overfit_proposital():  # teste de sanidade 7: com ~20 imagens, a perda vai a quase zero
    split = pd.read_csv(SPLIT)
    mtrn, y, _ = train._dados(split, "etapa_pac", 40)
    cache, linha = carrega(40)
    escolhidas = list(range(0, len(mtrn), len(mtrn) // 20))[:20]
    imgs = cache[[linha[mtrn[i]] for i in escolhidas]]
    y = torch.eye(len(SEEN))[[y[i] for i in escolhidas]]
    device = get_device()
    torch.manual_seed(0)
    modelo = nn.Sequential(cria_backbone(), nn.Linear(D, len(SEEN))).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3)  # lr maior só para decorar rápido
    for epoca in range(100):
        perda = train._epoca(modelo, opt, imgs, y, device, micro_batch_size=8, g=torch.Generator().manual_seed(epoca))
        if perda < 0.05:
            break
    assert perda < 0.05, f"a perda não caiu: {perda:.3f}"


@pytest.mark.slow
def test_ponta_a_ponta(tmp_path, monkeypatch):  # 2 épocas reais: gera os 3 arquivos no formato certo
    monkeypatch.setattr(train, "RESULTS", tmp_path)
    pasta = train.treina(0, 40, "pac", max_epocas=2)
    epocas = pd.read_csv(pasta / "epocas.csv")
    config = json.loads((pasta / "config.json").read_text())
    assert epocas.epoca.tolist() == [0, 1, 2] and epocas.perda_mtrn.iloc[1:].notna().all()
    assert set(train.METRICAS_MVAL) <= set(epocas.columns)
    assert "fim" in config and config["melhor_epoca"] in [0, 1, 2]
    estado = torch.load(pasta / "best_model.pt")
    assert set(estado) == set(cria_backbone(pretrained=False).state_dict())  # só o backbone, sem a cabeça de 4 saídas
