"""Etapa 5d: testes da avaliação no meta-teste."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import evaluate
from src.adapt import avalia_episodio, gerador
from src.cache import CACHE, carrega
from src.device import get_device
from src.episodes import episodio
from src.model import D, cria_backbone, embeddings

SPLIT = Path("data/splits/seed0.csv")
pytestmark = pytest.mark.skipif(not SPLIT.exists(), reason="rode python -m src.splits antes")


def test_avalia_formato_e_vazamento(tmp_path, monkeypatch):
    """Treinos falsos e embeddings aleatórios: confere as saídas, o pareamento e o P-PAC sem vazamento."""
    split = pd.read_csv(SPLIT)
    paths = split.path[split.magnificacao == 40].tolist()
    for div in evaluate.DIVISOES:  # treinos "completos" falsos
        pasta = tmp_path / "train" / f"40_{div}_seed0"
        pasta.mkdir(parents=True)
        (pasta / "config.json").write_text(json.dumps({"fim": "x", "melhor_epoca": 3, "melhor_hm": 70.0}))
        torch.save(cria_backbone(pretrained=False).state_dict(), pasta / "best_model.pt")
    g = torch.Generator().manual_seed(0)
    monkeypatch.setattr(evaluate, "TREINOS", tmp_path / "train")
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "N_MTST", 3)
    monkeypatch.setattr(evaluate, "K_SHOTS", [1, 5])
    monkeypatch.setattr(evaluate, "cria_backbone", lambda pretrained=True: cria_backbone(pretrained=False))
    monkeypatch.setattr(evaluate, "carrega", lambda mag: (np.zeros((len(paths), 1, 1, 3), np.uint8), {p: i for i, p in enumerate(paths)}))
    monkeypatch.setattr(evaluate, "embeddings", lambda b, imgs, device, bs: torch.randn(len(imgs), D, generator=g))

    df = pd.read_csv(evaluate.avalia(0, 40))
    assert len(df) == 2 * 3 * 2 * 3  # backbones × protocolos × k × episódios
    assert (df[df.protocolo == "P-PAC"][["frac_V1", "frac_V2"]] == 0).all().all()
    assert (df[df.backbone == "imagenet"].frac_V2 == 0).all()
    assert df[(df.protocolo == "P-IMG") & (df.backbone == "meta")].frac_V2.mean() > 0
    assert (df.groupby(["k_shot", "episodio"])[["classes_seen", "classes_unseen"]].nunique() == 1).all().all()  # pareado

    aud = pd.read_csv(tmp_path / "mtst" / "40_seed0_pacientes.csv.gz")
    pac = aud[aud.protocolo == "P-PAC"]
    assert not (pac.groupby(["k_shot", "episodio", "id_paciente"]).papel.nunique() > 1).any()  # regra 1
    config = json.loads((tmp_path / "mtst" / "40_seed0_config.json").read_text())
    assert "fim" in config and config["treinos"]["pac"]["melhor_epoca"] == 3
    assert len(list((tmp_path / "embeddings").glob("*.npz"))) == 4


@pytest.mark.slow
@pytest.mark.skipif(not (CACHE / "40.npz").exists(), reason="rode python -m src.cache antes")
def test_rotulos_embaralhados_auc_50():  # teste de sanidade 6, com embeddings reais (backbone do ImageNet)
    split = pd.read_csv(SPLIT)
    paths = split.path[(split.magnificacao == 40) & (split.etapa_pac == "mtst")].tolist()
    cache, linha_cache = carrega(40)
    device = get_device()
    E = embeddings(cria_backbone().to(device), cache[[linha_cache[p] for p in paths]], device)
    linha = {p: i for i, p in enumerate(paths)}
    rng = np.random.default_rng(0)
    hms = []
    for i in range(100):
        classes, ep = episodio(split, "P-PAC", "mtst", 5, 0, 40, i)
        q = ep.papel == "Q"
        ep.loc[q, "id_subtipo"] = rng.permutation(ep.loc[q, "id_subtipo"].to_numpy())  # rótulos do query sem relação com a imagem
        hms.append(avalia_episodio(E, linha, classes, ep, gerador([i]), set())["hm"])
    assert 40 < np.mean(hms) < 60, f"HM média {np.mean(hms):.1f}"
