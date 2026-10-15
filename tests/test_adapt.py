"""Etapa 5b: testes de model.py e adapt.py com dados sintéticos (rápidos, sem cache nem GPU)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.adapt import adapta, gerador, metricas
from src.model import D, cria_backbone, embeddings, normaliza


def episodio_sintetico(separacao, seed, k=5, k_tst=10):
    """3 classes; embeddings = centro da classe × separacao + ruído. separacao=0 → rótulos sem relação com X."""
    g = torch.Generator().manual_seed(seed)
    centros = torch.randn(3, D, generator=g)
    def lado(n):
        y = torch.eye(3).repeat_interleave(n, 0)
        return y @ centros * separacao + torch.randn(3 * n, D, generator=g), y
    (es, ys), (eq, yq) = lado(k), lado(k_tst)
    return es, ys, eq, yq


def test_normaliza():  # 0 → -1 e 255 → +1 (média 0,5 e desvio 0,5 do pré-treino)
    imgs = np.zeros((2, 4, 6, 3), np.uint8)
    imgs[1] = 255
    x = normaliza(imgs, cria_backbone(pretrained=False), "cpu")
    assert x.shape == (2, 3, 4, 6) and x[0].eq(-1).all() and x[1].eq(1).all()


def test_embeddings():  # formato N × 1024, determinístico e igual com micro-lotes diferentes
    backbone = cria_backbone(pretrained=False)
    imgs = np.random.default_rng(0).integers(0, 256, (3, 456, 700, 3), dtype=np.uint8)
    a, b = embeddings(backbone, imgs, "cpu", batch_size=1), embeddings(backbone, imgs, "cpu", batch_size=3)
    assert a.shape == (3, D) and torch.allclose(a, b, atol=1e-5)


def test_separavel_auc_alta():
    es, ys, eq, yq = episodio_sintetico(separacao=1.0, seed=0)
    probs = adapta(es, ys, eq, gerador([0]))
    m = metricas(["DC", "F", "PT"], yq, probs, pd.DataFrame({"id_paciente": ["p"] * 30}), set(), set())
    assert m["auc_seen"] > 95 and m["auc_unseen"] > 95 and m["acc_img"] > 90


def test_rotulos_aleatorios_auc_50():  # teste de sanidade 6, no nível da cabeça
    hms = []
    for s in range(30):
        es, ys, eq, yq = episodio_sintetico(separacao=0.0, seed=s)
        m = metricas(["DC", "F", "PT"], yq, adapta(es, ys, eq, gerador([s])), pd.DataFrame({"id_paciente": ["p"] * 30}), set(), set())
        hms.append(m["hm"])
    assert 40 < np.mean(hms) < 60


def test_reprodutivel():  # 5b-2: mesma chave → mesma cabeça → mesmas probabilidades
    es, ys, eq, _ = episodio_sintetico(separacao=0.5, seed=1)
    assert torch.equal(adapta(es, ys, eq, gerador([0, 100, 2, 5, 7])), adapta(es, ys, eq, gerador([0, 100, 2, 5, 7])))
    assert not torch.equal(adapta(es, ys, eq, gerador([0, 100, 2, 5, 7])), adapta(es, ys, eq, gerador([0, 100, 2, 5, 8])))


def test_metricas_a_mao():
    # 6 imagens no query, classes 0,0,0,0,1,2; previsões: acerta 3 das 4 da classe 0 e as outras duas
    y = torch.eye(3)[[0, 0, 0, 0, 1, 2]]
    probs = torch.eye(3)[[0, 0, 0, 1, 1, 2]] * 0.8 + 0.1
    qry = pd.DataFrame({"id_paciente": ["a", "a", "b", "b", "c", "d"]})
    m = metricas(["DC", "F", "PT"], y, probs, qry, pacientes_sup={"a"}, pacientes_mtrn={"a", "c"})
    assert m["acc_img"] == pytest.approx(5 / 6 * 100)
    # classe 0: paciente a = 1,0; b = 0,5 → 0,75; classes 1 e 2: 1,0 → média (0,75 + 1 + 1) / 3
    assert m["acc_bal_pac"] == pytest.approx((0.75 + 1 + 1) / 3 * 100)
    assert m["frac_V1"] == pytest.approx(2 / 6) and m["frac_V2"] == pytest.approx(3 / 6)
