"""
adapta:   cabeça linear NOVA (1.024 → 3) treinada só no support, sobre embeddings fixos.
          Linear + sigmoide + BCE = 3 regressões logísticas one-vs-rest.
metricas: a linha de resultado do episódio (AUC seen/unseen, HM, acurácias, frac_V1/V2).

Roda na CPU.
"""

import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics.functional.classification import binary_auroc, multilabel_auroc

from src.model import D

PASSOS = 100  # passos de AdamW por episódio
LR = 0.005   
FRAC = 0.5    # fração do support em cada passo


def gerador(chave):
    """Gerador do torch derivado da chave do episódio."""
    semente = np.random.SeedSequence(chave).spawn(1)[0].generate_state(1)[0]
    return torch.Generator().manual_seed(int(semente))


def _cabeca_nova(n_classes, g):
    """nn.Linear com a inicialização padrão do PyTorch, mas sorteada pelo gerador do episódio."""
    cabeca = nn.Linear(D, n_classes)
    with torch.no_grad():
        nn.init.kaiming_uniform_(cabeca.weight, a=math.sqrt(5), generator=g)
        nn.init.uniform_(cabeca.bias, -1 / math.sqrt(D), 1 / math.sqrt(D), generator=g)
    return cabeca


def adapta(emb_sup, y_sup, emb_qry, g):
    """Treina a cabeça no support e devolve as probabilidades (sigmoide) do query."""
    cabeca = _cabeca_nova(y_sup.shape[1], g)
    opt = torch.optim.AdamW(cabeca.parameters(), lr=LR)
    perda = nn.BCEWithLogitsLoss()
    n = len(emb_sup)
    lote = max(1, int(FRAC * n))  # D37: com k = 1 (n = 3), 1 imagem por passo, como no original
    for _ in range(PASSOS):
        idx = torch.randperm(n, generator=g)[:lote]
        opt.zero_grad()
        perda(cabeca(emb_sup[idx]), y_sup[idx]).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(cabeca(emb_qry))


def _auroc(y, p):
    """AUC em %; colunas = classes. Igual ao base.py do meta-cxr (média micro)."""
    if y.shape[1] == 1:
        return binary_auroc(p.view(-1), y.view(-1)).item() * 100
    return multilabel_auroc(p, y, num_labels=y.shape[1], average="micro").item() * 100


def metricas(classes, y, probs, qry, pacientes_sup, pacientes_mtrn):
    """Linha de resultado do episódio.

    classes: [seen, seen, unseen] (a ordem das colunas de y e probs)
    y, probs: (n_query, 3), rótulos one-hot e probabilidades
    qry: DataFrame do query, na mesma ordem das linhas, com a coluna id_paciente
    pacientes_sup / pacientes_mtrn: pacientes do support e do meta-treino da divisão usada
    """
    y = y.int()
    seen, unseen = _auroc(y[:, :2], probs[:, :2]), _auroc(y[:, 2:], probs[:, 2:])
    certo = (probs.argmax(1) == y.argmax(1)).numpy()
    por_pac = pd.DataFrame({"classe": y.argmax(1).numpy(), "paciente": qry.id_paciente.to_numpy(), "certo": certo})
    linha = {
        "auc_seen": seen,
        "auc_unseen": unseen,
        "hm": 2 * seen * unseen / (seen + unseen),
        "acc_img": certo.mean() * 100,  # query balanceado: igual à acurácia balanceada por imagem (5b-3)
        # acerto de cada paciente → média por classe → média das classes
        "acc_bal_pac": por_pac.groupby(["classe", "paciente"]).certo.mean().groupby("classe").mean().mean() * 100,
        "frac_V1": qry.id_paciente.isin(pacientes_sup).mean(),
        "frac_V2": qry.id_paciente.isin(pacientes_mtrn).mean(),
    }
    por_classe = multilabel_auroc(probs, y, num_labels=3, average="none") * 100
    linha |= {f"auc_{c}": v.item() for c, v in zip(classes, por_classe)}
    return linha


def avalia_episodio(E, linha, classes, ep, g, pacientes_mtrn):
    """Adapta e mede um episódio a partir de embeddings já calculados.

    E: embeddings (N, 1024); linha: path → linha de E; classes, ep: saída de episodes.episodio;
    g: gerador do episódio; pacientes_mtrn: pacientes do meta-treino da divisão usada.
    """
    s, q = ep[ep.papel == "S"], ep[ep.papel == "Q"]
    rotulos = lambda d: torch.eye(len(classes))[[classes.index(c) for c in d.id_subtipo]]
    probs = adapta(E[[linha[p] for p in s.path]], rotulos(s), E[[linha[p] for p in q.path]], g)
    return metricas(classes, rotulos(q), probs, q, set(s.id_paciente), pacientes_mtrn)
