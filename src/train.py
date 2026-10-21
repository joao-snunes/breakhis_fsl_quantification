"""meta-treino do backbone.

Uma rodada = (semente, magnificação, divisão). A divisão "pac" serve ao P-PAC e ao P-MIX; a
"img", ao P-IMG. O treino ocorre sobre o backbone + cabeça de 4 saídas (classes
seen), em lotes. A cada época, a meta-validação mede o backbone com as mesmas 100 tarefas
(com cabeças novas), com early stopping. A época 0 é o
backbone só do ImageNet e também concorre.

Grava em results/train/{mag}_{divisao}_seed{S}/: config.json, epocas.csv e best_model.pt.
Este módulo nunca carrega imagens do meta-teste.

Uso: uv run python -m src.train   (uma rodada)
"""

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn

from src.adapt import avalia_episodio, gerador
from src.cache import carrega
from src.device import get_device
from src.episodes import ETAPA_ID, K_MVAL, N_MVAL, episodio
from src.model import D, cria_backbone, embeddings, normaliza
from src.splits import SEEN

LOTE = 64          # lote efetivo do meta-treino
LR = 1e-4          # AdamW do meta-treino
MAX_EPOCAS = 150
PACIENCIA = 10
DIVISOES = {"pac": ("etapa_pac", "P-PAC"), "img": ("etapa_img", "P-IMG")}  # coluna da divisão
METRICAS_MVAL = ["hm", "auc_seen", "auc_unseen", "acc_img", "acc_bal_pac", "frac_V1", "frac_V2"]
RESULTS = Path("results/train")


def _dados(split, coluna, mag):
    """Caminhos do mtrn (com o rótulo 0..3 na ordem de SEEN) e da mval. Nada do mtst."""
    na_mag = split[split.magnificacao == mag]
    mtrn, mval = na_mag[na_mag[coluna] == "mtrn"], na_mag[na_mag[coluna] == "mval"]
    assert not set(mtrn.path) & set(mval.path)
    assert (pd.concat([mtrn, mval])[coluna] != "mtst").all(), "imagem do meta-teste no treino" 
    return mtrn.path.tolist(), [SEEN.index(s) for s in mtrn.id_subtipo], mval.path.tolist()


def _epoca(modelo, opt, imgs, y, device, micro_batch_size, g):
    """Uma passada pelo mtrn: lotes de LOTE imagens, cada um em micro-lotes com gradientes somados."""
    modelo.train()
    ordem = torch.randperm(len(imgs), generator=g).numpy()
    soma = 0.0
    for i in range(0, len(ordem), LOTE):
        lote = ordem[i:i + LOTE]
        opt.zero_grad()
        for j in range(0, len(lote), micro_batch_size):
            m = lote[j:j + micro_batch_size]
            x = normaliza(imgs[m], modelo[0], device)
            # média do micro-lote × peso dele no lote = contribuição exata para a média do lote
            perda = nn.functional.binary_cross_entropy_with_logits(modelo(x), y[m].to(device)) * len(m) / len(lote)
            perda.backward()
            soma += perda.item() * len(lote)
        opt.step()
    return soma / len(imgs)


def _valida(backbone, imgs_mval, linha, tarefas, pacientes_mtrn, device, seed, mag, micro_batch_size):
    """Médias das métricas nas 100 tarefas fixas da mval."""
    E = embeddings(backbone, imgs_mval, device, micro_batch_size)
    linhas = [avalia_episodio(E, linha, classes, ep, gerador([seed, mag, ETAPA_ID["mval"], K_MVAL, i]), pacientes_mtrn)
              for i, (classes, ep) in enumerate(tarefas)]
    return pd.DataFrame(linhas)[METRICAS_MVAL].mean()


def git_estado():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        sujo = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        return "desconhecido", None
    return commit, sujo


def treina(seed, mag, divisao, micro_batch_size=16, max_memory_usage=0.75, allow_swap=False, max_epocas=MAX_EPOCAS):
    assert LOTE % micro_batch_size == 0, "micro_batch_size precisa dividir 64"
    coluna, protocolo = DIVISOES[divisao]
    pasta = RESULTS / f"{mag}_{divisao}_seed{seed}"
    pasta.mkdir(parents=True, exist_ok=True)
    device = get_device(max_memory_usage=max_memory_usage, allow_swap=allow_swap)
    torch.manual_seed(seed)  # inicialização da cabeça de 4 saídas

    split = pd.read_csv(f"data/splits/seed{seed}.csv")
    paths_mtrn, y_mtrn, paths_mval = _dados(split, coluna, mag)
    cache, linha_cache = carrega(mag)
    imgs_mtrn = cache[[linha_cache[p] for p in paths_mtrn]]
    imgs_mval = cache[[linha_cache[p] for p in paths_mval]]
    del cache  # libera a magnificação inteira; ficam só mtrn e mval
    y_mtrn = torch.eye(len(SEEN))[y_mtrn]
    linha_mval = {p: i for i, p in enumerate(paths_mval)}
    tarefas = [episodio(split, protocolo, "mval", K_MVAL, seed, mag, i) for i in range(N_MVAL)] 
    pacientes_mtrn = set(split.id_paciente[split[coluna] == "mtrn"])

    commit, sujo = git_estado()
    config = {
        "seed": seed, "mag": mag, "divisao": divisao, "protocolo_mval": protocolo, "device": str(device),
        "micro_batch_size": micro_batch_size, "max_memory_usage": max_memory_usage, "allow_swap": allow_swap,
        "lote": LOTE, "lr": LR, "max_epocas": max_epocas, "paciencia": PACIENCIA, "k_mval": K_MVAL, "n_mval": N_MVAL,
        "n_mtrn": len(paths_mtrn), "n_mval_imagens": len(paths_mval),
        "versoes": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__},
        "git": {"commit": commit, "alteracoes_nao_commitadas": sujo}, "inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (pasta / "config.json").write_text(json.dumps(config, indent=2))

    backbone = cria_backbone().to(device)
    modelo = nn.Sequential(backbone, nn.Linear(D, len(SEEN))).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR)

    melhor, sem_melhora, epocas = -1.0, 0, []
    for epoca in range(max_epocas + 1):
        t = time.time()
        perda = _epoca(modelo, opt, imgs_mtrn, y_mtrn, device, micro_batch_size,
                       gerador([seed, mag, list(DIVISOES).index(divisao), epoca])) if epoca else float("nan")  # época 0: só ImageNet (D38)
        val = _valida(backbone, imgs_mval, linha_mval, tarefas, pacientes_mtrn, device, seed, mag, micro_batch_size)
        melhorou = val.hm > melhor
        if melhorou:  # 5c-1: a época 0 também concorre
            melhor, sem_melhora = val.hm, 0
            torch.save({k: v.cpu() for k, v in backbone.state_dict().items()}, pasta / "best_model.pt")
        else:
            sem_melhora += 1
        epocas.append({"epoca": epoca, "perda_mtrn": perda, **val, "melhor": melhorou, "segundos": time.time() - t})
        pd.DataFrame(epocas).to_csv(pasta / "epocas.csv", index=False)  # regravado a cada época: dá para acompanhar
        print(f"época {epoca:3d} | perda {perda:.4f} | HM mval {val.hm:.1f} | {'melhor' if melhorou else sem_melhora} | {time.time() - t:.0f}s", flush=True)
        if sem_melhora == PACIENCIA:
            break

    melhor_epoca = int(pd.DataFrame(epocas).query("melhor").epoca.max())
    config |= {"fim": time.strftime("%Y-%m-%d %H:%M:%S"), "epocas_rodadas": epoca, "melhor_epoca": melhor_epoca, "melhor_hm": melhor}
    (pasta / "config.json").write_text(json.dumps(config, indent=2))  # "fim" marca a rodada como completa
    return pasta


if __name__ == "__main__":
    SEED, MAG, DIVISAO = 0, 40, "pac"
    treina(SEED, MAG, DIVISAO)
