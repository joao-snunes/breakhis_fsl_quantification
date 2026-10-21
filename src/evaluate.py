"""Nota final no meta-teste.

Uma rodada = (semente, magnificação), com os três protocolos juntos, em episódios são
pareados. Faz 2 treinos da rodada completos (results/train/...):
    divisão "img" → P-IMG          divisão "pac" → P-MIX e P-PAC

Dois backbones por protocolo (5d-1):
    meta      o best_model.pt do meta-treino da divisão
    imagenet  o backbone só do ImageNet, que nunca viu paciente nenhum (sem V2)

Grava em results/:
    mtst/{mag}_seed{S}.csv              uma linha por episódio (backbone × protocolo × k)
    mtst/{mag}_seed{S}_pacientes.csv.gz auditoria: pacientes de cada papel em cada episódio
    mtst/{mag}_seed{S}_config.json      configuração; o campo "fim" marca a rodada completa
    embeddings/{mag}_{backbone}_{divisao}_seed{S}.npz   embeddings do mtst (D47)

Uso: uv run python -m src.evaluate   (uma rodada, definida pelas constantes do __main__)
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

from src import adapt
from src.adapt import avalia_episodio, gerador
from src.cache import carrega
from src.device import get_device
from src.episodes import ETAPA_ID, K_SHOTS, K_TST, N_MTST, PROTOCOLOS, episodio
from src.model import NOME, cria_backbone, embeddings
from src.train import DIVISOES, git_estado
from src.train import RESULTS as TREINOS

RESULTS = Path("results")
BACKBONES = ["meta", "imagenet"]
DIVISAO = {coluna: divisao for divisao, (coluna, _) in DIVISOES.items()}  # etapa_img → img, etapa_pac → pac


def _backbone_treinado(seed, mag, divisao, device):
    pasta = TREINOS / f"{mag}_{divisao}_seed{seed}"
    config = json.loads((pasta / "config.json").read_text())
    assert "fim" in config, f"treino incompleto: {pasta}"
    backbone = cria_backbone(pretrained=False)
    backbone.load_state_dict(torch.load(pasta / "best_model.pt"))
    return backbone.to(device), config


def avalia(seed, mag, batch_size=16, max_memory_usage=0.75, allow_swap=False):
    device = get_device(max_memory_usage=max_memory_usage, allow_swap=allow_swap)
    pasta = RESULTS / "mtst"
    pasta.mkdir(parents=True, exist_ok=True)
    (RESULTS / "embeddings").mkdir(parents=True, exist_ok=True)
    nome = f"{mag}_seed{seed}"

    split = pd.read_csv(f"data/splits/seed{seed}.csv")
    na_mag = split[split.magnificacao == mag]
    paths = {div: na_mag.path[na_mag[coluna] == "mtst"].tolist() for div, (coluna, _) in DIVISOES.items()}
    cache, linha_cache = carrega(mag)

    # embeddings do mtst: E[(backbone, divisao)] e o índice path → linha de cada divisão
    treinos, E = {}, {}
    imagenet = cria_backbone().to(device)
    for div in DIVISOES:
        imgs = cache[[linha_cache[p] for p in paths[div]]]
        meta, treinos[div] = _backbone_treinado(seed, mag, div, device)
        E[("meta", div)] = embeddings(meta, imgs, device, batch_size)
        E[("imagenet", div)] = embeddings(imagenet, imgs, device, batch_size)
        for backbone in BACKBONES:
            np.savez(RESULTS / "embeddings" / f"{mag}_{backbone}_{div}_seed{seed}.npz",
                     embeddings=E[(backbone, div)].numpy(), paths=np.array(paths[div]))
    del cache
    linha = {div: {p: i for i, p in enumerate(paths[div])} for div in DIVISOES}

    commit, sujo = git_estado()
    config = {
        "seed": seed, "mag": mag, "device": str(device), "backbones": BACKBONES, "modelo": NOME,
        "k_shots": K_SHOTS, "k_tst": K_TST, "n_mtst": N_MTST,
        "adaptacao": {"passos": adapt.PASSOS, "lr": adapt.LR, "frac": adapt.FRAC},
        "treinos": {div: {c: treinos[div].get(c) for c in ("melhor_epoca", "melhor_hm", "epocas_rodadas", "device", "git")}
                    for div in DIVISOES},
        "versoes": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__},
        "git": {"commit": commit, "alteracoes_nao_commitadas": sujo}, "inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (pasta / f"{nome}_config.json").write_text(json.dumps(config, indent=2))

    linhas, auditoria = [], []
    for protocolo, (coluna, _) in PROTOCOLOS.items():
        div = DIVISAO[coluna]
        # o backbone do ImageNet não treinou com paciente nenhum: sem exposição V2
        pacientes_mtrn = {"meta": set(split.id_paciente[split[coluna] == "mtrn"]), "imagenet": set()}
        for k in K_SHOTS:
            t = time.time()
            for i in range(N_MTST):
                classes, ep = episodio(split, protocolo, "mtst", k, seed, mag, i)
                base = {"magnificacao": mag, "semente": seed, "protocolo": protocolo, "k_shot": k, "episodio": i,
                        "classes_seen": "+".join(classes[:2]), "classes_unseen": classes[2]}
                for backbone in BACKBONES:  # o mesmo episódio e a mesma cabeça inicial nos dois backbones
                    m = avalia_episodio(E[(backbone, div)], linha[div], classes, ep,
                                        gerador([seed, mag, ETAPA_ID["mtst"], k, i]), pacientes_mtrn[backbone])
                    linhas.append({**base, "backbone": backbone, **m})
                aud = ep.groupby(["id_subtipo", "papel", "id_paciente"]).size().rename("n_imagens").reset_index()
                auditoria.append(aud.assign(protocolo=protocolo, k_shot=k, episodio=i))
            print(f"{protocolo} k={k}: {N_MTST} episódios em {time.time() - t:.0f}s", flush=True)

    pd.DataFrame(linhas).to_csv(pasta / f"{nome}.csv", index=False)
    pd.concat(auditoria).to_csv(pasta / f"{nome}_pacientes.csv.gz", index=False)
    config["fim"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (pasta / f"{nome}_config.json").write_text(json.dumps(config, indent=2))
    return pasta / f"{nome}.csv"


if __name__ == "__main__":
    SEED, MAG = 0, 40
    avalia(SEED, MAG)
