"""divide as imagens em meta-treino (mtrn), meta-validação (mval) e meta-teste (mtst).

Para cada semente, grava data/splits/seed{S}.csv com duas colunas de etapa:

- etapa_pac (P-PAC e P-MIX): sorteia PACIENTES. Todas as imagens de um paciente ficam
  na mesma etapa, nas 4 magnificações.
- etapa_img (P-IMG): sorteia IMAGENS soltas, com os mesmos tamanhos por subtipo,
  magnificação e etapa. Equivale ao train_test_split sem grupo.

Classes unseen não são sorteadas: vão inteiras para a sua etapa.

Uso: uv run python -m src.splits
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [0, 10, 42, 109, 2026]
FRAC = 0.2          # fração de pacientes de cada classe seen em mval e em mtst
MIN_PACIENTES = 2   # mínimo em mval e em mtst (o F fica 1/2/2)
SEEN = ["DC", "F", "MC", "TA"]
UNSEEN = {"mval": ["A", "LC"], "mtst": ["PT", "PC"]}


def sorteia_pacientes(df, rng):
    """Permuta pacientes das classes SEEN e distribui entre mtrn, mval e mtst.
    Marca pacientes das classes UNSEEN para suas respectivas etapas.
    """
    etapa = {}
    for subtipo in SEEN:
        pacientes = rng.permutation(sorted(df.loc[df.id_subtipo == subtipo, "id_paciente"].unique()))
        n = max(MIN_PACIENTES, round(FRAC * len(pacientes)))  # pacientes em mval e em mtst
        for i, p in enumerate(pacientes):
            etapa[p] = "mval" if i < n else "mtst" if i < 2 * n else "mtrn"
    for e, subtipos in UNSEEN.items():
        for p in df.loc[df.id_subtipo.isin(subtipos), "id_paciente"]:
            etapa[p] = e
    return etapa


def sorteia_imagens(df, rng):
    """Gera coluna P_IMG usada para a etapa onde é desconsiderado o paciente.
    Embaralha os rótulos de etapa_pac dentro de cada (subtipo seen, magnificação).
    Os tamanhos por etapa continuam os mesmos;
    """
    etapa_img = df.etapa_pac.copy()
    for _, grupo in df[df.id_subtipo.isin(SEEN)].groupby(["id_subtipo", "magnificacao"]):
        etapa_img[grupo.index] = grupo.etapa_pac.sample(frac=1, random_state=rng).to_numpy()
    return etapa_img


def split(df, seed):
    rng = np.random.default_rng(seed)
    out = df[["path", "id_paciente", "id_subtipo", "magnificacao"]].copy()
    out["etapa_pac"] = out.id_paciente.map(sorteia_pacientes(df, rng))
    out["etapa_img"] = sorteia_imagens(out, rng)

    # regra 1: nenhum paciente em duas etapas
    assert (out.groupby("id_paciente").etapa_pac.nunique() == 1).all(), "paciente em duas etapas"
    # regra 4: mesmos tamanhos por subtipo, magnificação e etapa nas duas divisões
    tamanhos = lambda col: out.groupby(["id_subtipo", "magnificacao", col]).size().to_numpy()
    assert (tamanhos("etapa_pac") == tamanhos("etapa_img")).all(), "tamanhos diferentes"
    return out


if __name__ == "__main__":
    df = pd.read_csv("data/breakhis.csv", keep_default_na=False)
    Path("data/splits").mkdir(exist_ok=True)
    for seed in SEEDS:
        out = split(df, seed)
        out.to_csv(f"data/splits/seed{seed}.csv", index=False)
        pacientes = out.drop_duplicates("id_paciente").query("id_subtipo in @SEEN")
        print(f"seed {seed}:", pd.crosstab(pacientes.id_subtipo, pacientes.etapa_pac).to_dict("index"))
