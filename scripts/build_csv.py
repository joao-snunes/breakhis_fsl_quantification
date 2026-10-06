"""Gera data/breakhis.csv a partir dos nomes dos arquivos do BreakHis.

Formato do nome (README da base):
    SOB_M_DC-14-2523-40-010.png → procedimento SOB, classe M, subtipo DC,
    ano 14, lâmina 2523 (+ letras de seção opcionais, ex. 22549AB),
    magnificação 40, sequência 010.

Colunas:
    path          caminho relativo a data/
    classe        B (benigno) ou M (maligno)
    subtipo       A, F, PT, TA, DC, LC, MC, PC
    paciente      ID oficial, igual ao nome da pasta (ex. 14-22549AB) → 82 pacientes
    caso          ID sem as letras de seção (ex. 14-22549); agrupa pastas que
                  compartilham o número da lâmina
    magnificacao  40, 100, 200, 400
    seq           sequência da imagem
"""

import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
PATTERN = re.compile(
    r"SOB_(?P<classe>[BM])_(?P<subtipo>[A-Z]+)-(?P<paciente>(?P<caso>\d+-\d+)[A-Z]*)"
    r"-(?P<magnificacao>40|100|200|400)-(?P<seq>\d+)\.png"
)


def build(root: Path = DATA / "BreaKHis_v1") -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.png")):
        m = PATTERN.fullmatch(p.name)
        assert m, f"nome fora do padrão: {p.name}"
        # a pasta do paciente (ex. SOB_B_A_14-22549AB) deve concordar com o nome do arquivo
        assert p.parent.parent.name.endswith(m["paciente"]), p
        assert p.parent.name == m["magnificacao"] + "X", p
        rows.append({"path": str(p.relative_to(DATA)), **m.groupdict()})
    df = pd.DataFrame(rows)
    df["magnificacao"] = df["magnificacao"].astype(int)
    df["seq"] = df["seq"].astype(int)
    return df[["path", "classe", "subtipo", "paciente", "caso", "magnificacao", "seq"]]


if __name__ == "__main__":
    df = build()
    df.to_csv(DATA / "breakhis.csv", index=False)
    print(f"{len(df)} imagens, {df.paciente.nunique()} pacientes, {df.caso.nunique()} casos")
