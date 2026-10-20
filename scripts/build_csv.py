"""Gera data/breakhis_raw.csv: uma linha por imagem do BreakHis, sem nenhum tratamento.

Formato do nome (README da base):
    SOB_B_A-14-22549AB-100-001.png → procedimento SOB, malignidade B, subtipo A,
    registro 14-22549, seção AB, magnificação 100, sequência 001.

Colunas:
    path          caminho relativo a data/
    malignidade   benign ou malignant
    id_subtipo    A, F, PT, TA, DC, LC, MC, PC
    des_subtipo   nome da pasta do subtipo (ex. ductal_carcinoma)
    id_paciente   registro do exame (ex. 14-22549)
    secao         letras de seção (ex. AB; vazio quando não há)
    magnificacao  40, 100, 200, 400
    seq           sequência da imagem
    rgb_hash      MD5 dos pixels RGB: imagens iguais têm o mesmo hash mesmo com
                  arquivos (bytes) diferentes

O tratamento (duplicatas etc.) fica em notebooks/02_clear_csv.ipynb.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path("data")
PATTERN = re.compile(
    r"SOB_(?P<malignidade>[BM])_(?P<id_subtipo>[A-Z]+)-(?P<id_paciente>\d+-\d+)(?P<secao>[A-Z]*)"
    r"-(?P<magnificacao>40|100|200|400)-(?P<seq>\d+)\.png"
)
MALIGNIDADE = {"B": "benign", "M": "malignant"}


def build(root: Path = DATA / "BreaKHis_v1") -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.png")):
        m = PATTERN.fullmatch(p.name)
        assert m, f"nome fora do padrão: {p.name}"
        # .../<malignidade>/SOB/<des_subtipo>/SOB_B_A_14-22549AB/100X/<arquivo>: pastas devem concordar com o nome
        mag_dir, pac_dir, sub_dir, _, mal_dir = p.parents[:5]
        assert mag_dir.name == m["magnificacao"] + "X", p
        assert pac_dir.name == f"SOB_{m['malignidade']}_{m['id_subtipo']}_{m['id_paciente']}{m['secao']}", p
        assert mal_dir.name == MALIGNIDADE[m["malignidade"]], p
        rows.append({
            "path": p.relative_to(DATA).as_posix(),
            "malignidade": MALIGNIDADE[m["malignidade"]],
            "id_subtipo": m["id_subtipo"],
            "des_subtipo": sub_dir.name,
            "id_paciente": m["id_paciente"],
            "secao": m["secao"],
            "magnificacao": int(m["magnificacao"]),
            "seq": int(m["seq"]),
            "rgb_hash": hashlib.md5(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv(DATA / "breakhis_raw.csv", index=False)
    print(f"{len(df)} imagens -> data/breakhis_raw.csv")
