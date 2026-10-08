"""Etapa 3: testes de sanidade das divisões (PROJETO.md, seção 12) sobre o CSV limpo real."""

from pathlib import Path

import pandas as pd
import pytest

from src.splits import SEEN, UNSEEN, split

CSV = Path("data/breakhis.csv")
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="rode build_csv.py e 02_clear_csv.ipynb antes")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV, keep_default_na=False)


@pytest.fixture(scope="module")
def out(df):
    return split(df, seed=0)


def test_paciente_em_uma_etapa_so(out):  # teste 1 (vale nas 4 magnificações juntas)
    assert (out.groupby("id_paciente").etapa_pac.nunique() == 1).all()


def test_tamanhos_pareados(out):  # teste 3
    pac = out.groupby(["id_subtipo", "magnificacao", "etapa_pac"]).size()
    img = out.groupby(["id_subtipo", "magnificacao", "etapa_img"]).size()
    assert pac.to_dict() == img.to_dict()


def test_cada_imagem_uma_vez(df, out):  # teste 4: nenhuma imagem perdida ou repetida
    assert out.path.is_unique and set(out.path) == set(df.path)


def test_unseen_na_etapa_fixa(out):
    for etapa, subtipos in UNSEEN.items():
        u = out[out.id_subtipo.isin(subtipos)]
        assert (u.etapa_pac == etapa).all() and (u.etapa_img == etapa).all()


def test_pacientes_por_etapa(out):  # D23: mval = mtst = máx(2, arred(0,2·n)); o F fica 1/2/2
    pac = out.drop_duplicates("id_paciente")
    n = pd.crosstab(pac.id_subtipo, pac.etapa_pac).loc[SEEN, ["mtrn", "mval", "mtst"]]
    assert n.loc["F"].tolist() == [1, 2, 2]
    assert n.loc["DC"].tolist() == [23, 7, 7]


def test_p_img_mistura_pacientes(out):  # contraprova: no P-IMG os pacientes seen se espalham
    seen = out[out.id_subtipo.isin(SEEN)]
    assert (seen.groupby("id_paciente").etapa_img.nunique() > 1).mean() > 0.5


def test_reprodutivel(df, out):  # teste 8
    assert split(df, seed=0).equals(out)
    assert not split(df, seed=10).etapa_pac.equals(out.etapa_pac)
