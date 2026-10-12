"""Etapa 4: testes de sanidade dos episódios (PROJETO.md, seções 6.4 e 12) sobre a divisão real da semente 0."""

from pathlib import Path

import pandas as pd
import pytest

from src.episodes import K_MVAL, K_SHOTS, K_TST, MAGS, PROTOCOLOS, episodio
from src.splits import SEEN, UNSEEN

SPLIT = Path("data/splits/seed0.csv")
pytestmark = pytest.mark.skipif(not SPLIT.exists(), reason="rode python -m src.splits antes")
N = 50  # episódios por configuração nos testes (o __main__ de episodes.py confere 1.000)


@pytest.fixture(scope="module")
def split():
    return pd.read_csv(SPLIT)


def configs():
    for mag in MAGS:
        yield mag, "mval", K_MVAL
        for k in K_SHOTS:
            yield mag, "mtst", k


def fracs(split, protocolo, imgs):
    """frac_V1 (paciente do query também no support) e frac_V2 (paciente do query visto no mtrn)."""
    coluna = PROTOCOLOS[protocolo][0]
    q = imgs[imgs.papel == "Q"]
    v1 = q.id_paciente.isin(imgs.id_paciente[imgs.papel == "S"]).mean()
    v2 = q.id_paciente.isin(split.id_paciente[split[coluna] == "mtrn"]).mean()
    return v1, v2


def test_formato(split):  # 3 classes (seen, seen, unseen), k no support e K_TST no query por classe
    for mag, etapa, k in configs():
        classes, imgs = episodio(split, "P-PAC", etapa, k, 0, mag, 0)
        assert classes[0] in SEEN and classes[1] in SEEN and classes[2] in UNSEEN[etapa]
        n = imgs.groupby(["id_subtipo", "papel"]).size()
        assert all(n[(c, "S")] == k and n[(c, "Q")] == K_TST for c in classes)


def test_pac_sem_vazamento(split):  # teste 2 + frac_V1 = frac_V2 = 0 no P-PAC
    for mag, etapa, k in configs():
        for i in range(N):
            assert fracs(split, "P-PAC", episodio(split, "P-PAC", etapa, k, 0, mag, i)[1]) == (0, 0)


def test_mix_sem_v2(split):  # P-MIX: divisão por paciente → sem V2
    for i in range(N):
        assert fracs(split, "P-MIX", episodio(split, "P-MIX", "mtst", 5, 0, 100, i)[1])[1] == 0


def test_contraprova_por_imagem_vaza(split):  # P-IMG tem V2; P-MIX tem V1
    img = [fracs(split, "P-IMG", episodio(split, "P-IMG", "mtst", 5, 0, 100, i)[1]) for i in range(N)]
    mix = [fracs(split, "P-MIX", episodio(split, "P-MIX", "mtst", 5, 0, 100, i)[1]) for i in range(N)]
    assert sum(v2 for _, v2 in img) / N > 0.3
    assert sum(v1 for v1, _ in mix) / N > 0.3


def test_pareado(split):  # D29: mesmas classes nos três protocolos
    for i in range(N):
        assert len({tuple(episodio(split, p, "mtst", 5, 0, 100, i)[0]) for p in PROTOCOLOS}) == 1


def test_reprodutivel(split):  # teste 8 e D30 (a mval se repete a cada chamada)
    a = episodio(split, "P-PAC", "mval", K_MVAL, 0, 100, 7)
    b = episodio(split, "P-PAC", "mval", K_MVAL, 0, 100, 7)
    assert a[0] == b[0] and a[1].equals(b[1])
    assert not a[1].equals(episodio(split, "P-PAC", "mval", K_MVAL, 0, 100, 8)[1])
