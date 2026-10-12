"""sorteia episódios.

Um episódio tem 3 classes (2 seen + 1 unseen da etapa). Para cada classe, k imagens vão
para o support e K_TST para o query.

Protocolos = de qual divisão vêm as imagens + como o episódio as sorteia:
    P-IMG: divisão por imagem,   sorteio por imagem
    P-MIX: divisão por paciente, sorteio por imagem
    P-PAC: divisão por paciente, sorteio por paciente (support e query em pacientes diferentes)

O gerador de cada episódio vem de [semente, magnificação, etapa, k, nº do episódio]. As classes saem iguais nos três protocolos (compartilham seed). A mval repete as mesmas tarefas.

Uso: uv run python -m src.episodes
"""

import numpy as np
import pandas as pd

from src.splits import SEEDS, SEEN, UNSEEN

K_SHOTS = [1, 5, 10]   
K_TST = 10             
K_MVAL = 5             
N_MVAL, N_MTST = 100, 1000
MAGS = [40, 100, 200, 400]
PROTOCOLOS = {         
    "P-IMG": ("etapa_img", False),
    "P-MIX": ("etapa_pac", False),
    "P-PAC": ("etapa_pac", True),
}
ETAPA_ID = {"mval": 1, "mtst": 2}


def _sorteia_pacientes_classe(imgs, k, por_paciente, rng):
    """Índices de support e query de UMA classe."""
    if not por_paciente:
        idx = rng.permutation(imgs.index)
        return idx[:k], idx[k:k + K_TST]
    # por paciente: pacientes entram no support até somarem k imagens; o query sai só dos restantes
    pacientes = rng.permutation(sorted(imgs.id_paciente.unique()))
    acumulado = imgs.id_paciente.value_counts()[pacientes].cumsum().to_numpy() # nro acumulado de imgs de todos os pacientes
    no_support = imgs.id_paciente.isin(pacientes[:np.searchsorted(acumulado, k) + 1]) # quais pacientes somam k (6 no mval, 10 no mtst)i magens no suporte
    assert (~no_support).sum() >= K_TST, "pacientes insuficientes para o query"  # D27: não deve ocorrer
    return (rng.choice(imgs.index[no_support], k, replace=False),
            rng.choice(imgs.index[~no_support], K_TST, replace=False)) # sorteia k imagens dos pacientes do support e K_TST dos pacientes restantes (query)


def episodio(split, protocolo, etapa, k, seed, mag, i):
    """Devolve (classes, imagens): classes em ordem seen, seen, unseen; imagens com coluna papel S/Q."""
    coluna, por_paciente = PROTOCOLOS[protocolo]
    rng = np.random.default_rng([seed, mag, ETAPA_ID[etapa], k, i])
    classes = [str(c) for c in (*rng.choice(SEEN, 2, replace=False), rng.choice(UNSEEN[etapa]))]
    pool = split[(split.magnificacao == mag) & (split[coluna] == etapa)]

    papel = {}
    for c in classes:
        sup, qry = _sorteia_pacientes_classe(pool[pool.id_subtipo == c], k, por_paciente, rng)
        papel |= dict.fromkeys(sup, "S") | dict.fromkeys(qry, "Q")
    imgs = pool.loc[list(papel), ["path", "id_paciente", "id_subtipo"]].assign(papel=list(papel.values())) # path, id_paciente, id_subtipo, papel (S/Q)

    assert len(imgs) == 3 * (k + K_TST) and imgs.path.is_unique  # nenhuma imagem nos dois lados
    if por_paciente:  # nenhum paciente repetido no support e no query
        assert not set(imgs.id_paciente[imgs.papel == "S"]) & set(imgs.id_paciente[imgs.papel == "Q"])
    return classes, imgs


if __name__ == "__main__":
    """
    para cada semente, magnificação e protocolo, gera 100 episodios de mval (5 imagens de support por classe) e 
    1000 episodios de mtst (com 1, 5 ou 10 imagens no support por classe, de acordo com a iteração), sempre com 10 imagens de query por classe.
    """
    import time
    for seed in SEEDS:
        split = pd.read_csv(f"data/splits/seed{seed}.csv")
        t = time.time()
        for mag in MAGS:
            for protocolo in PROTOCOLOS:
                for i in range(N_MVAL):
                    episodio(split, protocolo, "mval", K_MVAL, seed, mag, i)
                for k in K_SHOTS:
                    for i in range(N_MTST):
                        episodio(split, protocolo, "mtst", k, seed, mag, i)
        print(f"seed {seed}: {len(MAGS) * len(PROTOCOLOS) * (N_MVAL + len(K_SHOTS) * N_MTST)} episódios ok ({time.time() - t:.0f} s)")
