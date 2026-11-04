"""Grade completa: todas as sementes × magnificações (Etapa 8).

Cada rodada (semente, magnificação) = treino "pac" + treino "img" + avaliação no mtst,
igual ao piloto. Pula o que já tem "fim" no config.json, então dá para interromper e
rodar de novo: recomeça do primeiro passo incompleto (um treino interrompido recomeça do
zero, D50). Ordem: semente por fora, para cada semente fechar as 4 magnificações juntas.

Uso (Windows): UV_NO_SYNC=1 uv run python -u -m scripts.run_grid
"""

import json
import time

from src.episodes import MAGS
from src.evaluate import RESULTS, avalia
from src.splits import SEEDS
from src.train import DIVISOES, treina
from src.train import RESULTS as TREINOS


def _completo(config):
    return config.exists() and "fim" in json.loads(config.read_text())


def roda_grade(seeds=SEEDS, mags=MAGS):
    for seed in seeds:
        for mag in mags:
            t = time.time()
            for div in DIVISOES:
                if not _completo(TREINOS / f"{mag}_{div}_seed{seed}" / "config.json"):
                    print(f"===== treino {div} | semente {seed} | {mag}x", flush=True)
                    treina(seed, mag, div)
            if not _completo(RESULTS / "mtst" / f"{mag}_seed{seed}_config.json"):
                print(f"===== avaliação | semente {seed} | {mag}x", flush=True)
                avalia(seed, mag)
            print(f"===== semente {seed} | {mag}x pronta em {(time.time() - t) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    roda_grade()
