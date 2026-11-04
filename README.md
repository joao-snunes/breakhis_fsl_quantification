# breakhis_fsl_quantification

Quanto a nota de um método few-shot em histologia fica inflada quando os dados são separados
**por imagem** em vez de **por paciente**? Este repositório mede esse efeito na base BreakHis,
usando o método BatchBased + MobileNetV3-Small-0.75 do artigo MetaChest
(Montalvo-Lezama & Fuentes-Pineda, 2026). O método fica fixo; muda só a regra de separação.

As imagens de um paciente no BreakHis são recortes da mesma lâmina, com a mesma coloração. Se o
mesmo paciente aparece no treino e na avaliação, o modelo pode acertar por reconhecer a lâmina, e
não o subtipo do tumor. É o problema de usar `KFold` em vez de `GroupKFold` com o paciente como
grupo.

## O que é medido

Duas fontes de vazamento:

| Fonte | Onde | Exemplo |
|---|---|---|
| V1 | Dentro do episódio (support × query) | Imagens do paciente X no support e no query |
| V2 | Entre etapas (meta-treino × meta-teste) | O backbone treinou com o paciente X, que reaparece no meta-teste |

Três protocolos ligam ou desligam cada uma:

| Protocolo | Divisão entre etapas | Montagem do episódio | V2 | V1 |
|---|---|---|---|---|
| P-IMG | por imagem | por imagem | sim | sim |
| P-MIX | por paciente | por imagem | não | sim |
| P-PAC | por paciente | por paciente | não | não |

O vazamento total é a diferença de HM entre P-IMG (como o artigo faz) e P-PAC (sem vazamento).

## Método em resumo

- **Dados:** BreakHis, 8 subtipos, 4 magnificações (40×, 100×, 200×, 400×), cada uma um
  experimento separado. Paciente = número de registro sem as letras de seção. Depois de remover
  imagens repetidas e pacientes com dois subtipos: 7.458 imagens de 70 pacientes.
- **Classes:** seen (meta-treino, meta-validação e meta-teste) = DC, F, MC, TA; unseen da
  meta-validação = A, LC; unseen do meta-teste = PT, PC.
- **Divisão por paciente:** em cada classe seen, n = max(2, round(0,2 × pacientes)) pacientes na
  meta-validação, n no meta-teste e o resto no meta-treino; a mesma divisão nas 4 magnificações.
  A divisão por imagem sorteia imagens soltas com exatamente os mesmos tamanhos por classe e etapa.
- **Episódios:** 3 classes (2 seen + 1 unseen), k ∈ {1, 5, 10} imagens por classe no support e
  10 no query. No P-PAC, support e query vêm de pacientes diferentes. O episódio i tem as mesmas
  classes nos três protocolos.
- **Modelo:** MobileNetV3-Small-0.75 pré-treinada no ImageNet (`timm`
  `tf_mobilenetv3_small_075.in1k`), imagens na resolução nativa 700×456.
  - Meta-treino: backbone + cabeça de 4 saídas, AdamW lr 1e-4, lote 64, early stopping pela HM
    da meta-validação (100 tarefas fixas, paciência 10). A época 0 (só ImageNet) concorre.
  - Adaptação em cada episódio: backbone congelado, cabeça linear nova (1.024 → 3) com sigmoide,
    100 passos de AdamW lr 0,005 sobre metade do support por passo.
  - No meta-teste, cada episódio é avaliado com o backbone meta-treinado e com o só ImageNet
    (sem V2).
- **Métricas por episódio:** AUC seen, AUC unseen e a média harmônica das duas (HM, métrica
  principal), AUC por classe, acurácia por imagem e balanceada por paciente, e a exposição ao
  vazamento (`frac_V1`, `frac_V2`).
- **Grade:** 5 sementes (divisões de pacientes) × 4 magnificações × 3 protocolos × 3 k × 1.000
  episódios × 2 backbones; 40 treinos e 360.000 episódios avaliados.

## Resultados principais

HM no meta-teste com o backbone meta-treinado, média das 4 magnificações e 5 sementes:

| k | P-IMG | P-MIX | P-PAC | P-IMG − P-PAC |
|---|---|---|---|---|
| 1 | 80,8 | 73,1 | 55,1 | 25,8 |
| 5 | 91,5 | 89,2 | 56,5 | 35,0 |
| 10 | 94,6 | 94,4 | 56,7 | 37,8 |

- Separar por imagem infla a HM em 24 a 41 pontos (43% a 80%), conforme a magnificação e o k, e o
  P-IMG ficou acima do P-PAC em todas as 20 rodadas.
- Sem vazamento, a HM não melhora com mais exemplos por classe; o ganho com k vem do vazamento.
- O V1 sozinho (P-MIX) já explica a maior parte da diferença.
- O meta-treino só ajuda muito quando há V2 (+8 a +17 pontos no P-IMG); sem vazamento, o ganho
  médio é de ~2 pontos e muda de sinal entre rodadas.

Análise completa em `notebooks/04_resultados.ipynb`.

## Estrutura

```
scripts/build_csv.py         lê a base bruta → data/breakhis_raw.csv
notebooks/01_eda.ipynb       exploração da base bruta (anomalias de paciente)
notebooks/02_clear_csv.ipynb limpeza → data/breakhis.csv
src/splits.py                divisões por paciente e por imagem → data/splits/seed{S}.csv
src/episodes.py              sorteio dos episódios dos três protocolos
src/cache.py                 imagens em 700×456 → data/cache/{mag}.npz
src/device.py                escolhe o dispositivo (cuda/ROCm → mps → directml → cpu)
src/model.py                 backbone e embeddings
src/adapt.py                 cabeça de cada episódio e métricas
src/train.py                 meta-treino com early stopping → results/train/
src/evaluate.py              avaliação no meta-teste → results/mtst/
scripts/run_grid.py          grade completa, retomável
notebooks/04_resultados.ipynb tabelas e gráficos dos resultados
tests/                       testes (inclusive os de vazamento)
```

`data/` e `results/` ficam fora do Git.

## Ambiente

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12. O `torch` do PyPI já cobre CPU, Apple
Silicon (MPS) e NVIDIA (CUDA no Linux).

```bash
uv sync
uv run python -m src.device   # mostra a versão do torch e o dispositivo escolhido
uv run pytest
```

### Windows com GPU AMD (ROCm)

Receita validada numa RX 7600 (`gfx1102`), no Git Bash:

```bash
uv sync
uv pip install --reinstall-package torch --reinstall-package torchvision \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1102]==2.13.*" "torchvision[device-gfx1102]==0.28.*"
UV_NO_SYNC=1 uv run python -m src.device   # deve mostrar "cuda"
```

- Use o par `torch 2.13` + `torchvision 0.28`: no índice estável, o `torch 2.14` vem com o
  `torchvision 0.28`, que é incompatível (`torchvision::nms does not exist`).
- Rode sempre com `UV_NO_SYNC=1` (ou `uv run --no-sync`); sem isso, o `uv run` volta ao
  `torch` de CPU do lock.
- O MIOpen dessa versão falha ao compilar a BatchNorm; o `get_device()` o desliga quando o
  PyTorch é ROCm, e o PyTorch usa os próprios kernels.
- Para outra placa, troque `gfx1102` pela arquitetura dela.

## Como rodar

Todos os comandos rodam na raiz do repositório. No Windows com ROCm, prefixe com `UV_NO_SYNC=1`.

### 1. Dados

Baixe a base BreaKHis_v1 (por exemplo, pelo Kaggle) e extraia em `data/BreaKHis_v1/`, mantendo
as pastas originais (`data/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/100X/*.png`).
Depois:

```bash
uv run python scripts/build_csv.py   # data/breakhis_raw.csv, 7.909 imagens (alguns minutos)
```

Execute `notebooks/02_clear_csv.ipynb` (no Jupyter/VS Code, ou com o comando abaixo) para gerar
`data/breakhis.csv`, com 7.458 imagens:

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/02_clear_csv.ipynb
```

```bash
uv run python -m src.splits     # data/splits/seed{S}.csv para as 5 sementes
uv run python -m src.episodes   # opcional: sorteia todos os episódios e confere as regras
uv run python -m src.cache      # data/cache/{mag}.npz, ~1,9 GB por magnificação
```

### 2. Uma rodada (uma semente e uma magnificação)

A semente e a magnificação são constantes no fim de cada módulo (`SEED`, `MAG`, `DIVISAO`).

```bash
uv run python -m src.train      # um treino → results/train/{mag}_{divisao}_seed{S}/
uv run python -m src.evaluate   # precisa dos treinos "pac" e "img" → results/mtst/{mag}_seed{S}.csv
```

### 3. Grade completa

```bash
UV_NO_SYNC=1 PYTHONIOENCODING=utf-8 uv run python -u -m scripts.run_grid > results/grade.log 2>&1
```

Roda os 40 treinos e as 20 avaliações, semente por semente. Pula o que já tem `fim` no
`config.json`; se for interrompida, basta rodar de novo (um treino incompleto recomeça do zero).
Numa RX 7600 levou 11,1 h. Rode direto num terminal da máquina: numa sessão `ssh`, o processo
pode morrer quando a conexão cair.

### 4. Análise

Abra `notebooks/04_resultados.ipynb`, ou execute-o como no passo 1.

### Testes

```bash
uv run pytest            # testes rápidos (~30 s)
uv run pytest -m slow    # testes que treinam a rede (minutos; melhor com GPU)
uv run pytest -m ""      # todos
```

Os testes conferem, entre outras coisas, que nenhum paciente aparece em duas etapas, que no P-PAC
nenhum paciente está no support e no query do mesmo episódio e que as divisões por imagem e por
paciente têm os mesmos tamanhos.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `results/train/{mag}_{divisao}_seed{S}/epocas.csv` | Perda do meta-treino e métricas da meta-validação por época |
| `results/train/.../config.json` | Constantes, semente, dispositivo, versões, commit e melhor época |
| `results/mtst/{mag}_seed{S}.csv` | Uma linha por episódio × protocolo × k × backbone |
| `results/mtst/{mag}_seed{S}_pacientes.csv.gz` | Pacientes do support e do query de cada episódio (auditoria) |
| `results/mtst/{mag}_seed{S}_config.json` | Configuração da avaliação |
| `results/embeddings/*.npz` | Embeddings do meta-teste de cada backbone |

## Reprodutibilidade

Toda aleatoriedade vem de uma semente explícita. Cada episódio tem um gerador próprio, derivado de
`[semente, magnificação, etapa, k, nº do episódio]`, então divisões e episódios são idênticos em
qualquer máquina. A adaptação roda na CPU e é determinística; o treino na GPU pode variar um pouco
entre máquinas.

## Referência

Montalvo-Lezama, B.; Fuentes-Pineda, G. MetaChest: generalized few-shot learning of pathologies
from chest X-rays. *Visual Computing for Industry, Biomedicine, and Art*, 9:4, 2026.
Código original: https://github.com/bereml/meta-cxr
