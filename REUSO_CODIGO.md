# Reuso do código de referência (`bereml/meta-cxr`)

Mapa do que copiar, simplificar e descartar do código do artigo MetaChest, e do que
precisa ser criado do zero. Referência: https://github.com/bereml/meta-cxr (clonado
em `../meta-cxr`).

## 1. Visão geral do código original

| Arquivo | Linhas | O que faz |
|---|---|---|
| `config.toml` | — | Caminho da MetaChest e listas de classes mtrn/mval/mtst |
| `args.py` | 159 | ~60 argumentos de linha de comando (`argparse`) |
| `data.py` | 798 | Leitura da MetaChest, divisão por rótulos, amostrador multirrótulo, DataLoaders e funções de teste |
| `method/base.py` | 143 | Classe base Lightning, AUC seen/unseen/HM |
| `method/batchbased.py` | 169 | Meta-treino em lotes e adaptação por episódio |
| `method/protonet.py` | 216 | ProtoNet-ML |
| `network/factory.py` | 96 | `Net` = backbone + cabeça; congelar/descongelar |
| `network/models.py` | 250 | Registro de ~15 arquiteturas `timm` e carga de checkpoints |
| `pretrain.py` | 141 | Treino com PyTorch Lightning, early stopping, checkpoint |
| `adapt.py` | 108 | Avaliação no meta-teste e gravação do CSV de episódios |
| `utils.py` | 201 | IC, agregação de resultados (CSV/MD/TeX), chamadas por subprocess |
| `experiment.py` | 340 | Funções `paper_*` que reproduzem cada tabela do artigo |

## 2. Copiar (quase igual)

| Peça | Origem | Observação |
|---|---|---|
| Backbone + cabeça linear | `network/factory.py` (`Net`, `new_head`, `freeze_and_eval_backbone`) | Manter só `nn.Linear`; criar o backbone direto com `timm.create_model('tf_mobilenetv3_small_075.in1k', pretrained=True, num_classes=0)` (o mesmo do original, `network/models.py`) |
| Lógica da adaptação por episódio | `method/batchbased.py` (`adapt_episode`, `adapt_episode_inner`) | Cabeça nova → backbone congelado → 100 passos AdamW, lr 0,005, 50% do support por passo → sigmoide no query |
| Passo de treino do meta-treino | `method/batchbased.py` (`train_batch`) | Backbone e cabeça juntos, AdamW lr 1e-4, `BCEWithLogitsLoss` |
| Métricas AUC seen/unseen/HM | `method/base.py` (`auroc`, `compute_metrics`) | Copiar como está (`torchmetrics`, média `micro`). Atenção: assume as colunas de rótulos ordenadas **seen primeiro, unseen depois** |
| IC de 95% | `utils.py` (`compute_mean_ci`) | Copiar |
| IC entre sementes | `utils.py` (`agregate_run_df_aux`, com `scipy.stats.bootstrap`) | Opcional; pode simplificar |
| Normalização | `base.py` (`backbone.pretrained_cfg['mean'/'std']`) | Copiar a ideia |
| Hiperparâmetros padrão | `args.py` e `batchbased.py` (`add_args`) | Constantes no topo dos módulos |

## 3. Simplificar ou descartar

| # | O que existe no original | Por que não serve ou sobra | Substituir por |
|---|---|---|---|
| S1 | Pipeline da MetaChest (4 bases, Snakemake, `metachest.csv`, pastas `images-{size}`) | Outra base | `data/breakhis.csv` gerado do nome dos arquivos |
| S2 | Filtro de distribuição (`data_distro`, `distro/*.csv`) | Não se aplica | Remover |
| S3 | Imagens "No Finding" (`nf_df`, `_add_nf_data`, `_load_nf_data`; ao menos 1 por episódio) | O BreakHis não tem imagens normais; com rótulo único, as outras classes do episódio já são negativos para a AUC | Remover |
| S4 | Divisão das etapas por máscara de rótulos (`_filter_mset`) | Só funciona em multirrótulo; com rótulo único, as classes seen ficariam todas no meta-treino. E ignora o paciente | Divisão **explícita** salva em arquivo (`splits.py`), por paciente ou por imagem |
| S5 | Amostrador multirrótulo (`EpisodeSampler.generate_episode`, `sample_at_least_k_shot`: classes excluídas, ordenação por frequência, "pelo menos k") | Complexidade que só existe por causa do multirrótulo | Amostragem simples de **exatamente k** por classe, com a opção `group_by_patient` |
| S6 | ProtoNet-ML (`method/protonet.py`) | Fora do escopo (D7) | Remover; citar como trabalho futuro |
| S7 | Registro de ~15 arquiteturas e carga de checkpoints por nome (`network/models.py`, `backbones.py`) | Só uma arquitetura | Uma linha com `timm.create_model` e `torch.save/load` do `state_dict` |
| S8 | Registro de métodos (`METHODS`, `Registry`) | Só um método | Funções diretas |
| S9 | ~60 argumentos `argparse` (`args.py`) | Difícil de rastrear e reproduzir | Constantes nomeadas no topo de cada módulo |
| S10 | Experimentos chamados via `subprocess` (`utils.pretrain_adapt` monta `python pretrain.py --x y`) | Indireto e difícil de depurar | Chamadas de função em Python (`run_grid.py`) |
| S11 | `experiment.py` com `paper_arch`, `paper_resolution`, `paper_pretraining`, `paper_task_complexity_*` | Reproduzem tabelas do artigo | Um script de grade com os seus fatores (magnificação × protocolo × k × semente) |
| S12 | PyTorch Lightning (`LightningModule`, `Trainer`, `ModelCheckpoint`, `EarlyStopping`) | O BatchBased já usa otimização manual (`automatic_optimization = False`), então o Lightning ajuda pouco e esconde o loop | **Loop de treino em PyTorch puro** (~100 linhas: épocas → lotes → validação por episódios → early stopping → salvar o melhor). Mais fácil de explicar na banca |
| S13 | TensorBoard (`TensorBoardLogger`) | Opcional | Log em CSV por época (perda e HM da meta-validação) |
| S14 | Opções não usadas no padrão: `batchbased_mval_net_steps` (ajuste do backbone por episódio, padrão 0), `batchbased_reset_head` (padrão 0), `batchbased_sch_milestones` (definida, mas nenhum scheduler é criado) | Não alteram o método padrão | Remover |
| S15 | Passar as imagens pelo backbone congelado a cada um dos 100 passos da adaptação | Desperdício: com backbone congelado, em `eval()` e sem augmentation, o embedding não muda | **Pré-calcular os embeddings** uma vez por checkpoint e adaptar a cabeça sobre vetores em memória (equivalente e muito mais rápido). Na meta-validação durante o treino, recalcular por época |
| S16 | Conjunto fixo de tamanhos de imagem e pastas pré-redimensionadas | Uma resolução só | Redimensionar uma vez (384×384) e guardar em cache como `uint8` (~0,9 GB por magnificação; em float32 seriam ~3,5 GB), normalizando na hora do uso |
| S17 | Funções de teste manuais via `fire` (`test_build_dl`, `test_build_mdl`, `compute_mean_mdl`, `show_grid`) | Testes visuais e manuais | Testes automáticos com `pytest` (`tests/`) e um notebook de inspeção |
| S18 | Agregação para Markdown e LaTeX (`aggregate_exp_df`) | Útil, mas acoplada à estrutura de pastas do original | Agregação com `pandas` num notebook de resultados |
| S19 | Flags de desempenho (`benchmark`, `deterministic` com 3 modos, `num_workers=8`, `float32_matmul_precision`) | Ajuste fino desnecessário para ~2.000 imagens | Semente fixa (`torch`, `numpy`, `random`); `num_workers` pequeno ou 0 com cache em memória; fp32 por padrão (bf16 só como opção em CUDA, ver D18) |

## 4. Criar do zero (não existe no original)

| Peça | Arquivo sugerido | Função |
|---|---|---|
| CSV do BreakHis | `scripts/build_csv.py` → `notebooks/02_clear_csv.ipynb` | CSV bruto extraído do nome dos arquivos (com hash dos pixels) e, depois, o CSV limpo |
| Divisões por paciente e por imagem com tamanhos pareados | `src/splits.py` | Seção 6.3 do `PROJETO.md` |
| Amostrador com `group_by_patient` | `src/episodes.py` | Support e query em pacientes disjuntos; devolve os IDs dos pacientes |
| Métricas novas | `src/adapt.py` | Acurácia por imagem, acurácia balanceada por paciente, `frac_V1`, `frac_V2` |
| Embeddings pré-calculados | `src/model.py` | Pré-calcula e guarda os embeddings por checkpoint |
| Testes anti-vazamento | `tests/test_splits.py`, `tests/test_episodes.py` | Seção 12 do `PROJETO.md` |
| Contraprova | `notebooks/03_contraprova.ipynb` | Regressão logística do scikit-learn sobre os mesmos embeddings e episódios |
| Grade de experimentos | `scripts/run_grid.py` | Roda e retoma a grade, sem refazer o que já existe |

## 5. Estrutura sugerida do novo repositório

```
breakhis_fsl_quantification/
├── CLAUDE.md               # fora do Git: @docs/CLAUDE.md
├── README.md
├── data/                   # fora do Git
│   ├── BreaKHis_v1/
│   ├── breakhis_raw.csv    # build_csv.py
│   ├── breakhis.csv        # 02_clear_csv.ipynb
│   └── splits/
├── docs/                   # worktree da branch órfã docs
│   ├── CLAUDE.md
│   ├── PROJETO.md
│   ├── REUSO_CODIGO.md
│   └── DIARIO.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clear_csv.ipynb
│   ├── 03_contraprova.ipynb
│   └── 04_resultados.ipynb
├── scripts/
│   ├── build_csv.py
│   └── run_grid.py
├── src/
│   ├── device.py           # cuda/ROCm → mps → directml → cpu
│   ├── cache.py            # gera/carrega data/cache/{mag}.npz (700×456 nativa, uint8)
│   ├── splits.py
│   ├── episodes.py
│   ├── model.py            # backbone + normalização + embeddings
│   ├── adapt.py            # um episódio: cabeça nova + métricas
│   ├── train.py            # meta-treino + mval + early stopping (nunca lê o mtst)
│   └── evaluate.py         # nota final no mtst → CSVs
├── tests/
│   ├── test_splits.py
│   └── test_episodes.py
└── results/                # fora do Git
```

## 6. Estimativa de tamanho

O original tem cerca de 2.650 linhas de Python. Com as simplificações acima, o
núcleo do TCC deve ficar entre **600 e 900 linhas**, sem contar testes e notebooks.
