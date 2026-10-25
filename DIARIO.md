# Diário de laboratório

Formato: data · o que rodou · resultado · decisão.

## 05/10/2026 — Etapa 0: ambiente e repositório

- **Rodou:** revisão da documentação contra o `meta-cxr`; instalação de uv 0.12 e
  Python 3.12.15 no macOS (M1); `uv sync`; `tests/test_device.py`.
- **Resultado:**
  - torch 2.14.1 com MPS no Mac; o passo de treino da MobileNet roda.
  - O `meta-cxr` usa `tf_mobilenetv3_small_075.in1k` (não `mobilenetv3_small_075`):
    1,02 M de parâmetros, normalização média 0,5 / desvio 0,5.
  - No timm atual, `num_features` = 432; o embedding (com `num_classes=0`) tem 1024
    dimensões (`head_hidden_size`). O teste pegou a confusão.
- **Decisões:**
  - D18: sem CUDA; dispositivo automático (cuda/ROCm → mps → directml → cpu), fp32.
  - Modelo fiel ao artigo (`tf_`).
  - Repositório privado no início; documentação viva numa branch órfã `docs`
    montada em `docs/` via `git worktree`.
  - Removidas a seção 2.4 e as pendências P1 (replicação) e P2 (GPU).
- **Validado** pelo aluno, incluindo o Windows (RX 7600). Etapa 0 concluída.

## 05/10/2026 — Etapa 2 (parcial): CSV e análise exploratória

- **Rodou:** download do BreakHis (fonte oficial, UFPR/VRI) em `data/BreaKHis_v1/`;
  `scripts/build_csv.py`; `notebooks/01_eda.ipynb`.
- **Resultado:**
  - CSV com 7.909 imagens, igual à página oficial; todas as pastas têm as 4
    magnificações. 82 pares (subtipo, pasta), 81 IDs de pasta, 70 casos.
  - Anomalia A (hash dos pixels): 14-13412 tem as mesmas 123 imagens como DC e como
    LC; LC 14-12204 tem 2 pares repetidos a 40×.
  - Anomalia B: nove números de lâmina se repartem em várias pastas (letras de seção
    diferentes); 21998 (F/PT) e 29960 (A/F) cruzam subtipos. A cor média sugere
    origem comum em parte delas (22549AB × CD no percentil 0,1), mas não em todas.
  - Viabilidade (P-PAC, k_tst = 15): o gargalo é o query, não o k. F, MC e TA falham
    em 10% a 35% dos sorteios mesmo com k = 1; k = 20 é crítico para TA e MC; com
    unidade = caso, a adenose a 100× fica inviável.
- **Decisões:** nenhuma ainda. Registradas como pendentes D19 a D22 (PROJETO.md,
  seção 5), com detalhes na seção 4.2; P3 e P4 atualizadas.
- **Falta:** decidir D19 a D22, os k-shots e o k_tst; depois, fechar a Etapa 2 com o
  merge da branch `staging`.

## 07/10/2026 — Etapa 2: decisões sobre pacientes e limpeza da base

- **Decisões:**
  - **D21:** paciente = número de registro sem as letras de seção. Pastas com o
    mesmo número (ex. 22549AB, CD e G) são fragmentos da mesma peça cirúrgica, logo
    do mesmo paciente. Justificativa completa no PROJETO.md, seção 4.2-B.
  - **D19:** 14-13412 (mesmas imagens em DC e LC) mantido só em LC.
  - **D20:** deduplicar as imagens repetidas. Conferido: nenhuma imagem se repete
    entre benigno e maligno; se houvesse, sairia das duas classes.
  - **D22:** 14-21998 (F/PT) e 14-29960 (A/F) perdem as imagens de F.
- **Rodou:** pipeline de dados reorganizado em três passos: `build_csv.py` gera o
  `breakhis_raw.csv` (com `classe` benign/malignant, `id_subtipo`, `des_subtipo`,
  `id_paciente`, `secao` e `rgb_hash`); o `01_eda.ipynb` explora o CSV bruto; o novo
  `02_clear_csv.ipynb` aplica D19 a D22 e grava o `breakhis.csv` limpo.
- **Resultado:** 7.458 imagens usadas (de 7.909) e 70 pacientes. Removidas: D22 = 326,
  D19 = 123, D20 = 2.
- **Atenção para a Etapa 3:** o F ficou com 5 pacientes; com no mínimo 2 em mval e 2 em
  mtst, sobra 1 no meta-treino (P2).
- **Falta:** k-shots e k_tst (P3, P4); depois, o merge da `staging` na `main`.

## 08/10/2026 — Etapa 3: divisões entre etapas

- **Decisões:**
  - **D23 (P2):** mval = mtst = máx(2, arred(0,2·n)) pacientes por classe seen. O F
    fica 1/2/2: o mínimo de 2 é o que permite ao P-PAC separar support e query por
    paciente; ter 1 paciente de F no mtrn é limitação a declarar.
  - **D24:** o sorteio de pacientes é o mesmo nas 4 magnificações.
  - **D25:** um CSV por semente (`data/splits/seed{S}.csv`) com `etapa_pac` e
    `etapa_img` lado a lado.
  - **D26:** coluna `classe` renomeada para `malignidade`; "classe" no few-shot é o
    subtipo.
- **Rodou:** `src/splits.py` com as sementes 0, 10, 42, 109 e 2026 (constantes no código);
  `tests/test_splits.py` (8 testes passam); simulador interativo das divisões e dos
  episódios (artifact) para entender as etapas antes de decidir.
- **Resultado:**
  - Pacientes mtrn/mval/mtst em todas as sementes: DC 23/7/7, MC 3/2/2, TA 2/2/2,
    F 1/2/2 (o paciente de F no mtrn: 25197 em três sementes, 23060 e 23222).
  - A 40×, o mtrn tem 629 a 702 imagens seen (~50%), menos que os 60% previstos.
  - No P-IMG, 100% dos pacientes seen ficam em mais de uma etapa (vazamento V2 em
    todos), com os mesmos totais de imagens do P-PAC.
- **Falta:** k-shots e k_tst (P3, P4), antes da Etapa 4 (episódios).

## 12/10/2026 — Etapa 4 (decisões): episódios

- **Rodou:** viabilidade exaustiva do P-PAC sobre as divisões reais (5 sementes × 4
  magnificações × mval/mtst, todas as ordens de pacientes) para k ∈ {1, 3, 5, 10, 15}
  com k_tst = 10; leitura do `EpisodeSampler` e do `args.py` do `meta-cxr`.
- **Resultado:**
  - k = 1, 5 e 10: viáveis em 40 de 40 combinações; k = 15: em 18 de 40, e
    impossível em uma (TA, mtst, 100×, semente 2026).
  - Ressortear episódios inviáveis (o que o original faz) concentraria o query de
    classes com 2 pacientes sempre no mesmo paciente.
  - O original usa k = 30 / k_tst = 30 na mval e sorteia episódios novos por época.
- **Decisões:** D27 (k ∈ {1, 5, 10}; k = 15 e 20 excluídos e justificados), D28
  (k_tst = 10), D29 (episódios pareados entre protocolos), D30 (mval com as mesmas
  100 tarefas), D31 (geradores derivados da semente da rodada), D32 (k = 5 na mval),
  D33 (resultados em CSV). Texto de metodologia na seção 6.4.
- **D34:** a mval de cada backbone usa o mesmo estilo do seu protocolo (por paciente
  na divisão por paciente; por imagem na divisão por imagem), de ponta a ponta.
- **Implementação:** `src/episodes.py` (sorteio por imagem ou por paciente, geradores
  derivados de [semente, magnificação, etapa, k, episódio]) e `tests/test_episodes.py`
  (6 testes; 14 no total passam). 1.000 episódios por configuração em todas as
  sementes (186 mil) gerados sem erro, cerca de 2,6 ms cada.
- **Exposição ao vazamento no mtst a 100×** (média de 200 episódios × 5 sementes):
  P-IMG tem frac_V2 = 66,7% (todo o query das 2 classes seen vem de pacientes do mtrn)
  e frac_V1 de 22% (k = 1) a 76% (k = 10); P-MIX tem frac_V2 = 0, mas frac_V1 maior
  (40% a 93%), porque a etapa concentra poucos pacientes por classe; P-PAC, 0 e 0.

## 13/10/2026 — Etapa 5 (decisões)

- **Decisões:** D35 (entrada na resolução nativa 700×456; riscos na seção 9.6), D36
  (cache em disco, uma vez), D37 (lote da adaptação fiel ao original; k = 1 → 1 imagem
  por passo), D38 (avaliar a época 0, só ImageNet), D39 (sem balanceamento), D40
  (arquivos `cache`, `model`, `adapt`, `train`, `evaluate`), D41 (qualquer máquina;
  dispositivo no `config.json`; uma rodada numa máquina só; sem benchmark).
- **Achado:** 74 imagens (paciente 14-12465, PC) têm 700×456, não 700×460; as de 460
  perdem 2 linhas em cima e 2 embaixo para todas ficarem 700×456 sem interpolação.
- **Rodou:** `src/cache.py` (cache de 1,6 a 1,9 GB por magnificação, ~1 min, carrega em
  0,7 s) e `tests/test_cache.py`.
- **Custo no M1:** lote de 64 a 700×456 não cabe na memória da GPU; o macOS foi para o
  swap (15 GB) e o passo levou 78 s. Com lote de 16: 0,56 s por passo (2× o de 384×384) e
  3,5 GB. Daí D42 (`get_device(max_memory_usage=0.75, allow_swap=False)`, testado: erro
  em vez de swap) e D43 (lote efetivo de 64 em micro-lotes configuráveis,
  `micro_batch_size`, padrão 16). Estimativa: ~1 min por época, ~1 a 1,5 dia para os 40
  treinos.

## 15/10/2026 — Etapa 5b: modelo, adaptação e métricas

- **Decisões:** D44 (adaptação na CPU), D45 (gerador da cabeça derivado da chave do
  episódio, mesma inicialização nos três protocolos), D46 (só `acc_img`: com query
  balanceado é igual à balanceada), D47 (embeddings do mtst gravados para a Etapa 6).
- **Rodou:** `src/model.py` (backbone, normalização, embeddings), `src/adapt.py`
  (cabeça por episódio + métricas) e `tests/test_adapt.py` (6 testes sintéticos,
  incluindo rótulos aleatórios → HM ≈ 50 e métricas conferidas à mão).
- **Integração com dados reais** (semente 0, 100×, mtst, k = 5, 100 episódios, backbone
  só ImageNet, sem meta-treino): P-PAC HM 55,9 (seen 63,1, unseen 56,0); P-IMG HM 79,5
  (seen 80,9, unseen 79,3), com frac_V1 = 0,60. Sem meta-treino o V2 não age, então a
  diferença de ~24 pontos vem do V1: mesmo com features genéricas, a cabeça acerta muito
  mais quando o paciente do query está no support. Indício preliminar, a confirmar com a
  época 0 de todas as rodadas.
- **Custo:** embeddings ~9 s para 645 imagens (MPS); adaptação ~20 ms por episódio (CPU),
  ~1 h para os 180 mil episódios.

## 15/10/2026 — Etapa 5c: meta-treino

- **Decisões:** D48 (a época 0 concorre no early stopping), D49 (o lote final incompleto
  é usado), D50 (sem retomada; `fim` no `config.json` marca rodada completa), D51 (todas
  as médias da mval por época), D52 (testes lentos com `@pytest.mark.slow`).
- **Rodou:** `src/train.py` (`treina(...)`), `avalia_episodio` no `adapt.py` (usado pela
  mval e, depois, pelo mtst) e `tests/test_train.py`: regra 2 (rápido), overfit
  proposital com 20 imagens (teste de sanidade 7: perda < 0,05) e 2 épocas de ponta a
  ponta (40×, semente 0, divisão por paciente). 22 testes rápidos + 2 lentos passam.
- **Custo real:** ~37 s por época a 40× (divisão por paciente, micro-lotes de 16, MPS);
  a época 0 (só validação) leva ~10 s.
- **Observação:** no teste de 2 épocas, a HM da mval caiu de 57,7 (época 0) para 51,8 e
  50,2. Ainda não diz nada (2 épocas, uma rodada), mas é para acompanhar no piloto: o
  early stopping com a época 0 concorrendo (D48) cobre o caso de o meta-treino piorar.

## 20/10/2026 — Windows com GPU e reprodutibilidade entre máquinas

- **Decisões do 5d:** 5d-1 (avaliar também o backbone só ImageNet no mtst), 5d-2
  (auditoria de pacientes em `.csv.gz`), 5d-3 (`config.json` da avaliação), 5d-4 e 5d-5
  (teste de sanidade 6 com embeddings reais e a rodada completa da semente 0 a 40×,
  ambos só com autorização explícita). Execuções pesadas passam a ser no Windows, via
  `ssh win`, a partir da sessão do Mac.
- **Rodou:** clone do repositório no Windows; cópia da base do Mac (7.909 imagens,
  4,1 GB, ~5 min); pipeline de dados completo no Windows (`build_csv`, `02_clear_csv`,
  `splits`, `cache`), com a correção `as_posix()` nos caminhos do CSV.
- **Resultado:** conteúdo idêntico ao do Mac em todos os CSVs (hashes iguais ignorando o
  fim de linha `\r\n`) e no cache de imagens (bit a bit). Divisões e episódios não
  dependem da máquina; só o treino na GPU pode variar (D18).
- **GPU (D53):** RX 7600 confirmada como `gfx1102`. Do índice estável da AMD, o par
  `torch 2.14` + `torchvision 0.28` é incompatível; o par `2.13` + `0.28` funciona. O
  MIOpen falha ao compilar a BatchNorm (cabeçalho ausente no pacote para Windows);
  desligado, o PyTorch usa os próprios kernels. Lote 16 a 700×456: 0,21 s por passo
  (2,7× o M1); lote 32: 0,44 s; lote 64 dá erro de memória imediato com o limite de 75%.
  Estimativa: ~15 s por época, contra ~45 s a 1 min no M1. Os 22 testes rápidos passam
  na RX 7600.

## 21/10/2026 — Etapa 5 concluída: avaliação e piloto (semente 0, 40×)

- **Rodou:** `src/evaluate.py` (5d) e `tests/test_evaluate.py`; os 3 testes lentos na
  RX 7600 (1min20s, todos passam); a rodada completa da semente 0 a 40× no Windows em
  39 min (treinos `pac` e `img` de ~10 e ~12 min, ~16 s por época; avaliação de ~17 min).
- **Reprodutibilidade:** no teste de ponta a ponta, perda e HM da mval por época
  coincidiram entre M1 e RX 7600 (ex.: época 1, perda 0,5794 e HM 51,8 nas duas).
- **Treinos:** `pac` parou na época 34 (melhor 24, HM da mval 58,9 contra 57,7 na época
  0; a perda do mtrn foi a ~0,002); `img` parou na 45 (melhor 35, HM da mval 95,1 contra
  84,3 na época 0), sempre com frac_V2 = 0,67 na mval.
- **Resultado no mtst (HM, meta, k = 1/5/10):** P-IMG 80,9/91,5/94,8; P-MIX
  75,5/91,0/95,8; P-PAC 55,6/56,7/56,1. Só ImageNet: P-PAC 56,9/58,1/58,0; P-IMG
  68,8/84,8/90,4. Δ_total (meta) de 25 a 39 pontos (+45% a +69%). Tabela completa e
  leitura na seção 16 do PROJETO.md.
- **Análise:** o meta-treino só ajuda no P-IMG (V2); no P-PAC troca AUC unseen por seen
  e perde um pouco na HM; a HM por episódio varia muito (dp ~20) e depende mais dos
  pacientes sorteados que do backbone; AUC do TA abaixo de 50 no P-PAC sugere atalho pela
  lâmina.
- **Achado metodológico:** a decomposição Δ_V2 ≈ P-IMG − P-MIX não funciona, porque o
  P-MIX tem mais V1 que o P-IMG; registrada como P7 (seção 6.2), a decidir na Etapa 9.
- **Falta:** levar o piloto ao orientador (Etapa 7); contraprova (Etapa 6); grade
  completa (Etapa 8, ~13 a 15 h no Windows; a avaliação na CPU é o gargalo).

## 21/10/2026 — Etapa 6: contraprova com o scikit-learn

- **Rodou:** PR #3 (Etapa 5) mesclada na `main`. Embeddings do piloto copiados do
  Windows (4 arquivos, ~11 MB); `data/splits/seed0.csv` idêntico nas duas máquinas.
  `notebooks/03_contraprova.ipynb`: `OneVsRestClassifier(LogisticRegression(C=1))`
  nos mesmos embeddings e episódios do `40_seed0.csv` (18.000 episódios, ~5 min na CPU
  do M1, sem aviso de convergência), com as probabilidades por `sigmoid(decision_function)`
  e as métricas da mesma `metricas()` do `adapt.py`.
- **Conferência:** a cabeça em PyTorch, rodada de novo no Mac em 54 episódios, reproduz o
  CSV do Windows (diferença máxima de 7·10⁻⁶ na HM).
- **Resultado:** HM média das duas cabeças a no máximo 0,64 ponto (meta, P-IMG, k = 1);
  nos outros grupos, de 0 a 0,5. Correlação da HM por episódio de 0,98 a 0,99 com o
  backbone do ImageNet e de 0,93 a 0,97 com o meta-treinado; nele, a diferença por episódio
  é maior (mediana de 2 a 3 pontos no P-PAC), sinal de que a cabeça é mais sensível à
  regularização sobre essas features. Δ_total (P-IMG − P-PAC) igual nas duas cabeças, com
  diferença de no máximo 0,9 ponto: as conclusões do piloto não dependem da implementação
  da cabeça.
- **Notas:** o notebook roda com `uv run --with nbconvert jupyter nbconvert --execute`,
  sem dependência nova no `pyproject.toml`.
- **Falta:** levar o piloto ao orientador (Etapa 7); grade completa (Etapa 8).
