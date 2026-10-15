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
