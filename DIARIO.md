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
