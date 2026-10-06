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
