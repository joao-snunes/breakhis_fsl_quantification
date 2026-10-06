# breakhis_fsl_quantification

Quantificação do vazamento de dados **por imagem × por paciente** em avaliação
few-shot de imagens histológicas H&E (BreakHis), com o método BatchBased +
MobileNetV3-Small-0.75 do artigo MetaChest (Montalvo-Lezama & Fuentes-Pineda, 2026).

## Ambiente

Requer [uv](https://docs.astral.sh/uv/). O `torch` do PyPI já cobre CPU, Apple
Silicon (MPS) e NVIDIA (CUDA no Linux).

```bash
uv sync
uv run python -m src.device   # mostra a versão do torch e o dispositivo escolhido
uv run pytest
```

### Windows com GPU AMD

Duas rotas, nesta ordem de preferência:

1. **ROCm**: instalar as wheels de PyTorch para ROCm da AMD (projeto
   [TheRock](https://github.com/ROCm/TheRock)) com `uv pip install`, seguindo a
   documentação atual da AMD para a sua placa. O PyTorch-ROCm aparece como `cuda`.
2. **DirectML** (fallback): `uv pip install torch-directml`.

Nos dois casos, o `uv run` reinstalaria o `torch` do lock; use `uv run --no-sync`
(ou `UV_NO_SYNC=1`). Confira com `uv run --no-sync python -m src.device`.
