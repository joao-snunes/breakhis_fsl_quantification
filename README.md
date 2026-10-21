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
