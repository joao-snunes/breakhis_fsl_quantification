"""Escolha do dispositivo de cálculo.

Ordem do modo "auto": CUDA → MPS → DirectML → CPU.
- CUDA: GPU NVIDIA e também AMD com ROCm (o PyTorch-ROCm se apresenta como "cuda").
- MPS: GPU integrada dos Macs com Apple Silicon.
- DirectML: GPU AMD quando não há ROCm (pacote opcional torch-directml).
"""

import torch


def _directml():
    import torch_directml  # opcional: só existe se instalado à parte
    return torch_directml.device()


def get_device(name: str = "auto") -> torch.device:
    """Devolve o dispositivo pedido; "auto" escolhe o melhor disponível."""
    if name == "directml":
        return _directml()
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        return _directml()
    except ImportError:
        return torch.device("cpu")


if __name__ == "__main__":
    print(torch.__version__, get_device())
