"""Escolha do dispositivo de cálculo e limite de memória.

Ordem do modo "auto": CUDA → MPS → DirectML → CPU.
- CUDA: GPU NVIDIA e também AMD com ROCm (o PyTorch-ROCm se apresenta como "cuda").
- MPS: GPU integrada dos Macs com Apple Silicon.
- DirectML: GPU AMD quando não há ROCm (pacote opcional torch-directml).
"""

import torch


def _directml():
    import torch_directml 
    return torch_directml.device()


def _escolhe(name):
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


def get_device(name: str = "auto", max_memory_usage: float = 0.75, allow_swap: bool = False) -> torch.device:
    """Devolve o dispositivo pedido ("auto" escolhe o melhor) e aplica o limite de memória."""
    assert 0 < max_memory_usage <= 1, "max_memory_usage é uma fração entre 0 e 1"
    device = _escolhe(name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(max_memory_usage)
    elif device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.0 if allow_swap else max_memory_usage)  # 0 = sem limite
    return device


if __name__ == "__main__":
    print(torch.__version__, get_device())
