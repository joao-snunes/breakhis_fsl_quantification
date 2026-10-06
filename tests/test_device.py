"""Etapa 0: a MobileNet do artigo roda um passo de treino no dispositivo da máquina."""

import timm
import torch

from src.device import get_device


def test_train_step_on_device():
    device = get_device()
    # pretrained=False: o teste não depende de internet; a arquitetura é a mesma
    backbone = timm.create_model("tf_mobilenetv3_small_075", pretrained=False, num_classes=0)
    head = torch.nn.Linear(backbone.head_hidden_size, 4)
    model = torch.nn.Sequential(backbone, head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 384, 384, device=device)
    y = torch.eye(4, device=device)[[0, 1]]  # rótulos one-hot
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert backbone.head_hidden_size == 1024  # embedding; num_features (432) é outra coisa, ver PROJETO.md 9.1
    assert torch.isfinite(loss)
