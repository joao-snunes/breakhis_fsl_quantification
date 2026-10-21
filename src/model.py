"""backbone e embeddings.

backbone = MobileNetV3-Small-0.75 pré-treinada no ImageNet-1K, sem a camada de classificação
(num_classes=0). Usa embbeding de dimensão 1.024.
"""

import timm
import torch

NOME = "tf_mobilenetv3_small_075.in1k"
D = 1024  # dimensão do embedding (head_hidden_size)


def cria_backbone(pretrained=True):
    return timm.create_model(NOME, pretrained=pretrained, num_classes=0)


def normaliza(imgs, backbone, device):
    """uint8 (N, H, W, 3) → float (N, 3, H, W) normalizado com a média e o desvio do pré-treino."""
    cfg = backbone.pretrained_cfg
    media = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    desvio = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    x = torch.as_tensor(imgs).to(device).permute(0, 3, 1, 2).float() / 255
    return (x - media) / desvio


def embeddings(backbone, imgs, device, batch_size=16):
    """Embeddings (N, 1024) na CPU, com o backbone congelado, em micro-lotes de batch_size imagens."""
    backbone.eval()
    with torch.no_grad():
        partes = [backbone(normaliza(imgs[i:i + batch_size], backbone, device)).cpu()
                  for i in range(0, len(imgs), batch_size)]
    return torch.cat(partes)
