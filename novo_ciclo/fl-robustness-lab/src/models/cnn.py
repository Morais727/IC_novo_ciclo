"""CNN leve para CIFAR-10.

Mantida pequena de propósito (~190k parâmetros): permite que os agregadores
que operam sobre o vetor completo de parâmetros (BOBA/SVD, whitening,
distâncias par-a-par do Krum) rodem em GPUs modestas (ex: 2GB VRAM) e em CPU
em tempo hábil para experimentação. `fc1` corresponde à camada tipicamente
identificada na literatura como "Backdoor-Critical layer" em CNNs.
"""
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3, side: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 32x16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 64x8x8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 64x4x4 (FMNIST: 64x3x3)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * side * side, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class RagnarCNN(nn.Module):
    """CNN do artigo do RAGNAR (CoUrb-2024/cliente_principal.py, versão PyTorch da CNN Keras).

    Convoluções sem padding, pool só após as duas primeiras, fc1 com 64 unidades.
    10 tensores de parâmetros; o RAGNAR do CoUrb usa os 9 primeiros -> 54 features (6 por tensor).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(self.flatten(x)))
        return self.fc2(x)


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.gn1 = nn.GroupNorm(4, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.gn2 = nn.GroupNorm(4, cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(4, cout))

    def forward(self, x):
        out = nn.functional.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return nn.functional.relu(out + (x if self.short is None else self.short(x)))


class ResNet20GN(nn.Module):
    """ResNet-20 de CIFAR (He et al., 2016; 3 estágios 16/32/64 × 3 blocos, ~0,27 M parâmetros, 65 tensores) com
    GroupNorm no lugar de BatchNorm — prática comum em FL não-IID (Hsieh et al., 2020), e evita médias móveis fora
    do vetor de parâmetros. Fase 5 (21/10/2026): modelo mais profundo para testar a defesa por tensor."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, 1, 1, bias=False); self.gn1 = nn.GroupNorm(4, 16)
        layers, cin = [], 16
        for cout, stride in ((16, 1), (32, 2), (64, 2)):
            for k in range(3):
                layers.append(_Block(cin, cout, stride if k == 0 else 1)); cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.gn1(self.conv1(x)))
        x = self.layers(x)
        return self.fc(nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


class ResNet18GN(nn.Module):
    """ResNet-18 de CIFAR (He et al., 2016; 4 estágios 64/128/256/512 × 2 blocos, ~11,2 M parâmetros) com GroupNorm,
    como a ResNet20GN. Fase 8 (27/10/2026): uma das redes do artigo do LP (Zhuang et al., ICLR 2024), para medir o
    overhead e testar o LP em rede grande."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False); self.gn1 = nn.GroupNorm(4, 64)
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            for k in range(2):
                layers.append(_Block(cin, cout, stride if k == 0 else 1)); cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.gn1(self.conv1(x)))
        x = self.layers(x)
        return self.fc(nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


class VGG19GN(nn.Module):
    """VGG-19 de CIFAR (Simonyan e Zisserman, 2014; 16 convoluções + 1 linear, ~20 M parâmetros) com GroupNorm após
    cada convolução. Fase 8: a outra rede do artigo do LP."""

    CFG = (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M")

    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        layers, cin = [], in_ch
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, ceil_mode=True))   # ceil: FMNIST 28x28 chega a 1x1, não 0x0
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1), nn.GroupNorm(4, v), nn.ReLU(inplace=True)]; cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(nn.functional.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


class _MBConv(nn.Module):
    """Bloco MBConv do EfficientNet (Tan e Le, 2019): expansão 1x1 -> conv depthwise -> squeeze-excite ->
    projeção 1x1, com conexão residual quando stride=1 e cin=cout. BatchNorm trocada por GroupNorm (padrão do
    lab p/ FL não-IID); ativação SiLU (swish), como no artigo. SE sem normalização (dois 1x1 + sigmoide)."""

    def __init__(self, cin, cout, expand, stride, kernel, se_ratio=0.25):
        super().__init__()
        cmid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, cmid, 1, bias=False), nn.GroupNorm(4, cmid), nn.SiLU(inplace=True)]
        layers += [nn.Conv2d(cmid, cmid, kernel, stride, kernel // 2, groups=cmid, bias=False),
                   nn.GroupNorm(4, cmid), nn.SiLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        se = max(1, int(cin * se_ratio))
        self.se = nn.Sequential(nn.Conv2d(cmid, se, 1), nn.SiLU(inplace=True),
                                nn.Conv2d(se, cmid, 1), nn.Sigmoid())
        self.project = nn.Sequential(nn.Conv2d(cmid, cout, 1, bias=False), nn.GroupNorm(4, cout))

    def forward(self, x):
        out = self.conv(x)
        out = out * self.se(nn.functional.adaptive_avg_pool2d(out, 1))
        out = self.project(out)
        return out + x if self.use_res else out


class EfficientNetGN(nn.Module):
    """EfficientNet-B0 (Tan e Le, ICML 2019) com GroupNorm no lugar de BatchNorm — o 2º modelo do artigo do
    LaySelFL. Adaptado a imagens pequenas: stem com stride 1 (em vez de 2), para 32x32 (CIFAR) e 28x28 (FMNIST)
    não colapsarem cedo demais; o resto segue a configuração B0 (7 estágios de MBConv). ~4 M parâmetros.
    Todos os canais são divisíveis por 4 (exigência do GroupNorm(4, ·))."""

    #        (expand, cout, nº camadas, stride, kernel)
    CFG = ((1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3),
           (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3))

    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False),
                                  nn.GroupNorm(4, 32), nn.SiLU(inplace=True))
        blocks, cin = [], 32
        for expand, cout, n, stride, k in self.CFG:
            for i in range(n):
                blocks.append(_MBConv(cin, cout, expand, stride if i == 0 else 1, k)); cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(cin, 1280, 1, bias=False),
                                  nn.GroupNorm(4, 1280), nn.SiLU(inplace=True))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.head(self.blocks(self.stem(x)))
        return self.fc(nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


def build_model(name: str = "smallcnn", num_classes: int = 10, dataset: str = "cifar10") -> nn.Module:
    in_ch, side = (1, 3) if dataset == "fmnist" else (3, 4)
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes, in_ch=in_ch, side=side)
    if name == "resnet20gn":
        return ResNet20GN(num_classes=num_classes, in_ch=in_ch)
    if name == "resnet18gn":
        return ResNet18GN(num_classes=num_classes, in_ch=in_ch)
    if name == "vgg19gn":
        return VGG19GN(num_classes=num_classes, in_ch=in_ch)
    if name == "efficientnetgn":
        return EfficientNetGN(num_classes=num_classes, in_ch=in_ch)
    if name == "cnn_ragnar":
        return RagnarCNN(num_classes=num_classes)
    raise ValueError(f"Modelo desconhecido: {name}")
