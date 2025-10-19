import torch
import torch.nn as nn
from torchvision.ops import stochastic_depth
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models.resnet import ResNet, Bottleneck


class BottleneckSD(Bottleneck):
    """Bottleneck avec stochastic depth sur la branche résiduelle (main path)."""
    def __init__(self, *args, drop_prob: float = 0.0, sd_mode: str = "row", **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_prob = drop_prob      # proba de drop pour CE bloc
        self.sd_mode = sd_mode          # "row" = par échantillon, "batch" = même masque pour tout le batch

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # --- Stochastic depth sur la branche résiduelle ---
        if self.training and self.drop_prob > 0.0:
            out = stochastic_depth(out, self.drop_prob, self.sd_mode, True)
        # ---------------------------------------------------

        out += identity
        out = self.relu(out)
        return out


def resnet152_with_stochastic_depth(p_L: float = 0.5, sd_mode: str = "batch", pretrained: bool = False):
    """
    Construit un ResNet-152 avec stochastic depth.
    p_L : probabilité max au dernier bloc (décroissance linéaire 0 -> p_L).
    sd_mode : "row" (par échantillon) ou "batch".
    pretrained : charge les poids ImageNet officiels et les mappe sur ce modèle (strict=False).
    """
    # Architecture ResNet-152 : [3, 8, 36, 3] bottlenecks
    layers = [3, 8, 36, 3]
    model = ResNet(block=BottleneckSD, layers=layers)

    # Nombre total de blocs
    L = sum(layers)
    idx = 0

    # Assigner une proba par bloc avec décroissance linéaire (0 -> p_L)
    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for m in stage:
            idx += 1
            m.drop_prob = p_L * (idx / L)
            m.sd_mode = sd_mode

    # Optionnel : charger des poids pré-entraînés officiels
    if pretrained:
        ref = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
        # `missing` contiendra les nouveaux attributs (drop_prob, sd_mode) non présents dans les poids
    
    return model


class ClassifierMixture(nn.Module):
    def __init__(self, in_dim=2048, num_classes=1000, n_heads=3, p_active=0.33, seed=0):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(in_dim, num_classes) for _ in range(n_heads)])
        self.p_active = p_active
        self.g = torch.Generator()
        self.seed = seed
        
    def forward(self, feat):
        # Sélection aléatoire d’une seule tête (ou basée sur un routeur appris)
        self.g.manual_seed(self.seed)
        self.seed += 1
        if self.training and torch.rand((), generator=self.g) < 1 - self.p_active:
            active = 0
        else:
            active = torch.randint(1, len(self.heads), (), generator=self.g).item()
        return self.heads[active](feat)

def add_conditional_heads(m: nn.Module, n_heads=3):
    # Remplace la fc par une mixture conditionnelle
    in_dim = m.fc.in_features
    num_classes = m.fc.out_features
    m.fc = ClassifierMixture(in_dim, num_classes, n_heads=n_heads)
    return m


