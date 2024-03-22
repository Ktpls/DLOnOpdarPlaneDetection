# %%
# basics
from .nntrackerdev_predef import *

# %%
# nn def

# %%
# nn def


class ParameterRequiringGradModule(torch.nn.Module):
    def parameters(
        self, recurse: bool = True
    ) -> typing.Iterator[torch.nn.parameter.Parameter]:
        return filter(
            lambda x: x.requires_grad is not False, super().parameters(recurse)
        )


def PositionalEmbedding2D(shape, depth, len_max=None):
    if len_max is None:
        len_max = 10000
    h, w = shape
    y = np.arange(0, h).reshape([-1, 1, 1])
    x = np.arange(0, w).reshape([1, -1, 1])
    d = np.arange(0, depth).reshape([1, 1, -1])
    pe = np.zeros(list(shape) + [depth], np.float32)

    def BroadCastToBeLikePe(vec):
        return np.zeros_like(pe) + vec

    d_like_pe = BroadCastToBeLikePe(d)

    sinx = np.sin(x / (len_max ** (d / depth)))
    cosx = np.cos(x / (len_max ** ((d - 1) / depth)))
    siny = np.sin(y / (len_max ** (d / depth)))
    cosy = np.cos(y / (len_max ** ((d - 1) / depth)))
    evens = BroadCastToBeLikePe(sinx) + BroadCastToBeLikePe(siny)
    odds = BroadCastToBeLikePe(cosx) + BroadCastToBeLikePe(cosy)
    pe[d_like_pe % 2 == 0] = evens[d_like_pe % 2 == 0]
    pe[d_like_pe % 2 == 1] = odds[d_like_pe % 2 == 1]

    return pe.reshape([1, -1, depth])


class SemanticInjectionModule(torch.nn.Module):
    def __init__(self, localdim, globaldim=None, outdim=None):
        if globaldim is None:
            globaldim = localdim
        if outdim is None:
            outdim = localdim
        self.localdim = localdim
        self.globaldim = globaldim
        self.outdim = outdim
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(localdim, outdim, 1),
        )
        self.global1 = nn.Sequential(
            nn.Conv2d(globaldim, outdim, 1),
            nn.Sigmoid(),
        )
        self.global2 = nn.Sequential(
            nn.Conv2d(globaldim, outdim, 1),
        )
        self.bn = nn.BatchNorm2d(outdim)

    def forward(self, local, globalsemantic):
        x = self.local(local) * self.global1(globalsemantic) + self.global2(
            globalsemantic
        )
        return self.bn(x)


class AddPositionalEmbedding(nn.Module):
    def __init__(self, shape, depth, len_max=None):
        super().__init__()
        self.pe = nn.Parameter(
            torch.tensor(
                PositionalEmbedding2D(shape, depth, len_max),
                dtype=torch.float32,
                requires_grad=False,
            )
        )

    def forward(self, x):
        return x + self.pe


class PermuteModule(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class nntracker_pi(torch.nn.Module):
    resize_size = 128
    interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    antialias = True

    def __init__(self):
        super().__init__()

        useBn = True
        incver = "v3"
        self.mod = nn.Sequential(
            # 128
            inception.even(3, 8, bn=useBn, version=incver),
            res_through(
                inception.even(8, 8, bn=useBn, version=incver),
                inception.even(8, 8, bn=useBn, version=incver),
            ),
            nn.MaxPool2d(2),  # 64
            inception.even(8, 16, bn=useBn, version=incver),
            res_through(
                inception.even(16, 16, bn=useBn, version=incver),
                inception.even(16, 16, bn=useBn, version=incver),
            ),
            nn.MaxPool2d(2),  # 32
            inception.even(16, 32, bn=useBn, version=incver),
            res_through(
                inception.even(32, 32, bn=useBn, version=incver),
                inception.even(32, 32, bn=useBn, version=incver),
            ),
            nn.MaxPool2d(2),  # 16
            inception.even(32, 64, bn=useBn, version=incver),
            res_through(
                inception.even(64, 64, bn=useBn, version=incver),
                inception.even(64, 64, bn=useBn, version=incver),
            ),
            PermuteModule((0, 2, 3, 1)),
            nn.Flatten(1, 2),  # keep depth unflattened
            AddPositionalEmbedding((16, 16), 64, None),
            nn.TransformerEncoderLayer(64, 8, 16),
            nn.TransformerEncoderLayer(64, 8, 16),
            nn.TransformerEncoderLayer(64, 8, 16),
            nn.TransformerEncoderLayer(64, 8, 16),
            nn.Flatten(1, -1),
            nn.Linear(64 * 16**2, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 4),
            nn.LeakyReLU(),
        )

    def forward(self, m):
        m = TTF.resize(
            m,
            size=nntracker_pi.resize_size,
            interpolation=nntracker_pi.interpolation,
            antialias=nntracker_pi.antialias,
        )
        out = self.mod(m)
        return out


class nntracker_respi(ParameterRequiringGradModule):
    def __init__(
        self,
        frozenLayers=(
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ),
    ):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        backbone = torchvision.models.resnet18(weights=weights)
        backboneOutShape = 512
        for name, param in backbone.named_parameters():
            matched = False
            for fl in frozenLayers:
                if name.startswith(fl):
                    param.requires_grad = False
                    matched = True
                    break
            if not matched:
                param.requires_grad = True
        self.backbone = backbone
        self.backbonepreproc = weights.transforms()

        self.mod = nn.Sequential(
            res_through(
                nn.Linear(backboneOutShape, backboneOutShape),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(backboneOutShape, backboneOutShape),
                nn.LeakyReLU(),
                nn.Dropout(),
            ),
            nn.Linear(backboneOutShape, 4),
            nn.LeakyReLU(),
        )

    def forward(self, m):
        m = self.backbonepreproc(m)
        m = self.backbone.conv1(m)
        m = self.backbone.bn1(m)
        m = self.backbone.relu(m)
        m = self.backbone.maxpool(m)
        m = self.backbone.layer1(m)
        m = self.backbone.layer2(m)
        m = self.backbone.layer3(m)
        m = self.backbone.layer4(m)
        m = self.backbone.avgpool(m)
        m = torch.flatten(m, 1)
        out = self.mod(m)
        return out


def getmodel(modelpath, device, **kwargs):
    model = setModule(nntracker_respi(**kwargs), path=modelpath, device=device)
    print(model)
    return model


def getDevice():
    print(getDeviceInfo())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device
