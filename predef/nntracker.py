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


class nntracker_pi(ParameterRequiringGradModule):
    resize_size = 128
    interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    antialias = True

    class CertainScale(nn.Module):
        def __init__(self, zoom, proc) -> None:
            super().__init__()
            self.zoom = zoom
            self.proc = proc

    def __init__(self):
        super().__init__()

        useBn = True
        incver = "v3"
        chanS0 = 8
        sizeS0 = 32
        chanS1 = 32
        sizeS1 = 16
        chanS2 = 128
        sizeS2 = 8
        dimPerHead = 4
        self.scale0 = nntracker_pi.CertainScale(
            zoom=nn.Sequential(
                inception.even(3, chanS0, bn=useBn, version=incver),
                res_through(
                    inception.even(chanS0, chanS0, bn=useBn, version=incver),
                ),
                # 128
                nn.MaxPool2d(4),
                # 32
                res_through(
                    inception.even(chanS0, chanS0, bn=useBn, version=incver),
                ),
            ),
            proc=nn.Sequential(
                res_through(
                    inception.even(chanS0, chanS0, bn=useBn, version=incver),
                    inception.even(chanS0, chanS0, bn=useBn, version=incver),
                ),
                PermuteModule((0, 2, 3, 1)),
                nn.Flatten(1, 2),  # keep depth unflattened
                AddPositionalEmbedding((sizeS0, sizeS0), chanS0, sizeS0),
                nn.TransformerEncoderLayer(chanS0, chanS0 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS0, chanS0 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS0, chanS0 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS0, chanS0 // dimPerHead, 16),
                nn.Flatten(1, -1),
            ),
        )
        self.scale1 = nntracker_pi.CertainScale(
            zoom=nn.Sequential(
                inception.even(chanS0, chanS1, bn=useBn, version=incver),
                res_through(
                    inception.even(chanS1, chanS1, bn=useBn, version=incver),
                ),
                # 32
                nn.MaxPool2d(2),
                # 16
                res_through(
                    inception.even(chanS1, chanS1, bn=useBn, version=incver),
                ),
            ),
            proc=nn.Sequential(
                res_through(
                    inception.even(chanS1, chanS1, bn=useBn, version=incver),
                    inception.even(chanS1, chanS1, bn=useBn, version=incver),
                ),
                PermuteModule((0, 2, 3, 1)),
                nn.Flatten(1, 2),
                AddPositionalEmbedding((sizeS1, sizeS1), chanS1, sizeS1),
                nn.TransformerEncoderLayer(chanS1, chanS1 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS1, chanS1 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS1, chanS1 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS1, chanS1 // dimPerHead, 16),
                nn.Flatten(1, -1),
            ),
        )
        self.scale2 = nntracker_pi.CertainScale(
            zoom=nn.Sequential(
                inception.even(chanS1, chanS2, bn=useBn, version=incver),
                res_through(
                    inception.even(chanS2, chanS2, bn=useBn, version=incver),
                ),
                # 16
                nn.MaxPool2d(2),
                res_through(
                    inception.even(chanS2, chanS2, bn=useBn, version=incver),
                ),
                # 8
            ),
            proc=nn.Sequential(
                res_through(
                    inception.even(chanS2, chanS2, bn=useBn, version=incver),
                    inception.even(chanS2, chanS2, bn=useBn, version=incver),
                ),
                PermuteModule((0, 2, 3, 1)),
                nn.Flatten(1, 2),
                AddPositionalEmbedding((sizeS2, sizeS2), chanS2, sizeS2),
                nn.TransformerEncoderLayer(chanS2, chanS2 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS2, chanS2 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS2, chanS2 // dimPerHead, 16),
                nn.TransformerEncoderLayer(chanS2, chanS2 // dimPerHead, 16),
                nn.Flatten(1, -1),
            ),
        )
        self.head = nn.Sequential(
            nn.Linear(
                chanS0 * sizeS0**2 + chanS1 * sizeS1**2 + chanS2 * sizeS2**2, 4096
            ),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 4),
            nn.LeakyReLU(),
        )

    def forward(self, m):
        # preproc
        m = TTF.resize(
            m,
            size=nntracker_pi.resize_size,
            interpolation=nntracker_pi.interpolation,
            antialias=nntracker_pi.antialias,
        )
        s0 = self.scale0.zoom(m)
        s1 = self.scale1.zoom(s0)
        s2 = self.scale2.zoom(s1)
        s0 = self.scale0.proc(s0)
        s1 = self.scale1.proc(s1)
        s2 = self.scale2.proc(s2)
        out = torch.concat([s0, s1, s2], dim=1)
        out = self.head(out)
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
        loadPretrainedBackbone=True,
    ):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        backbone = torchvision.models.resnet18(
            weights=weights if loadPretrainedBackbone else None
        )
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


def getmodel(model0, modelpath, device):
    model = setModule(model0, path=modelpath, device=device)
    print(model)
    return model


def getDevice():
    print(getDeviceInfo())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device
