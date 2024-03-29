# %%
# basics
from .nntrackerdev_predef import *


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
    """
    got pic
    preproc
    scale0
    |---zoom to scale1
    |   |---zoom to scale2
    |   |   |
    #---#---to scale0
    |
    #summary
    |---zoom to scale1
    |   |---zoom to scale2
    |   |   |
    """

    sizeInput = 128
    interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    antialias = False
    useBn = False
    incver = "v2"

    def __init__(self):
        super().__init__()
        incept = functools.partial(
            inception.even, bn=nntracker_pi.useBn, version=nntracker_pi.incver
        )

        chanS0 = 8
        factorS0 = 4
        sizeS0 = nntracker_pi.sizeInput // factorS0
        chanS1 = 32
        factorS1 = 2
        sizeS1 = sizeS0 // factorS1
        chanS2 = 128
        factorS2 = 2
        sizeS2 = sizeS1 // factorS2
        self.preproc = nn.Sequential(
            incept(3, chanS0),
            nn.MaxPool2d(factorS0),
        )
        self.s0tos1 = nn.Sequential(
            nn.MaxPool2d(factorS1),
        )
        self.s1tos2 = nn.Sequential(
            nn.MaxPool2d(factorS2),
        )
        self.featExtS0 = nn.Sequential(
            incept(chanS0, chanS0),
            incept(chanS0, chanS0),
        )
        self.s2tos1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=factorS2),
        )
        self.s1tos0 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=factorS1),
        )
        self.featExtS1 = nn.Sequential(
            incept(chanS0, chanS1),
            incept(chanS1, chanS1),
            incept(chanS1, chanS1),
        )
        self.featExtS2 = nn.Sequential(
            incept(chanS0, chanS2),
            incept(chanS2, chanS2),
            incept(chanS2, chanS2),
        )
        chanSummary = chanS2
        self.scaleSummary = nn.Sequential(
            inception.even(
                chanS0 + chanS1 + chanS2,
                chanSummary,
                bn=nntracker_pi.useBn,
                version=nntracker_pi.incver,
            )
        )
        # self.featExtS0Sum = nn.Sequential(
        #     incept(chanSummary + chanS0, chanS0),
        #     incept(chanS0, chanS0),
        # )
        # self.featExtS1Sum = nn.Sequential(
        #     incept(chanSummary + chanS1, chanS1),
        #     incept(chanS1, chanS1),
        # )
        # self.featExtS2Sum = nn.Sequential(
        #     incept(chanSummary + chanS2, chanS2),
        #     incept(chanS2, chanS2),
        # )

        self.head = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(chanSummary * sizeS2**2, 1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 4),
            nn.LeakyReLU(),
        )

    def forward(self, m):
        def concatChan(*l):
            return torch.concat(l, dim=1)

        # preproc
        m = TTF.resize(
            m,
            size=nntracker_pi.sizeInput,
            interpolation=nntracker_pi.interpolation,
            antialias=nntracker_pi.antialias,
        )
        s0 = self.preproc(m)
        s1 = self.s0tos1(s0)
        s2 = self.s1tos2(s1)
        # featExt
        s0 = self.featExtS0(s0)
        s1 = self.featExtS1(s1)
        s2 = self.featExtS2(s2)
        summary = self.scaleSummary(
            concatChan(
                self.s1tos2(self.s0tos1(s0)),
                self.s1tos2(s1),
                s2,
            )
        )
        # featExtSum
        # summary_s0 = self.s1tos0(summary)
        # s0 = self.featExtS0Sum(concatChan(summary_s0, s0))
        # summary_s1 = summary
        # s1 = self.featExtS1Sum(concatChan(summary_s1, s1))
        # summary_s2 = self.s1tos2(summary_s1)
        # s2 = self.featExtS2Sum(concatChan(summary_s2, s2))
        # scaleAll = concatChan(
        #     torch.flatten(s0, 1), torch.flatten(s1, 1), torch.flatten(s2, 1)
        # )
        out = self.head(summary)
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
                nn.Sequential(
                    nn.Linear(backboneOutShape, backboneOutShape),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                ),
                nn.Sequential(
                    nn.Linear(backboneOutShape, backboneOutShape),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                ),
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


def getmodel(model0: torch.nn.Module, modelpath, device):
    model = setModule(model0, path=modelpath, device=device)
    print(
        f"#param={np.sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])}"
    )
    # print(model)
    return model


def getDevice():
    print(getDeviceInfo())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device
