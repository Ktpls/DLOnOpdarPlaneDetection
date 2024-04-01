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


class GlobalAvePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def static_forward(x):
        return torch.mean(x, dim=(2, 3))

    def forward(self, x):
        return GlobalAvePooling.static_forward(x)


class ConvNext(nn.Module):
    def __init__(self, infeat, outfeat, ks=3):
        super().__init__()
        self.conv = nn.Conv2d(infeat, outfeat, ks, padding="same")

    def forward(self, x):
        return self.conv(x)


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
    useBn = True
    incver = "v2"

    def __init__(self):
        super().__init__()
        vp_incept = lambda infeat, outfeat: (
            inception.even(
                infeat=infeat,
                outfeat=outfeat,
                bn=nntracker_pi.useBn,
                version=nntracker_pi.incver,
            )
        )
        vp_conv = lambda infeat, outfeat: (
            (
                nn.Sequential(
                    nn.Conv2d(infeat, outfeat, 3, padding="same"),
                    nn.BatchNorm2d(outfeat),
                )
            )
            if nntracker_pi.useBn
            else nn.Conv2d(infeat, outfeat, 3, padding="same")
        )
        vp_next = lambda infeat, outfeat: ()
        visionProcessor = vp_conv

        chanS0 = 16
        factorS0 = 4
        chanS1 = 32
        factorS1 = 2
        chanS2 = 64
        factorS2 = 2
        self.preproc = nn.Sequential(
            visionProcessor(3, chanS0),
            nn.MaxPool2d(factorS0),
        )
        self.s0tos1 = nn.Sequential(
            nn.MaxPool2d(factorS1),
        )
        self.s1tos2 = nn.Sequential(
            nn.MaxPool2d(factorS2),
        )
        self.s2tos1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=factorS2),
        )
        self.s1tos0 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=factorS1),
        )
        self.featExtS0 = OneShotAggregationResThrough(
            visionProcessor(chanS0, chanS0),
            visionProcessor(chanS0, chanS0),
            chanTotal=3 * chanS0,
            chanDest=chanS0,
        )
        self.featExtS1 = OneShotAggregationResThrough(
            visionProcessor(chanS0, chanS1),
            visionProcessor(chanS1, chanS1),
            visionProcessor(chanS1, chanS1),
            chanTotal=chanS0 + 3 * chanS1,
            chanDest=chanS1,
        )

        self.featExtS2 = OneShotAggregationResThrough(
            visionProcessor(chanS0, chanS2),
            visionProcessor(chanS2, chanS2),
            visionProcessor(chanS2, chanS2),
            chanTotal=chanS0 + 3 * chanS2,
            chanDest=chanS2,
        )

        chanBroadcast = chanS0 + chanS1 + chanS2
        self.broadcast = OneShotAggregationResThrough(
            visionProcessor(chanS0 + chanS1 + chanS2, chanBroadcast),
            visionProcessor(chanBroadcast, chanBroadcast),
            visionProcessor(chanBroadcast, chanBroadcast),
            chanTotal=(chanS0 + chanS1 + chanS2) + 3 * chanBroadcast,
            chanDest=chanBroadcast,
        )

        self.featExtS0Br = OneShotAggregationResThrough(
            visionProcessor(chanBroadcast + chanS0, chanS0),
            visionProcessor(chanS0, chanS0),
            visionProcessor(chanS0, chanS0),
            chanTotal=chanBroadcast + chanS0 + 3 * chanS0,
            chanDest=chanS0,
        )

        self.featExtS1Br = OneShotAggregationResThrough(
            visionProcessor(chanBroadcast + chanS1, chanS1),
            visionProcessor(chanS1, chanS1),
            visionProcessor(chanS1, chanS1),
            chanTotal=chanBroadcast + chanS1 + 3 * chanS1,
            chanDest=chanS1,
        )

        self.featExtS2Br = OneShotAggregationResThrough(
            visionProcessor(chanBroadcast + chanS2, chanS2),
            visionProcessor(chanS2, chanS2),
            visionProcessor(chanS2, chanS2),
            chanTotal=chanBroadcast + chanS2 + 3 * chanS2,
            chanDest=chanS2,
        )

        self.head = nn.Sequential(
            nn.Linear(chanBroadcast, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 4),
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
        # broadcast within scale
        br = self.broadcast(
            concatChan(
                s0,
                self.s1tos0(s1),
                self.s1tos0(self.s2tos1(s2)),
            )
        )
        # continue featExt
        brS0 = br
        s0 = self.featExtS0Br(concatChan(brS0, s0))
        brS1 = self.s0tos1(br)
        s1 = self.featExtS1Br(concatChan(brS1, s1))
        brS2 = self.s1tos2(brS1)
        s2 = self.featExtS2Br(concatChan(brS2, s2))
        out = self.head(
            torch.concat(
                (
                    GlobalAvePooling.static_forward(s0),
                    GlobalAvePooling.static_forward(s1),
                    GlobalAvePooling.static_forward(s2),
                ),
                dim=-1,
            )
        )
        return out


class nntracker_respi(ParameterRequiringGradModule):
    def __init__(
        self,
        freeLayers=(
            "features.15",
            "features.16",
        ),
        loadPretrainedBackbone=True,
    ):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(
            weights=weights if loadPretrainedBackbone else None
        )
        backboneOutShape = 960
        self.backbone = backbone
        self.setBackboneFree(freeLayers)
        self.backbonepreproc = weights.transforms(antialias=True)

        self.mod = nn.Sequential(
            nn.Sequential(
                nn.Linear(backboneOutShape, 256),
                nn.LeakyReLU(),
                nn.Dropout(),
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(),
            ),
            nn.Linear(128, 4),
            nn.LeakyReLU(),
        )

    def setBackboneFree(self, freeLayers):
        for name, param in self.backbone.named_parameters():
            if any([name.startswith(fl) for fl in freeLayers]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mod(x)
        return x


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
