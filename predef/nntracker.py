# %%
# basics
from .nntrackerdev_predef import *


# %%
# nn def


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


class SemanticInjection(torch.nn.Module):
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


class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 2 == 0
        self.outHalf = out_channels // 2
        self.wayComplex = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
                nn.Conv2d(out_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
                nn.Conv2d(out_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            ),
        )
        self.waySimple = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            ),
        )
        self.combiner = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x):
        o_simp = self.waySimple(x)
        o_comp0 = self.wayComplex[0](x)
        o_comp1 = self.wayComplex[1](o_comp0)
        o_comp2 = self.wayComplex[2](o_comp1)
        return self.combiner(
            torch.concat(
                [
                    o_simp[: self.outHalf],
                    o_comp0[: self.outHalf],
                    o_comp1[: self.outHalf],
                    o_comp2[: self.outHalf],
                ],
                dim=1,
            )
        )


class ELAN_H(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 4 == 0
        self.outHalf = out_channels // 2
        self.outQuad = out_channels // 4
        self.wayComplex = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, self.outHalf, 3, padding="same"),
                nn.BatchNorm2d(self.outHalf),
                nn.Hardswish(),
            ),
            nn.Sequential(
                nn.Conv2d(self.outHalf, self.outQuad, 3, padding="same"),
                nn.BatchNorm2d(self.outQuad),
                nn.Hardswish(),
            ),
            nn.Sequential(
                nn.Conv2d(self.outQuad, self.outQuad, 3, padding="same"),
                nn.BatchNorm2d(self.outQuad),
                nn.Hardswish(),
            ),
            nn.Sequential(
                nn.Conv2d(self.outQuad, self.outQuad, 3, padding="same"),
                nn.BatchNorm2d(self.outQuad),
                nn.Hardswish(),
            ),
            nn.Sequential(
                nn.Conv2d(self.outQuad, self.outQuad, 3, padding="same"),
                nn.BatchNorm2d(self.outQuad),
                nn.Hardswish(),
            ),
        )
        self.waySimple = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            ),
        )
        self.combiner = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x):
        o_simp = self.waySimple(x)
        o_comp0 = self.wayComplex[0](x)
        o_comp1 = self.wayComplex[1](o_comp0)
        o_comp2 = self.wayComplex[2](o_comp1)
        o_comp3 = self.wayComplex[3](o_comp2)
        o_comp4 = self.wayComplex[4](o_comp3)
        return self.combiner(
            torch.concat(
                [
                    o_simp,
                    o_comp0,
                    o_comp1,
                    o_comp2,
                    o_comp3,
                    o_comp4,
                ],
                dim=1,
            )
        )


class MPn(nn.Module):
    def __init__(self, in_channels, n_value, downSamplingStride=2):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % 2 == 0
        cHalf = in_channels // 2
        self.cHalf = cHalf
        out_channels = n_value * in_channels
        self.out_channels = out_channels
        self.wayPooling = nn.Sequential(
            nn.MaxPool2d(downSamplingStride, downSamplingStride),
            nn.Conv2d(cHalf, cHalf, 3, padding="same"),
            nn.BatchNorm2d(cHalf),
            nn.Hardswish(),
        )
        self.wayConv = nn.Sequential(
            nn.Conv2d(
                cHalf, cHalf, downSamplingStride, stride=downSamplingStride, padding=0
            ),
            nn.BatchNorm2d(cHalf),
            nn.Hardswish(),
            nn.Conv2d(cHalf, cHalf, 3, padding="same"),
            nn.BatchNorm2d(cHalf),
            nn.Hardswish(),
        )
        self.combiner = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x):
        o_pool = self.wayPooling(x[:, : self.cHalf])
        o_conv = self.wayConv(x[:, self.cHalf :])
        return self.combiner(torch.concat([o_pool, o_conv], dim=1))


class ParameterRequiringGradModule(torch.nn.Module):
    def parameters(
        self, recurse: bool = True
    ) -> typing.Iterator[torch.nn.parameter.Parameter]:
        return filter(
            lambda x: x.requires_grad is not False, super().parameters(recurse)
        )


class FinalModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parameters(
        self, recurse: bool = True
    ) -> typing.Iterator[torch.nn.parameter.Parameter]:
        return filter(
            lambda x: x.requires_grad is not False, super().parameters(recurse)
        )


class nntracker_respi(ParameterRequiringGradModule):
    def __init__(
        self,
        freeLayers=list(),
        loadPretrainedBackbone=True,
        dropout=0.2,
    ):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(
            weights=weights if loadPretrainedBackbone else None
        )
        self.backbone = backbone
        self.setBackboneFree(freeLayers)
        self.backbonepreproc = weights.transforms(antialias=True)
        chanProc16 = 40
        chanProc8 = 112
        chanProc4 = 960
        chanProc4Simplified = 160
        self.upsampler = nn.Upsample(scale_factor=2, mode="bilinear")

        # self.chan4Simplifier = nn.Sequential(
        #     nn.Conv2d(chanProc4, chanProc4Simplified, 1, stride=1, padding="same"),
        #     nn.BatchNorm2d(chanProc4Simplified),
        #     nn.Hardswish(),
        # )

        self.summing4And8 = nn.Sequential(
            nn.Conv2d(
                chanProc8 + chanProc4Simplified, chanProc8, 3, stride=1, padding="same"
            ),
            nn.BatchNorm2d(chanProc8),
            nn.Hardswish(),
        )

        self.proc16 = nn.Sequential(
            nn.Conv2d(chanProc16 + chanProc8, chanProc16, 3, stride=1, padding="same"),
            nn.BatchNorm2d(chanProc16),
            nn.Hardswish(),
        )

        self.down16to8 = nn.Sequential(
            nn.Conv2d(chanProc16, chanProc16, 3, stride=1, padding="same"),
            nn.BatchNorm2d(chanProc16),
            nn.Hardswish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.proc8 = nn.Sequential(
            nn.Conv2d(chanProc16 + chanProc8, chanProc8, 3, stride=1, padding="same"),
            nn.BatchNorm2d(chanProc8),
            nn.Hardswish(),
            nn.Conv2d(chanProc8, chanProc8, 3, stride=1, padding="same"),
            nn.BatchNorm2d(chanProc8),
            nn.Hardswish(),
        )

        self.down8to4 = nn.Sequential(
            nn.Conv2d(chanProc8, chanProc8, 3, stride=1, padding="same"),
            nn.BatchNorm2d(chanProc8),
            nn.Hardswish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.proc4 = nn.Sequential(
            nn.Conv2d(
                chanProc8 + chanProc4Simplified,
                chanProc4Simplified,
                3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(chanProc4Simplified),
            nn.Hardswish(),
            nn.Conv2d(
                chanProc4Simplified, chanProc4Simplified, 3, stride=1, padding="same"
            ),
            nn.BatchNorm2d(chanProc4Simplified),
            nn.Hardswish(),
        )

        last_channel = 1280 // 2

        self.discriminatorFinal = nn.Sequential(
            nn.Linear(chanProc4Simplified + chanProc8 + chanProc16, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(last_channel, 4),
        )

    def setBackboneFree(self, freeLayers):
        for name, param in self.backbone.named_parameters():
            if any([name.startswith(fl) for fl in freeLayers]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            if i == 6:
                out16 = x
            elif i == 12:
                out8 = x
            elif i == 15:
                out4 = x
                break
        # out4 = self.chan4Simplifier(out4)
        summed = self.summing4And8(torch.concat([out8, self.upsampler(out4)], dim=1))
        sum16 = self.proc16(torch.concat([out16, self.upsampler(summed)], dim=1))
        sum8 = self.proc8(torch.concat([summed, self.down16to8(sum16)], dim=1))
        sum4 = self.proc4(torch.concat([out4, self.down8to4(sum8)], dim=1))

        pathes = [sum4, sum8, sum16]
        x = torch.concat(
            [torch.flatten(self.backbone.avgpool(o), 1) for o in pathes],
            dim=1,
        )
        x = self.discriminatorFinal(x)
        return x


def getmodel(model0: torch.nn.Module, modelpath, device):
    model = setModule(model0, path=modelpath, device=device)
    paramNum = np.sum(
        [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    )
    print(f"{paramNum=}")
    # print(model)
    return model


def getDevice():
    print(getDeviceInfo())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device
