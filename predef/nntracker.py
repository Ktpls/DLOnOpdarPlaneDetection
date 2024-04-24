# %%
# basics
from typing import Any
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
        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(localdim, outdim, 1),
        )
        self.global1 = torch.nn.Sequential(
            torch.nn.Conv2d(globaldim, outdim, 1),
            torch.nn.Sigmoid(),
        )
        self.global2 = torch.nn.Sequential(
            torch.nn.Conv2d(globaldim, outdim, 1),
        )
        self.bn = torch.nn.BatchNorm2d(outdim)

    def forward(self, local, globalsemantic):
        x = self.local(local) * self.global1(globalsemantic) + self.global2(
            globalsemantic
        )
        return self.bn(x)


class AddPositionalEmbedding(torch.nn.Module):
    def __init__(self, shape, depth, len_max=None):
        super().__init__()
        self.pe = torch.nn.Parameter(
            torch.tensor(
                PositionalEmbedding2D(shape, depth, len_max),
                dtype=torch.float32,
                requires_grad=False,
            )
        ).requires_grad_(False)

    def forward(self, x):
        return x + self.pe


class PermuteModule(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class GlobalAvePooling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def static_forward(x):
        return torch.mean(x, dim=(2, 3))

    def forward(self, x):
        return GlobalAvePooling.static_forward(x)


class SpatialPositioning(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        h, w = shape
        self.weightX = torch.nn.Parameter(
            torch.linspace(
                0,
                1,
                w,
                dtype=torch.float32,
                requires_grad=False,
            ).reshape([1, 1, 1, -1])
        ).requires_grad_(False)
        self.weightY = torch.nn.Parameter(
            torch.linspace(
                0,
                1,
                h,
                dtype=torch.float32,
                requires_grad=False,
            ).reshape([1, 1, -1, 1])
        ).requires_grad_(False)
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        x = self.softmax(x)
        X = torch.sum(self.weightX * x, dim=[-1, -2])
        Y = torch.sum(self.weightY * x, dim=[-1, -2])
        x = torch.concat([X, Y], dim=-1)
        return x


class ConvBnHs(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        ifBn=True,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = torch.nn.BatchNorm2d(out_channels) if ifBn else None
        self.ifBn = ifBn
        self.hs = torch.nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.hs(x)
        return x


class PartialChannel(torch.nn.Module):
    def __init__(self, idx: typing.Union[int, slice]) -> None:
        super().__init__()
        self.idx = idx
        torch.Tensor.__getitem__

    def forward(self, x):
        return x[:, self.idx, :, :]


class ELAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 2 == 0
        outHalf = out_channels // 2
        self.wayComplex = torch.nn.ModuleList(
            [
                ConvBnHs(in_channels, outHalf, 3),
                torch.nn.Sequential(
                    ConvBnHs(outHalf, outHalf, 3),
                    ConvBnHs(outHalf, outHalf, 3),
                ),
                torch.nn.Sequential(
                    ConvBnHs(outHalf, outHalf, 3),
                    ConvBnHs(outHalf, outHalf, 3),
                ),
            ]
        )
        self.waySimple = ConvBnHs(in_channels, outHalf, 3)
        self.combiner = ConvBnHs(out_channels * 2, out_channels, 1)

    def forward(self, x):
        o_simp = self.waySimple(x)
        o_comp0 = self.wayComplex[0](x)
        o_comp1 = self.wayComplex[1](o_comp0)
        o_comp2 = self.wayComplex[2](o_comp1)
        return self.combiner(
            torch.concat(
                [
                    o_simp,
                    o_comp0,
                    o_comp1,
                    o_comp2,
                ],
                dim=1,
            )
        )


class ELAN_H(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 4 == 0
        outHalf = out_channels // 2
        outQuad = out_channels // 4
        self.wayComplex = torch.nn.ModuleList(
            ConvBnHs(in_channels, outHalf, 3),
            ConvBnHs(outHalf, outQuad, 3),
            ConvBnHs(outQuad, outQuad, 3),
            ConvBnHs(outQuad, outQuad, 3),
            ConvBnHs(outQuad, outQuad, 3),
        )
        self.waySimple = ConvBnHs(in_channels, outHalf, 3)
        self.combiner = ConvBnHs(out_channels * 2, out_channels, 1)

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


class MPn(torch.nn.Module):
    def __init__(self, in_channels, n_value=1, downSamplingStride=2):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % 2 == 0
        out_channels = n_value * in_channels
        cPath = out_channels // 2
        self.out_channels = out_channels
        self.wayPooling = torch.nn.Sequential(
            torch.nn.MaxPool2d(downSamplingStride, downSamplingStride),
            ConvBnHs(in_channels, cPath, 3),
        )
        self.wayConv = torch.nn.Sequential(
            ConvBnHs(
                in_channels,
                cPath,
                downSamplingStride,
                stride=downSamplingStride,
                padding=0,
            ),
            ConvBnHs(cPath, cPath, 3),
        )
        self.combiner = ConvBnHs(cPath * 2, out_channels, 3)

    def forward(self, x):
        o_pool = self.wayPooling(x)
        o_conv = self.wayConv(x)
        return self.combiner(torch.concat([o_pool, o_conv], dim=1))


class FinalModule(torch.nn.Module):
    def parameters(
        self, recurse: bool = True
    ) -> typing.Iterator[torch.nn.parameter.Parameter]:
        return filter(
            lambda x: x.requires_grad is not False, super().parameters(recurse)
        )


class PRNAddition:
    def __init__(self, rho) -> None:
        self.rho = rho

    def __call__(self, before: torch.Tensor, after: torch.Tensor) -> Any:
        chan = int(self.rho * min(before.shape[1], after.shape[1]))
        before = before[:, :chan]

    def staticMethod(self):
        return functools.partial(self.__call__, self=self)


class nntracker_respi(FinalModule):
    # mobile net v3 large
    EXPORT_x16 = 6
    EXPORT_x8 = 12
    EXPORT_x4 = 15
    chanProc16 = 40
    chanProc8 = 112
    chanProc4 = 160
    chanProc4Simplified = 160
    last_channel = 1280 // 2

    def __init__(
        self,
        freeLayers=list(),
        loadPretrainedBackbone=True,
        dropout=0.5,
    ):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(
            weights=weights if loadPretrainedBackbone else None
        )
        self.backbone = self.setBackboneFree(backbone, freeLayers)
        self.backbonepreproc = weights.transforms(antialias=True)
        self.upsampler = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.chan4Simplifier = ConvBnHs(self.chanProc4, self.chanProc4Simplified, 1)

        self.summing4And8 = torch.nn.Sequential(
            ConvBnHs(
                self.chanProc8 + self.chanProc4Simplified,
                self.chanProc8,
                3,
            )
        )

        self.proc16 = torch.nn.Sequential(
            ConvBnHs(
                self.chanProc16 + self.chanProc8,
                self.chanProc16,
                3,
            )
        )

        self.down16to8 = torch.nn.Sequential(
            ConvBnHs(self.chanProc16, self.chanProc16, 3),
            torch.nn.MaxPool2d(2, 2),
            # MPn(chanProc16),
        )

        self.proc8 = torch.nn.Sequential(
            ConvBnHs(
                self.chanProc16 + self.chanProc8,
                self.chanProc8,
                3,
            ),
            ConvBnHs(self.chanProc8, self.chanProc8, 3),
        )

        self.down8to4 = torch.nn.Sequential(
            ConvBnHs(self.chanProc8, self.chanProc8, 3),
            torch.nn.MaxPool2d(2, 2),
            # MPn(chanProc8),
        )

        self.proc4 = torch.nn.Sequential(
            ConvBnHs(
                self.chanProc8 + self.chanProc4Simplified, self.chanProc4Simplified, 3
            ),
            ConvBnHs(self.chanProc4Simplified, self.chanProc4Simplified, 3),
        )

        self.discriminatorFinal = torch.nn.Sequential(
            torch.nn.Linear(
                self.chanProc4Simplified + self.chanProc8 + self.chanProc16,
                self.last_channel,
            ),
            torch.nn.Hardswish(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.last_channel, 4),
        )

    def setBackboneFree(self, backbone: torch.nn.Module, freeLayers):
        for name, param in backbone.named_parameters():
            if any([name.startswith(fl) for fl in freeLayers]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return backbone

    def fpnForward(self, x):
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            if i == self.EXPORT_x16:
                out16 = x
            elif i == self.EXPORT_x8:
                out8 = x
            elif i == self.EXPORT_x4:
                out4 = x
                break
        out4 = self.chan4Simplifier(out4)
        return out16, out8, out4

    def neckForward(self, out16, out8, out4):
        summed = self.summing4And8(torch.concat([out8, self.upsampler(out4)], dim=1))
        sum16 = self.proc16(torch.concat([out16, self.upsampler(summed)], dim=1))
        sum8 = self.proc8(torch.concat([summed, self.down16to8(sum16)], dim=1))
        sum4 = self.proc4(torch.concat([out4, self.down8to4(sum8)], dim=1))
        return sum16, sum8, sum4

    def headForward(self, sum16, sum8, sum4):
        pathes = [sum4, sum8, sum16]
        x = torch.concat(
            [torch.flatten(self.backbone.avgpool(o), 1) for o in pathes],
            dim=1,
        )
        x = self.discriminatorFinal(x)
        return x

    def forward(self, x):
        out16, out8, out4 = self.fpnForward(x)
        sum16, sum8, sum4 = self.neckForward(out16, out8, out4)
        x = self.headForward(sum16, sum8, sum4)
        return x


class nntracker_respi_MPn(nntracker_respi):
    def __init__(self, freeLayers=list(), loadPretrainedBackbone=True, dropout=0.5):
        super().__init__(freeLayers, loadPretrainedBackbone, dropout)

        self.down16to8 = torch.nn.Sequential(
            MPn(self.chanProc16),
        )

        self.down8to4 = torch.nn.Sequential(
            MPn(self.chanProc8),
        )


class nntracker_respi_spatialpositioning_head(nntracker_respi):
    last_channel = nntracker_respi.chanProc4Simplified * 2

    def __init__(self, freeLayers=list(), loadPretrainedBackbone=True, dropout=0.5):
        super().__init__(freeLayers, loadPretrainedBackbone, dropout)
        self.locator16 = SpatialPositioning([16, 16])
        self.locator8 = SpatialPositioning([8, 8])
        self.locator4 = SpatialPositioning([4, 4])

        self.discriminatorFinal = torch.nn.Sequential(
            torch.nn.Linear(
                2 * (self.chanProc4Simplified + self.chanProc8 + self.chanProc16),
                self.last_channel,
            ),
            torch.nn.Dropout(dropout),
            torch.nn.Hardswish(),
            res_through(
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.last_channel,
                        self.last_channel,
                    ),
                    torch.nn.Dropout(dropout),
                    torch.nn.Hardswish(),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.last_channel,
                        self.last_channel,
                    ),
                    torch.nn.Dropout(dropout),
                    torch.nn.Hardswish(),
                ),
            ),
            torch.nn.Linear(self.last_channel, 4),
        )

    def headForward(self, sum16, sum8, sum4):
        x = torch.concat(
            [
                self.locator4(sum4),
                self.locator8(sum8),
                self.locator16(sum16),
            ],
            dim=1,
        )
        x = self.discriminatorFinal(x)
        return x


class nntracker_respi_mnv3s(nntracker_respi):
    # mobile net v3 small
    EXPORT_x16 = 3
    EXPORT_x8 = 8
    EXPORT_x4 = 11
    chanProc16 = 24
    chanProc8 = 48
    chanProc4 = 96
    chanProc4Simplified = 160
    last_channel = 1024 // 2

    def __init__(self, freeLayers=list(), loadPretrainedBackbone=True, dropout=0.2):
        super().__init__(freeLayers, loadPretrainedBackbone, dropout)
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_small(
            weights=weights if loadPretrainedBackbone else None
        )
        self.backbone = self.setBackboneFree(backbone, freeLayers)
        self.backbonepreproc = weights.transforms(antialias=True)
