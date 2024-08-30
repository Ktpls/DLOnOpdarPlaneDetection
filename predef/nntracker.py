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
        self.out_channels = out_channels
        cPath = out_channels // 2
        self.wayPooling = torch.nn.Sequential(
            torch.nn.MaxPool2d(downSamplingStride, downSamplingStride),
            ConvBnHs(in_channels, cPath, 3),
        )
        self.wayConv = torch.nn.Sequential(
            ConvBnHs(in_channels, cPath, 1),
            ConvBnHs(
                cPath,
                cPath,
                kernel_size=3,
                stride=downSamplingStride,
                padding=1,
            ),
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

    def calcloss(self, *arg, **kw): ...

    def trainprogress(self, datatuple): ...

    def inferenceProgress(self, datatuple):
        pass

    def save(self, path):
        savemodel(self, path)


class PRNAddition:
    def __init__(self, rho) -> None:
        self.rho = rho

    def __call__(self, before: torch.Tensor, after: torch.Tensor) -> Any:
        chan = int(self.rho * min(before.shape[1], after.shape[1]))
        before = before[:, :chan]

    def staticMethod(self):
        return functools.partial(self.__call__, self=self)


class InspFuncMixture(torch.nn.Module):
    def __init__(self, methodByChannel):
        super().__init__()
        self.methodByChannel = methodByChannel

    def forward(self, x):
        result = [self.methodByChannel[c](x[:, c : c + 1]) for c in range(x.shape[1])]
        result = torch.concat(result, dim=1)
        return result


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
        dropoutp=0.5,
    ):
        super().__init__()
        self.dropoutp = dropoutp
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(weights=weights)
        self.backbone = backbone
        self.backbonepreproc = weights.transforms(antialias=True)
        self.upsampler = torch.nn.Upsample(scale_factor=2, mode="bilinear")

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
            torch.nn.Dropout(dropoutp),
            torch.nn.Linear(self.last_channel, 4),
            InspFuncMixture(
                [
                    torch.sigmoid,
                    torch.nn.functional.hardswish,
                    torch.nn.functional.hardswish,
                    torch.nn.functional.hardswish,
                ]
            ),
        )

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

    def trainprogress(self, datatuple):
        self.train()
        src, lbl, pi = datatuple
        src = src.to(device)
        pi = pi.to(device)
        pihat = self.forward(src)
        loss = self.calcloss(pi, pihat)
        return loss

    def inferenceProgress(self, src):
        return self.forward(src)

    def demo(self, dataset: labeldataset):
        mpp = MassivePicturePlot([7, 8])
        samplenum = np.prod([7, 4])
        imshowconfig = {"vmin": 0, "vmax": 1}
        ps = perf_statistic()
        prog = Progress(samplenum)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for i in range(samplenum):
                for datatuple in dataloader:
                    break
                src, lbl, pi = datatuple
                ps.start()
                pihat = self.inferenceProgress(src)
                ps.stop().countcycle()
                pi = pi.squeeze(0).cpu().numpy()
                pihat = pihat.squeeze(0).cpu().numpy()
                src = src.squeeze(0)
                lbl = lbl.squeeze(0)
                src, lbl = [tensorimg2ndarray(d) for d in [src, lbl]]

                mpp.toNextPlot()
                plt.title(PI2Str(pi))
                plt.imshow(src)

                mpp.toNextPlot()
                lblComparasion = (
                    np.array(
                        [
                            lbl,
                            np.zeros_like(lbl),
                            planeInfo2Lbl(pihat, stdShape),
                        ]
                    )
                    .squeeze(-1)
                    .transpose([1, 2, 0])
                )

                plt.title(PI2Str(pihat))
                plt.imshow(lblComparasion, label="lblComparasion", **imshowconfig)
                prog.update(i)
            prog.setFinish()

    def calcloss(self, pi: torch.Tensor, pihat: torch.Tensor):
        device = pihat.device
        batch, dimPi = pi.shape
        isObj = pi[:, 0].unsqueeze(1)
        isObjHat = pihat[:, 0].unsqueeze(1)
        boolIsObj = isObj > 0.5

        coef = torch.tensor(
            [1, 1, 3], dtype=torch.float32, device=device, requires_grad=False
        ).unsqueeze(0)
        detailMask = boolIsObj.to(dtype=torch.float32)
        detailMse = (pihat - pi)[:, 1:] ** 2
        detailLoss = detailMask * coef * detailMse
        detailLoss = torch.sum(detailLoss, dim=-1, keepdim=True)

        # alpha = 1 - 52 / 64  # from experiment
        # alphat = torch.where(boolIsObj, alpha, 1 - alpha)
        alphat = 1
        pt = torch.where(boolIsObj, isObjHat, 1 - isObjHat)
        gamma = 1
        classLoss = -alphat * (1 - pt) ** gamma * torch.log(pt)

        loss = torch.mean(5 * detailLoss + classLoss)
        return loss


class nntracker_respi_MPn(nntracker_respi):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.down16to8 = torch.nn.Sequential(
            MPn(self.chanProc16),
        )

        self.down8to4 = torch.nn.Sequential(
            MPn(self.chanProc8),
        )


class nntracker_respi_ELAN(nntracker_respi):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.summing4And8 = torch.nn.Sequential(
            ELAN(self.chanProc8 + self.chanProc4Simplified, self.chanProc8)
        )
        self.proc16 = torch.nn.Sequential(
            ELAN(self.chanProc16 + self.chanProc8, self.chanProc16)
        )
        self.proc8 = torch.nn.Sequential(
            ELAN(self.chanProc16 + self.chanProc8, self.chanProc8),
            ELAN(self.chanProc8, self.chanProc8),
        )
        self.proc4 = torch.nn.Sequential(
            ELAN(self.chanProc8 + self.chanProc4Simplified, self.chanProc4Simplified),
            ELAN(self.chanProc4Simplified, self.chanProc4Simplified),
        )


class nntracker_respi_spatialpositioning_head(nntracker_respi):
    last_channel = nntracker_respi.chanProc4Simplified

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.locator16 = SpatialPositioning([16, 16])
        self.locator8 = SpatialPositioning([8, 8])
        self.locator4 = SpatialPositioning([4, 4])

        self.discriminatorFinal = torch.nn.Sequential(
            torch.nn.Linear(
                2 * (self.chanProc4Simplified + self.chanProc8 + self.chanProc16),
                self.last_channel,
            ),
            torch.nn.Dropout(self.dropoutp),
            torch.nn.Hardswish(),
            # res_through(
            #     torch.nn.Sequential(
            #         torch.nn.Linear(
            #             self.last_channel,
            #             self.last_channel,
            #         ),
            #         torch.nn.Dropout(self.dropoutp),
            #         torch.nn.Hardswish(),
            #     ),
            #     torch.nn.Sequential(
            #         torch.nn.Linear(
            #             self.last_channel,
            #             self.last_channel,
            #         ),
            #         torch.nn.Dropout(self.dropoutp),
            #         torch.nn.Hardswish(),
            #     ),
            # ),
            torch.nn.Linear(self.last_channel, 4),
            InspFuncMixture(
                [
                    torch.sigmoid,
                    torch.nn.functional.hardswish,
                    torch.nn.functional.hardswish,
                    torch.nn.functional.hardswish,
                ]
            ),
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

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.backbonepreproc = weights.transforms(antialias=True)


class nntracker_respi_resnet(nntracker_respi_MPn):
    EXPORT_x16 = 5
    EXPORT_x8 = 6
    EXPORT_x4 = 7
    chanProc16 = 128
    chanProc8 = 256
    chanProc4 = 512
    chanProc4Simplified = 256
    last_channel = 512

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbonepreproc = weights.transforms(antialias=True)

    def fpnForward(self, x):
        """
        0 torch.Size([2, 64, 64, 64])
        1 torch.Size([2, 64, 64, 64])
        2 torch.Size([2, 64, 64, 64])
        3 torch.Size([2, 64, 32, 32])
        4 torch.Size([2, 64, 32, 32])
        5 torch.Size([2, 128, 16, 16])
        6 torch.Size([2, 256, 8, 8])
        7 torch.Size([2, 512, 4, 4])
        """
        for i, m in enumerate(
            [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
        ):
            x = m(x)
            # print(i, x.shape)
            if i == self.EXPORT_x16:
                out16 = x
            elif i == self.EXPORT_x8:
                out8 = x
            elif i == self.EXPORT_x4:
                out4 = x
                break
        out4 = self.chan4Simplifier(out4)
        return out16, out8, out4
