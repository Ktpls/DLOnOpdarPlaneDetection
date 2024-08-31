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
            ).reshape([1, 1, 1, -1]),
            requires_grad=False,
        )
        self.weightY = torch.nn.Parameter(
            torch.linspace(
                0,
                1,
                h,
                dtype=torch.float32,
                requires_grad=False,
            ).reshape([1, 1, -1, 1]),
            requires_grad=False,
        )
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        xsm = self.softmax(x)
        M = torch.sum(x * xsm, dim=[-1, -2])
        X = torch.sum(self.weightX * xsm, dim=[-1, -2])
        Y = torch.sum(self.weightY * xsm, dim=[-1, -2])
        x = torch.concat([X, Y, M], dim=-1)
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
                ConvGnHs(in_channels, outHalf, 3),
                torch.nn.Sequential(
                    ConvGnHs(outHalf, outHalf, 3),
                    ConvGnHs(outHalf, outHalf, 3),
                ),
                torch.nn.Sequential(
                    ConvGnHs(outHalf, outHalf, 3),
                    ConvGnHs(outHalf, outHalf, 3),
                ),
            ]
        )
        self.waySimple = ConvGnHs(in_channels, outHalf, 3)
        self.combiner = ConvGnHs(out_channels * 2, out_channels, 1)

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
            ConvGnHs(in_channels, outHalf, 3),
            ConvGnHs(outHalf, outQuad, 3),
            ConvGnHs(outQuad, outQuad, 3),
            ConvGnHs(outQuad, outQuad, 3),
            ConvGnHs(outQuad, outQuad, 3),
        )
        self.waySimple = ConvGnHs(in_channels, outHalf, 3)
        self.combiner = ConvGnHs(out_channels * 2, out_channels, 1)

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
            ConvGnHs(in_channels, cPath),
        )
        self.wayConv = torch.nn.Sequential(
            ConvGnHs(in_channels, cPath, kernel_size=1),
            ConvGnHs(cPath, cPath, stride=downSamplingStride, padding=1),
        )
        self.combiner = ConvGnHs(cPath * 2, out_channels)

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


class Neck(torch.nn.Module):
    def __init__(
        self,
        conv4,
        conv8,
        conv16,
        down16to8,
        conv8_2,
        down8to4,
        conv4_2,
    ):
        super().__init__()
        self.conv4 = conv4
        self.conv8 = conv8
        self.conv16 = conv16
        self.down16to8 = down16to8
        self.conv8_2 = conv8_2
        self.down8to4 = down8to4
        self.conv4_2 = conv4_2


class nntracker_respi(FinalModule):
    # mobile net v3 large
    chan16 = 40
    chan8 = 112
    chan4c = 160
    chan4 = 160
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
        self.chan4Simplifier = ConvGnHs(self.chan4c, self.chan4, kernel_size=1)
        self.backbonepreproc = weights.transforms(antialias=True)

        self.upsampler = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.concater = ModuleFunc(lambda x: torch.concat(x, dim=-3))

        self.neck = Neck(
            conv4=ConvGnHs(self.chan4, self.chan4),
            conv8=torch.nn.Sequential(
                ConvGnHs(self.chan8 + self.chan4, self.chan8),
                ConvGnHs(self.chan8, self.chan8),
            ),
            conv16=torch.nn.Sequential(
                ConvGnHs(self.chan16 + self.chan8, self.chan16),
                ConvGnHs(self.chan16, self.chan16),
            ),
            down16to8=torch.nn.Sequential(
                ConvGnHs(self.chan16, self.chan16),
                torch.nn.MaxPool2d(2, 2),
            ),
            conv8_2=ConvGnHs(self.chan16 + self.chan8, self.chan8),
            down8to4=torch.nn.Sequential(
                ConvGnHs(self.chan8, self.chan8),
                torch.nn.MaxPool2d(2, 2),
            ),
            conv4_2=ConvGnHs(self.chan8 + self.chan4, self.chan4),
        )

        self.discriminatorFinal = torch.nn.Sequential(
            torch.nn.Linear(
                self.chan4 + self.chan8 + self.chan16,
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
        export16 = 6
        export8 = 12
        export4 = 15
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            if i == export16:
                out16 = x
            elif i == export8:
                out8 = x
            elif i == export4:
                out4 = x
                break
        out4 = self.chan4Simplifier(out4)
        return out16, out8, out4

    def neckForward(self, o16, o8, o4):
        x = o4
        x = self.neck.conv4(x)
        m4 = x
        x = self.upsampler(x)
        x = self.concater([x, o8])
        x = self.neck.conv8(x)
        m8 = x
        x = self.upsampler(x)
        x = self.concater([x, o16])
        x = self.neck.conv16(x)
        o16 = x
        x = self.neck.down16to8(x)
        x = self.concater([x, m8])
        x = self.neck.conv8_2(x)
        o8 = x
        x = self.neck.down8to4(x)
        x = self.concater([x, m4])
        x = self.neck.conv4_2(x)
        o4 = x
        return o16, o8, o4

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
        figDemo = plt.figure(figsize=(20, 20))
        plotShape = [7, 8]
        mpp = MassivePicturePlot(plotShape, fig=figDemo)
        samplenum = np.prod(plotShape)
        imshowconfig = {"vmin": 0, "vmax": 1}
        ps = perf_statistic()
        prog = Progress(samplenum)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss = list()
        with torch.no_grad():
            for i in range(samplenum):
                for datatuple in dataloader:
                    break
                src, lbl, pi = datatuple
                ps.start()
                pihat = self.inferenceProgress(src)
                ps.stop().countcycle()
                loss.append(self.calcloss(pi, pihat).item())
                pi = pi.squeeze(0).cpu().numpy()
                pihat = pihat.squeeze(0).cpu().numpy()
                src = src.squeeze(0)
                lbl = lbl.squeeze(0)
                src, lbl = [tensorimg2ndarray(d) for d in [src, lbl]]

                # mpp.toNextPlot()
                # plt.title(PI2Str(pi))
                # plt.imshow(src)

                mpp.toNextPlot()
                lblRebuild = planeInfo2Lbl(pihat, stdShape)
                zeroLikeLbl = np.zeros_like(lblRebuild)
                mixture = np.where(
                    lblRebuild > 0.5,
                    0.7 * src
                    + 0.3
                    * np.concatenate(
                        [
                            lblRebuild,
                            zeroLikeLbl,
                            zeroLikeLbl,
                        ],
                        axis=-1,
                    ),
                    src,
                )

                plt.title(PI2Str(pihat))
                plt.imshow(mixture, label="lblComparasion", **imshowconfig)
                prog.update(i)
            prog.setFinish()

        loss = np.array(loss)
        loss = np.sort(loss)
        figStatistics = plt.figure()
        plt.hist(loss, bins=30, color="blue", edgecolor="black")
        plt.xlabel("Loss Value")
        plt.ylabel("Frequency")
        plt.title("Loss Distribution")
        print(f"aveLoss={np.mean(loss)}")
        print(f"stdErr={np.std(loss)}")
        print(f"95%quantile={loss[int(len(loss) * 0.95 + 0.5)]}")

    def calcloss(self, pi: torch.Tensor, pihat: torch.Tensor):
        device = pihat.device
        batch, dimPi = pi.shape
        isObj = pi[:, 0].unsqueeze(1)
        isObjHat = pihat[:, 0].unsqueeze(1)
        boolIsObj = isObj > 0.5

        coef = torch.tensor(
            [1, 1, 2], dtype=torch.float32, device=device, requires_grad=False
        ).unsqueeze(0)
        detailMask = boolIsObj.to(dtype=torch.float32)
        detailMse = (pihat - pi)[:, 1:] ** 2
        detailLoss = detailMask * coef * detailMse
        detailLoss = torch.sum(detailLoss, dim=-1, keepdim=True)

        # alpha = 1 - 52 / 64  # from experiment
        # alphat = torch.where(boolIsObj, alpha, 1 - alpha)
        alphat = 1
        pt = torch.where(boolIsObj, isObjHat, 1 - isObjHat)
        gamma = 2
        classLoss = -alphat * (1 - pt) ** gamma * torch.log(pt)

        loss = torch.mean(1 * detailLoss + classLoss)
        return loss


class nntracker_respi_MPn(nntracker_respi):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.neck.down16to8 = MPn(self.chan16)
        self.neck.down8to4 = MPn(self.chan8)


class nntracker_respi_spatialpositioning_head(nntracker_respi_MPn):
    last_channel = nntracker_respi.chan4

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.locator16 = SpatialPositioning([16, 16])
        self.locator8 = SpatialPositioning([8, 8])
        self.locator4 = SpatialPositioning([4, 4])

        self.discriminatorFinal = torch.nn.Sequential(
            torch.nn.Linear(
                3 * (self.chan4 + self.chan8 + self.chan16),  # 2 for [x, y]
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


class nntracker_respi_mnv3s(nntracker_respi_spatialpositioning_head):
    # mobile net v3 small
    chan16 = 24
    chan8 = 48
    chan4 = 96
    chan4c = 96

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.backbonepreproc = weights.transforms(antialias=True)
        self.chan4Simplifier = None

    def fpnForward(self, x):
        export16 = 3
        export8 = 8
        export4 = 11
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            if i == export16:
                out16 = x
            elif i == export8:
                out8 = x
            elif i == export4:
                out4 = x
                break
        return out16, out8, out4


class nntracker_respi_resnet(nntracker_respi_spatialpositioning_head):
    chan16 = 128
    chan8 = 256
    chan4c = 512
    chan4 = 256
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
        export16 = 5
        export8 = 6
        export4 = 7
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
            if i == export16:
                out16 = x
            elif i == export8:
                out8 = x
            elif i == export4:
                out4 = x
                break
        out4 = self.chan4Simplifier(out4)
        return out16, out8, out4
