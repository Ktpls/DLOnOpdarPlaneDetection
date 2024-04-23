from utilitypack.utility import *
from utilitypack.util_torch import *
import torchvision
from torch import nn
# (?<!torch\.)(?=nn\.) => torch.
from torch.utils.data import DataLoader
stdShape = [128, 128]
import importlib
import sys


def import_or_reload(module_name):
    module = sys.modules.get(module_name)
    try:
        if module is None:
            module = importlib.import_module(module_name)
        else:
            module = importlib.reload(module)
    except ImportError:
        module = None

    return module


class ImgReader:
    def read(self, path): ...


class ImgReaderFolder(ImgReader):
    def __init__(self, folder):
        self.folder = folder

    def read(self, path):
        m = os.path.join(self.folder, path)
        m = cv.imread(m, 1)
        m = m.astype(np.float32) / 255
        return m


class ImgReaderZip(ImgReader):
    def __init__(self, zipf):
        from zipfile import ZipFile

        self.zipf = ZipFile(zipf)

    def read(self, path):
        m = self.zipf.read(path)
        m = np.frombuffer(m, dtype=np.uint8)
        m = cv.imdecode(m, 1)
        m = m.astype(np.float32) / 255
        return m


def fit_errmax(P):
    ave = P.sum(0) / P.shape[0]
    ave = np.repeat(ave.reshape([1, 2]), P.shape[0], axis=0)
    Pcenterized = P - ave
    # layout: X==P[:,0], Y==P[:,1]
    delta = (Pcenterized[:, 0] ** 2 - Pcenterized[:, 1] ** 2).sum()
    gamma = (Pcenterized[:, 0] * Pcenterized[:, 1]).sum()
    base = np.sqrt(delta**2 + 4 * gamma**2)
    if base < 0.1:
        base = 0.1
    cosphi = delta / base
    sinphi = 2 * gamma / base
    cosita = -cosphi
    sinita = -sinphi
    Apsi = np.sqrt((1 - cosita) / 2)
    Bpsi = np.sqrt((cosita + 1) / 2)
    Bpsi = Bpsi if sinita > 0 else -Bpsi
    return -Apsi, Bpsi, Pcenterized


def mat2pointset(m):
    idx = np.array(np.where(m > 0))
    X = idx[0].reshape([idx.shape[1], 1])
    Y = idx[1].reshape([idx.shape[1], 1])
    P = np.concatenate((X, Y), axis=1)
    return P


def estimateWingSpan(m):
    ps = mat2pointset(m)
    # at least 2 points
    if len(ps) < 2:
        return 0
    A, B, Pc = fit_errmax(ps)
    dist2 = (A * Pc[:, 0] + B * Pc[:, 1]) ** 2
    dist2max = dist2.max()
    return 2 * np.sqrt(dist2max)


def lbl2PlaneInfo(lbl: np.ndarray):
    h, w = lbl.shape
    lblsurface = lbl.sum(axis=(-1, -2), keepdims=True)
    if lblsurface > 10:
        X = np.arange(w).reshape(1, 1, w)
        Y = np.arange(h).reshape(1, h, 1)
        meanX = (lbl * X).sum(axis=(-1, -2), keepdims=True) / lblsurface
        meanY = (lbl * Y).sum(axis=(-1, -2), keepdims=True) / lblsurface

        dist = np.sqrt((X - meanX) ** 2 + (Y - meanY) ** 2) * lbl

        wingSpan = 2 * np.max(dist) / w
        return (
            1,
            meanX[0, 0, 0] / w,
            meanY[0, 0, 0] / h,
            wingSpan,
        )
    else:
        return (0, 0, 0, 0)


def planeInfo2Lbl(tup, lblShape):
    h, w = lblShape
    isObj, meanX, meanY, wingSpan = tup
    meanX = meanX * w
    meanY = meanY * h
    wingSpan = wingSpan * w
    lbl = np.zeros(lblShape + [1], dtype=np.float32)
    X = np.arange(w).reshape(1, w, 1)
    Y = np.arange(h).reshape(h, 1, 1)
    dist = np.sqrt((X - meanX) ** 2 + (Y - meanY) ** 2)
    lbl[dist < wingSpan / 2] = 1
    return lbl


@dataclasses.dataclass
class SampleItem:
    name: str
    spl: np.ndarray
    lbl: np.ndarray
    pi: tuple


import copy


class NonAffineTorchAutoAugment(torchvision.transforms.AutoAugment):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = (img * 255).to(dtype=torch.uint8)
        img = super().forward(img)
        img = img.to(dtype=torch.float32) / 255
        return img

    def _get_policies(
        self, policy: torchvision.transforms.autoaugment.AutoAugmentPolicy
    ):
        policies = [
            [("AutoContrast", 0.5, None), ("Equalize", 0.9, None)],
            [("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)],
            [("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)],
            [("Brightness", 0.1, 3), ("Color", 0.7, 0)],
            [("Brightness", 0.9, 6), ("Color", 0.2, 8)],
            [("Color", 0.4, 0), ("Equalize", 0.6, None)],
            [("Color", 0.4, 3), ("Brightness", 0.6, 7)],
            [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
            [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
            [("Color", 0.9, 9), ("Equalize", 0.6, None)],
            [("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)],
            [("Equalize", 0.0, None), ("Equalize", 0.8, None)],
            [("Equalize", 0.2, None), ("AutoContrast", 0.6, None)],
            [("Equalize", 0.2, None), ("Equalize", 0.6, None)],
            [("Equalize", 0.3, None), ("AutoContrast", 0.4, None)],
            [("Equalize", 0.4, None), ("Solarize", 0.2, 4)],
            [("Equalize", 0.6, None), ("Equalize", 0.5, None)],
            [("Equalize", 0.6, None), ("Posterize", 0.4, 6)],
            [("Equalize", 0.6, None), ("Solarize", 0.6, 6)],
            [("Equalize", 0.8, None), ("Equalize", 0.6, None)],
            [("Equalize", 0.8, None), ("Invert", 0.1, None)],
            [("Invert", 0.1, None), ("Contrast", 0.2, 6)],
            [("Invert", 0.6, None), ("Equalize", 1.0, None)],
            [("Invert", 0.9, None), ("AutoContrast", 0.8, None)],
            [("Invert", 0.9, None), ("Equalize", 0.6, None)],
            [("Posterize", 0.6, 7), ("Posterize", 0.6, 6)],
            [("Posterize", 0.8, 5), ("Equalize", 1.0, None)],
            [("Sharpness", 0.3, 9), ("Brightness", 0.7, 6)],
            [("Sharpness", 0.4, 7), ("Invert", 0.6, None)],
            [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
            [("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)],
            [("Solarize", 0.5, 2), ("Invert", 0.0, None)],
            [("Solarize", 0.6, 3), ("Equalize", 0.6, None)],
            [("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)],
        ]

        # # AutoAugmentPolicy.IMAGENET
        # policies = [
        #     (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
        #     (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        #     (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        #     (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        #     (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        #     (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
        #     (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
        #     (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
        #     (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
        #     (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        #     (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
        #     (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
        #     (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
        #     (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        #     (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        #     (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        #     (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
        #     (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
        #     (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
        #     (("Color", 0.4, 0), ("Equalize", 0.6, None)),
        #     (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        #     (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        #     (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        #     (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        #     (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        # ]
        policies = [
            [
                s
                for s in p
                if s[0]
                not in ["TranslateX", "TranslateY", "Rotate", "ShearX", "ShearY"]
            ]
            for p in policies
        ]
        policies = [p for p in policies if len(p) >= 2]
        return policies


class AffineMats:
    zoom = lambda rate: np.array(
        [[rate, 0, 0], [0, rate, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    shift = lambda x, y: np.array(
        [[1, 0, x], [0, 1, y], [0, 0, 1]],
        dtype=np.float32,
    )
    flip = lambda lr, ud: np.array(
        [[lr, 0, 0], [0, ud, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    rot = lambda the: np.array(
        [[np.cos(the), np.sin(the), 0], [-np.sin(the), np.cos(the), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    identity = lambda: np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )


def safeAffineAug(spl, lbl):
    lbl = NormalizeImgToChanneled_CvFormat(lbl)
    lblSurface = np.sum(lbl)
    h, w, c = spl.shape
    wh = np.array([w, h])
    rounds = 0
    while True:
        if rounds >= 20:
            raise ValueError("too many regenerations!")
        rounds += 1
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        zoomrate = np.random.uniform(0.75, 1.25)
        ifflip = np.random.choice([1, -1], size=2, replace=True)
        movvec = np.random.uniform(-0.4, 0.4, size=2) * wh

        trMat = (
            AffineMats.shift(*(0.5 * wh))
            @ AffineMats.shift(*movvec)
            @ AffineMats.flip(*ifflip)
            @ AffineMats.zoom(zoomrate)
            @ AffineMats.rot(theta)
            @ AffineMats.shift(*(-0.5 * wh))
        )[0:2, :]

        lblAffined = cv.warpAffine(
            lbl,
            trMat,
            wh,
            borderMode=cv.BORDER_REPLICATE,
        )
        lblAffined = np.where(lblAffined > 0.5, 1.0, 0.0).astype(np.float32)
        SurfLbl1Affined = np.sum(lblAffined)

        expectedSurface = lblSurface * zoomrate
        insightRate = SurfLbl1Affined / (expectedSurface + EPS)
        if insightRate >= 0.8:
            # plane reserved nicely
            lblResult = lblAffined
        elif insightRate <= 0.4:
            # consider as no plane
            lblResult = np.zeros_like(lblAffined)
        else:
            # not typical, not wanted
            continue

        lblNonReplicated = cv.warpAffine(
            lbl,
            trMat,
            wh,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        lblNonReplicated = np.where(lblNonReplicated > 0.5, 1.0, 0.0).astype(np.float32)

        # hope not interpolated too much
        if np.sum(np.abs(lblNonReplicated - lblAffined)) / (SurfLbl1Affined + 1) >= 0.2:
            continue

        spl1 = cv.warpAffine(
            spl,
            trMat,
            wh,
            borderMode=cv.BORDER_REPLICATE,
        )
        return spl1, lblResult


def draw_random_line(image, n):
    image = np.ascontiguousarray(image)
    height, width, _ = image.shape
    color = (0, 0, 0)  # Black color
    for l in range(n):
        start_point = (np.random.randint(0, width), np.random.randint(0, height))
        end_point = (np.random.randint(0, width), np.random.randint(0, height))
        cv.line(image, start_point, end_point, color, 1)
    return image


def gaussianNoise(src):
    sig = np.random.uniform(-0.05, 0.15)
    if sig >= EPS:
        max_, min_ = np.max(src), np.min(src)
        noise = np.random.normal(0, sig, src.shape) * (max_ - min_)
        src = src + noise
    # gsig = np.random.uniform(-0.05, 0.15)
    # if gsig >= EPS:
    #     gray = np.mean(src, axis=2)
    #     max_, min_ = np.max(gray), np.min(gray)
    #     noise = np.expand_dims(np.random.normal(0, gsig, src.shape[0:2]), axis=2) * (
    #         max_ - min_
    #     )
    #     src = src + noise
    return np.clip(src, 0, 1, dtype=np.float32)


@dataclasses.dataclass
class NnTrackerDataset:
    path: str
    sel: str
    datasettype: str


class labeldataset(torch.utils.data.Dataset):
    class AugSteps(enum.Enum):
        affine = 0
        randLine = 1
        autoaug = 2
        gausNoise = 3

    def __init__(self) -> None:
        super().__init__()

    def init(
        self,
        path,
        selection,
        size,
        pathtype="fld",
        sheetname=None,
        stdShape=None,
        augSteps=list(),
    ):
        self.size = size
        self.augSteps = {s: True for s in augSteps}
        if selection is not None:
            selection = Xls2ListList(selection, sheetname)
            selection = [s[0] for s in selection]
            selection = [s for s in selection if s is not None]
            self.names = selection
            reader: ImgReader = None
            if pathtype == "fld":
                reader = ImgReaderFolder(path)
            elif pathtype == "zip":
                reader = ImgReaderZip(path)
            else:
                raise TypeError(f"inproper path type {pathtype}")

            items = []
            prog = Progress(len(selection))
            for i, p in enumerate(selection):
                spl = reader.read(f"spl/{p}")
                lbl = reader.read(f"lbl/{p}")

                if stdShape is not None:
                    spl = cv.resize(spl, stdShape)
                    lbl = cv.resize(lbl, stdShape)
                    lbl = cv.threshold(lbl[:, :, 0:1], 0.5, 1, cv.THRESH_BINARY)[1]
                pi = lbl2PlaneInfo(lbl)
                # simple check
                assert pi[3] < 1, f"image {p} have too big wingspan"
                items.append(SampleItem(p, spl, lbl, pi))
                prog.update(i)
            prog.setFinish()
        else:
            items = list()
        self.items: list[SampleItem] = items
        self.augger = NonAffineTorchAutoAugment()
        self.totensor = torchvision.transforms.ToTensor()

        return self

    def __len__(self):
        return self.size

    def dataAug(self, item: SampleItem):
        spl, lbl = item.spl, item.lbl

        if labeldataset.AugSteps.affine in self.augSteps:
            spl, lbl = safeAffineAug(spl, lbl)

        if labeldataset.AugSteps.randLine in self.augSteps:
            spl = draw_random_line(spl, np.random.randint(-3, 5))

        if labeldataset.AugSteps.autoaug in self.augSteps:
            spl = self.totensor(spl)
            spl = self.augger(spl)
            spl = tensorimg2ndarray(spl)

        if labeldataset.AugSteps.gausNoise in self.augSteps:
            spl = gaussianNoise(spl)

        return SampleItem("", spl, lbl, lbl2PlaneInfo(lbl))

    def procItemToTensor(self, item):
        return (
            self.totensor(item.spl),
            self.totensor(item.lbl),
            torch.tensor(item.pi, dtype=torch.float32),
        )

    def __getitem__(self, idx):
        index = self.rndIndex()
        item = self.items[index]
        item = self.dataAug(item)
        tup = self.procItemToTensor(item)
        return tup

    def rndIndex(self):
        return int(len(self.items) * np.random.random())

    def rawlength(self):
        return len(self.items)

    def rawgetitem(self, rawidx):
        return self.items[rawidx]

    def getname(self, rawidx):
        return self.names[rawidx]


def XYWH2XYXY(X, Y, W, H):
    return (X - W / 2, Y - H / 2, X + W / 2, Y + H / 2)


def XYXY2XYWH(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def AABBOf(lbl, noobjthresh=5):
    assert len(lbl.shape) == 2
    y, x = np.where(lbl > 0)
    if len(y) < noobjthresh:
        return (0, 0, 0, 0, 0)
    x1, x2 = np.min(x), np.max(x)
    y1, y2 = np.min(y), np.max(y)
    # (x1, x2, y1, y2, c)

    return XYXY2XYWH(x1, y1, x2, y2) + (1,)


class MassivePicturePlot:
    def __init__(self, plotShape, fig=None):
        self.plotShape = plotShape
        self.fig = fig if fig else plt.figure(figsize=(20, 20))
        self.i = 1

    def toNextPlot(self) -> plt.Axes:
        if self.isFull():
            raise IndexError("Too many pictures")
        ax = self.fig.add_subplot(self.plotShape[0], self.plotShape[1], self.i)
        self.i += 1
        return ax

    def isFull(self):
        return self.i > np.prod(self.plotShape)


def PI2Str(pi):
    return ",".join([f"{i:.2f}" for i in pi])


@dataclasses.dataclass
class ModelEvaluation:
    dataset: labeldataset
    model: torch.nn.Module = None
    device: str = "cpu"
    calcloss: typing.Callable = None

    @dataclasses.dataclass
    class InferenceResult:
        src: torch.Tensor
        lbl: torch.Tensor
        pi: torch.Tensor
        pihat: torch.Tensor
        loss: float
        timeConsumption: float

    def DrawData(self, i=None):
        if i is None:
            # draw random
            return self.dataset[0]
        else:
            return self.dataset.items[i]

    def Inferencable(self):
        return self.model is not None and self.calcloss is not None

    def Inference(self, src: torch.Tensor):
        return self.model.forward(src.unsqueeze(0).to(self.device))

    def Calcloss(self, pi: torch.Tensor, pihat: torch.Tensor):
        return self.calcloss(pi.unsqueeze(0).to(self.device), pihat).item()

    def IterDataAndInference(self, iterWork: typing.Callable, num_draws):
        """
        iteration common method
        """
        if self.Inferencable():
            self.model.eval()
        with torch.no_grad():
            prog = Progress(num_draws)
            for i in range(num_draws):
                while True:
                    src, lbl, pi = self.DrawData()
                    if self.Inferencable():
                        tstart = time.perf_counter()
                        pihat = self.Inference(src)
                        tend = time.perf_counter()
                        timeConsumption = tend - tstart
                        loss = self.Calcloss(pi, pihat)
                    else:
                        pihat, timeConsumption, loss = [None] * 3
                    ret = iterWork(
                        ModelEvaluation.InferenceResult(
                            src=src,
                            lbl=lbl,
                            pi=pi,
                            pihat=pihat,
                            loss=loss,
                            timeConsumption=timeConsumption,
                        )
                    )
                    # break on not specicified to be False
                    if ret != False:
                        break
                prog.update(i)
            prog.setFinish()

    def lossDistribution(self, num_draws=100):
        loss_values = []
        self.IterDataAndInference(
            lambda result: loss_values.append(result.loss),
            num_draws=num_draws,
        )
        loss_values = np.array(loss_values)

        plt.hist(loss_values, bins=100, color="blue", edgecolor="black")
        plt.xlabel("Loss Value")
        plt.ylabel("Frequency")
        plt.title("Loss Distribution")
        aveLoss = np.sum(loss_values) / len(loss_values)
        stdErr = np.std(loss_values)
        print(f"{aveLoss=}")
        print(f"{stdErr=}")

    def viewmodel(self, isWanted=None):
        isWanted = isWanted if isWanted else lambda x: True
        mpp = MassivePicturePlot([7, 8])
        samplenum = np.prod([7, 4])
        imshowconfig = {"vmin": 0, "vmax": 1}
        totalinferencetime = 0
        infercount = 0
        totalLoss = 0

        def iterWork(result: ModelEvaluation.InferenceResult):
            nonlocal totalinferencetime, infercount, totalLoss
            if not isWanted(result.loss):
                return False
            totalLoss += result.loss
            totalinferencetime += result.timeConsumption
            infercount += 1
            pihat = result.pihat[0].cpu().numpy()

            pi = result.pi.numpy()
            src, lbl = [tensorimg2ndarray(d) for d in [result.src, result.lbl]]

            mpp.toNextPlot()
            plt.title(PI2Str(pi))
            plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))

            mpp.toNextPlot()
            lblComparasion = (
                np.array(
                    [
                        lbl,
                        # planeInfo2Lbl(pi, stdShape),
                        np.zeros_like(lbl),
                        planeInfo2Lbl(pihat, stdShape),
                    ]
                )
                .squeeze(-1)
                .transpose([1, 2, 0])
            )

            plt.title(PI2Str(pihat))
            plt.imshow(lblComparasion, label="lblComparasion", **imshowconfig)

        self.IterDataAndInference(iterWork, samplenum)

        print(f"average inference time={totalinferencetime / samplenum}")
        print(f"average loss={totalLoss / samplenum}")


def savemodel(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def DatasetBenchmark(dataset: labeldataset):
    sampleNum = 8192
    prog = Progress(sampleNum)
    ps = perf_statistic(True)
    for i in range(sampleNum):
        dataset[0]
        prog.update(i)
    ps.stop()
    prog.setFinish()
    print(f"draw {sampleNum} samples in {ps.time()}s, {sampleNum/ps.time()}it/s")


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
