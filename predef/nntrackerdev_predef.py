from .utilkaggle import *

stdShape = [128, 128]


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
    X = np.arange(w).reshape(1, 1, w)
    Y = np.arange(h).reshape(1, h, 1)
    lblsurface = lbl.sum(axis=(-1, -2), keepdims=True) + 1
    meanX = (lbl * X).sum(axis=(-1, -2), keepdims=True) / lblsurface
    meanY = (lbl * Y).sum(axis=(-1, -2), keepdims=True) / lblsurface
    meanX = meanX[0, 0, 0] / w
    meanY = meanY[0, 0, 0] / h
    wingSpan = estimateWingSpan(lbl) / w
    isObj = 1 if lbl.sum() > 10 else 0
    return (
        isObj,
        meanX,
        meanY,
        wingSpan,
    )


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
            [("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)],
            [("Sharpness", 0.4, 7), ("Invert", 0.6, None)],
            [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
            [("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)],
            [("Solarize", 0.5, 2), ("Invert", 0.0, None)],
            [("Solarize", 0.6, 3), ("Equalize", 0.6, None)],
            [("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)],
        ]
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


def affineAugment(m, theta, zoomrate, ifflip, movvec):

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

    h, w, c = m.shape

    m = cv.warpAffine(
        m,
        (
            shift(0.5 * w, 0.5 * h)
            @ shift(*movvec)
            @ flip(*ifflip)
            @ zoom(zoomrate)
            @ rot(theta)
            @ shift(-0.5 * w, -0.5 * h)
        )[0:2, :],
        np.flip(m.shape[0:2]),
        borderMode=cv.BORDER_REPLICATE,
    )
    return m


def noiseAugment(m):
    # rand line
    def draw_random_line(image, n):
        height, width, _ = image.shape
        color = (0, 0, 0)  # Black color
        for l in range(n):
            start_point = (np.random.randint(0, width), np.random.randint(0, height))
            end_point = (np.random.randint(0, width), np.random.randint(0, height))
            cv.line(image, start_point, end_point, color, 1)
        return image

    # m = draw_random_line(m, 5)

    def gaussianNoise(src):
        noise = np.random.normal(0, 0.1, src.shape)
        src = np.clip(src + noise, 0, 1, dtype=np.float32)
        return src

    m = gaussianNoise(m)
    return m


class labeldataset(torch.utils.data.Dataset):

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
        rtDataAugOn=None,
    ):
        self.size = size
        if rtDataAugOn is None:
            rtDataAugOn = False
        self.rtDataAugOn = rtDataAugOn
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
            items.append(SampleItem(p, spl, lbl, pi))
            prog.update(i)
        self.items = items
        prog.setFinish()
        self.augger = NonAffineTorchAutoAugment()
        self.totensor = torchvision.transforms.ToTensor()

        return self

    def __len__(self):
        return self.size

    def rtDataAug(self, item: SampleItem):
        spl, lbl = item.spl, item.lbl
        lbl = NormalizeImgToChanneled_CvFormat(lbl)

        h, w, c = spl.shape
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        zoomrate = np.random.uniform(0.75, 1.25)
        ifflip = np.random.choice([1, -1], size=2, replace=True)
        movvec = np.random.uniform(-0.3, 0.3, size=2) * [w, h]
        spl = affineAugment(spl, theta, zoomrate, ifflip, movvec)
        lbl = affineAugment(lbl, theta, zoomrate, ifflip, movvec)
        lbl = np.where(lbl > 0.5, 1, 0)

        spl = self.totensor(spl)
        spl = self.augger(spl)
        spl = tensorimg2ndarray(spl)

        spl = noiseAugment(np.ascontiguousarray(spl))

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
        if self.rtDataAugOn:
            item = self.rtDataAug(item)
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


def viewmodel(model, device, datasetusing):
    model.eval()

    plotShape = [7, 4]
    subplotShape = [1, 2]
    samplenum = np.prod(plotShape)
    npp = nestedPyPlot(plotShape, subplotShape, plt.figure(figsize=(20, 20)))
    imshowconfig = {"vmin": 0, "vmax": 1}
    totalinferencetime = 0
    infercount = 0

    def PI2Str(pi):
        return ",".join([f"{i:.2f}" for i in pi])

    with torch.no_grad():
        for i in range(samplenum):
            src, lbl, pi = datasetusing[0]
            tstart = time.perf_counter()
            pihat = model.forward(src.reshape((1,) + src.shape).to(device))
            totalinferencetime += time.perf_counter() - tstart
            infercount += 1
            pihat = pihat[0].cpu().numpy()

            pi = pi.numpy()
            src, lbl = [tensorimg2ndarray(d) for d in [src, lbl]]

            # lblFromPi = planeInfo2Lbl(pi, stdShape)
            lblhat = planeInfo2Lbl(pihat, stdShape)
            npp.subplot(i, 0)
            plt.title(PI2Str(pi))
            plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
            npp.subplot(i, 1)
            lblComparasion = (
                np.array(
                    [
                        lbl,
                        # lblFromPi,
                        np.zeros_like(lbl),
                        lblhat,
                    ]
                )
                .squeeze(-1)
                .transpose([1, 2, 0])
            )

            plt.title(PI2Str(pihat))
            plt.imshow(lblComparasion, label="lblComparasion", **imshowconfig)
    print(f"average inference time={totalinferencetime / samplenum}")


def savemodel(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")
