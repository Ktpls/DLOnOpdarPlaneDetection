from predef.nntrackerdev_predef import *
from predef.nntracker import *

getDeviceInfo()
device = getDevice()


class T2I(torch.nn.Module):
    resolution = [28, 28]
    chanInput = 1
    nPrompt = 10
    chanPrompt = 4
    chanImgEncoded = 4
    subsampleRate0_1 = 2
    subsampleRate1_2 = 2
    chanFeature0 = 32
    chanFeature1 = 64
    chanFeature2 = 128
    featExport0 = 1
    featExport1 = 4
    featExport2 = 7

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        resolution = self.resolution
        chanInput = self.chanInput
        nPrompt = self.nPrompt
        chanPrompt = self.chanPrompt
        chanImgEncoded = self.chanImgEncoded
        subsampleRate0_1 = self.subsampleRate0_1
        subsampleRate1_2 = self.subsampleRate1_2
        chanFeature0 = self.chanFeature0
        chanFeature1 = self.chanFeature1
        chanFeature2 = self.chanFeature2
        self.promptEncoder = torch.nn.Sequential(
            torch.nn.Linear(nPrompt, 128),
            torch.nn.Hardswish(),
            torch.nn.Linear(128, chanPrompt * np.prod(resolution)),
            torch.nn.Hardswish(),
            ModuleFunc(lambda x: x.reshape([-1, chanPrompt, *resolution])),
        )
        self.imgEncoder = torch.nn.Sequential(
            ConvBnHs(chanInput, chanImgEncoded, 7),
            ConvBnHs(chanImgEncoded, chanImgEncoded),
        )
        self.featureExtractor = torch.nn.Sequential(
            ConvBnHs(chanImgEncoded + chanPrompt, chanFeature0),
            res_through(
                ConvBnHs(chanFeature0, chanFeature0),
                ConvBnHs(chanFeature0, chanFeature0),
            ),
            torch.nn.MaxPool2d(subsampleRate0_1, subsampleRate0_1),
            ConvBnHs(chanFeature0, chanFeature1),
            res_through(
                ConvBnHs(chanFeature1, chanFeature1),
                ConvBnHs(chanFeature1, chanFeature1),
            ),
            torch.nn.MaxPool2d(subsampleRate1_2, subsampleRate1_2),
            ConvBnHs(chanFeature1, chanFeature2),
            res_through(
                ConvBnHs(chanFeature2, chanFeature2),
                ConvBnHs(chanFeature2, chanFeature2),
            ),
        )
        self.Path0 = torch.nn.Sequential(
            res_through(
                ConvBnHs(chanFeature0, chanFeature0),
                ConvBnHs(chanFeature0, chanFeature0),
            ),
        )
        self.Path1 = torch.nn.Sequential(
            res_through(
                ConvBnHs(chanFeature1, chanFeature1),
                ConvBnHs(chanFeature1, chanFeature1),
            ),
        )
        self.Path2 = torch.nn.Sequential(
            res_through(
                ConvBnHs(chanFeature2, chanFeature2),
                ConvBnHs(chanFeature2, chanFeature2),
            ),
        )
        self.path2topath0 = torch.nn.Upsample(
            scale_factor=subsampleRate1_2 * subsampleRate0_1, mode="bilinear"
        )
        self.path1topath0 = torch.nn.Upsample(
            scale_factor=subsampleRate0_1, mode="bilinear"
        )
        self.output = torch.nn.Sequential(
            ConvBnHs(chanFeature0 + chanFeature1 + chanFeature2, chanFeature0),
            res_through(
                ConvBnHs(chanFeature0, chanFeature0),
                ConvBnHs(chanFeature0, chanFeature0),
            ),
            ConvBnHs(chanFeature0, chanInput),
        )

    def forward(self, noiimg, cls):
        imgEncoded = self.imgEncoder(noiimg)
        promptEncoded = self.promptEncoder(cls)
        x = torch.cat([imgEncoded, promptEncoded], dim=-3)
        for i, m in enumerate(self.featureExtractor):
            x = m(x)
            if i == self.featExport0:
                x0 = x
            if i == self.featExport1:
                x1 = x
            if i == self.featExport2:
                x2 = x
        x0 = self.Path0(x0)
        x1 = self.Path1(x1)
        x2 = self.Path2(x2)
        x = torch.concat([x0, self.path1topath0(x1), self.path2topath0(x2)], dim=-3)
        x = self.output(x)
        return x


class PlaneDiffusion(T2I):
    resolution = [128, 128]
    chanInput = 3
    nPrompt = 4
    chanPrompt = 4
    chanImgEncoded = 4
    subsampleRate0_1 = 4
    subsampleRate1_2 = 4
    chanFeature0 = 32
    chanFeature1 = 64
    chanFeature2 = 128
    featExport0 = 1
    featExport1 = 4
    featExport2 = 7


class DiffusionUtil:
    @staticmethod
    def noiseLike(x, sigma):
        return torch.randn_like(x) * sigma

    @staticmethod
    def noisedImg(img, sigma=1, snr=0.5):
        noise = DiffusionUtil.noiseLike(img, sigma)
        noiimg = noise * (1 - snr) + img * snr
        return noiimg


datasetroot = r"dataset/"

print("loading dataset")
datasets = {
    "LE2REnh": NnTrackerDataset(r"LE2REnh/LE2REnh.zip", r"LE2REnh/all.xlsx", "zip"),
    "SmallAug": NnTrackerDataset(r"SmallAug/SmallAug.zip", r"SmallAug/all.xlsx", "zip"),
    "largeEnoughToRecon": NnTrackerDataset(
        r"largeEnoughToRecon/largeEnoughToRecon.zip",
        r"largeEnoughToRecon/all.xlsx",
        "zip",
    ),
    "origins_nntracker": NnTrackerDataset(
        r"origins_nntracker/origins_nntracker.zip",
        r"origins_nntracker/hardones.xlsx",
        "zip",
    ),
}
train_data = datasets["largeEnoughToRecon"]
train_data = labeldataset().init(
    os.path.join(datasetroot, train_data.path),
    os.path.join(datasetroot, train_data.sel),
    8192,
    train_data.datasettype,
    None,
    stdShape,
    augSteps=[
        labeldataset.AugSteps.affine,
        # labeldataset.AugSteps.randLine,
        # labeldataset.AugSteps.autoaug,
        # labeldataset.AugSteps.gausNoise,
    ],
)
batch_size = 2
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# %%
# set module
modelPath = "T2I.pth"
model = setModule(PlaneDiffusion(), modelPath, device)


# %%
# loss


def calcLoss(img, pred):
    edgeKernelSize = 5

    def calcEdge(img):
        return img - torch.nn.functional.conv2d(
            img,
            torch.ones([1, 3, edgeKernelSize, edgeKernelSize])
            / (3 * edgeKernelSize**2),
            padding="same",
        )

    def x2px(x):
        return x**2 + torch.abs(x)

    lossRebuild = x2px(img - pred)
    lossClearness = x2px(calcEdge(img) - calcEdge(pred))
    mse = torch.mean(lossRebuild + lossClearness, dim=(1, 2, 3))
    coef = 1
    loss = torch.sum(mse * coef)
    return loss


# %%
# train
def train(datatuple):
    spl, lbl, pi = datatuple
    noisedSpl = DiffusionUtil.noisedImg(spl)
    spl, noisedSpl, pi = spl.to(device), noisedSpl.to(device), pi.to(device)
    pred = model(noisedSpl, pi)
    loss = calcLoss(spl, pred)
    return loss


model.train()
trainpipe.train(
    train_dataloader,
    torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3,
    ),
    train,
    epochnum=10,
    outputperbatchnum=100,
)

# %%
# save
savemodel(model, modelPath)

# %%
# view


def view():
    class md(ModelDemo):

        def iterWork(self, i):
            spl, lbl, pi = train_data[np.random.randint(0, len(train_data))]
            noisedSpl = DiffusionUtil.noisedImg(spl)
            pred = (
                model(noisedSpl.unsqueeze(0).to(device), pi.unsqueeze(0).to(device))
                .squeeze(0)
                .cpu()
                .numpy()
            )
            self.mpp.toNextPlot()
            plt.imshow(np.flip(tensorimg2ndarray(spl), axis=-1))
            self.mpp.toNextPlot()
            plt.imshow(planeInfo2Lbl(pi.numpy(), stdShape), cmap="gray")
            self.mpp.toNextPlot()
            plt.imshow(np.flip(tensorimg2ndarray(pred), axis=-1))

    pltShape = np.array([6, 2])
    mppShape = pltShape * [1, 3]

    md(model, mppShape, np.prod(pltShape)).do()


view()
