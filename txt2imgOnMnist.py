from torch.nn.modules.module import Module
from utilitypack.utility import *
from utilitypack.util_torch import *
from utilitypack.util_pyplot import *
from torch.utils.data import DataLoader
import torchvision
from predef.planeDiffusionPredef import *

getDeviceInfo()
device = getDevice()


# %%


class PlaneDiffusionSizedMnist(T2I):
    resolution = [128, 128]
    chanInput = 1
    nPrompt = 10
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

    def toPlaneDiffusionSize(self, img):
        return torch.nn.functional.interpolate(img, self.resolution, mode="bilinear")


class GaussianMNIST(torchvision.datasets.MNIST):

    superClass = torchvision.datasets.MNIST
    meanStdTable = {
        torchvision.datasets.FashionMNIST: [0.287, 0.352],
        torchvision.datasets.MNIST: [0.131, 0.308],
    }

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.superClass.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.superClass.__name__, "processed")

    def __init__(self, imgSnr=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imgSnr = imgSnr
        self.meanStd = self.meanStdTable[self.superClass]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, clsIdx = super().__getitem__(index)
        img = 1 - img
        cls = self.clsIndex2cls(clsIdx)
        return img, cls, clsIdx

    def clsIndex2cls(self, clsIdx):
        cls = torch.zeros(10)
        cls[clsIdx] = 1
        return cls


# %%
# dataset
imgSnr = 0.5
training_data = GaussianMNIST(
    imgSnr=imgSnr,
    root="dataset",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_data = GaussianMNIST(
    imgSnr=imgSnr,
    root="dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
batch_size = 2
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# %%
# set module
modelPath = "T2IMnist.pth"
model = setModule(PlaneDiffusionSizedMnist(), modelPath, device)


# %%
# loss


edgeKernelSize = 3


def calcLoss(img, pred):
    def calcEdge(img):
        return img - torch.nn.functional.conv2d(
            img,
            torch.ones([1, 1, edgeKernelSize, edgeKernelSize]) / edgeKernelSize**2,
            padding="same",
        )

    def x2px(x):
        return x**2 + torch.abs(x)

    lossRebuild = x2px(img - pred)
    # lossClearness = x2px(calcEdge(img) - calcEdge(pred))
    mse = torch.mean(lossRebuild, dim=(1, 2, 3))
    coef = 1
    loss = torch.sum(mse * coef)
    return loss


# %%
# train
sigmaScale = 0.25


def train(datatuple):
    img, cls, clsIdx = datatuple
    img = model.toPlaneDiffusionSize(img)
    batch_size, channel, height, width = img.shape
    sigma = torch.rand([batch_size, 1, 1, 1]) * sigmaScale
    noisedimg = DiffusionUtil.noisedImg(img, sigma=sigma)
    pred = model(noisedimg.to(device), cls.to(device))
    loss = calcLoss(img.to(device), pred.to(device))
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


def view():
    class md(ModelDemo):

        def iterWork(self, i):
            img, cls, clsIdx = training_data[np.random.randint(0, len(training_data))]
            img = model.toPlaneDiffusionSize(img.unsqueeze(0)).squeeze(0)
            sigma = torch.rand([1, 1, 1]) * 0.25
            noisedimg = DiffusionUtil.noisedImg(img, sigma=sigma)
            pred = (
                model(noisedimg.unsqueeze(0).to(device), cls.unsqueeze(0).to(device))
                .squeeze(0)
                .cpu()
                .numpy()
            )
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(img))
            sigma = sigma.item()
            plt.title(f"{clsIdx=}, {sigma=:.3f}")

            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(noisedimg))

            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(pred))

    pltShape = np.array([6, 2])
    mppShape = pltShape * [1, 3]

    md(model, mppShape, np.prod(pltShape)).do()


def viewDenoiseIter():
    pltShape = np.array([6, 1])
    iterNum = 6
    mppShape = pltShape * [1, (1 + 1 + iterNum)]
    datasetUsing = training_data
    mean, std = datasetUsing.meanStd

    class md(ModelDemo):

        def iterWork(self, i):
            img, cls, clsIdx = training_data[np.random.randint(0, len(training_data))]
            noiseLikeImg = torch.randn_like(img) * std + mean

            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(img))
            plt.title(f"{clsIdx=}")

            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(noiseLikeImg))

            t = noiseLikeImg.unsqueeze(0).to(device)
            cls = cls.unsqueeze(0).to(device)
            for j in range(iterNum):
                pred = model(DiffusionUtil.noisedImg(t, sigma=0.25), cls)

                self.mpp.toNextPlot()
                plt.imshow(tensorimg2ndarray(pred.squeeze(0).cpu().numpy()))

                t = pred

    md(model, mppShape, np.prod(pltShape)).do()


view()
