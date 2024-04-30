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
model = setModule(T2I(), modelPath, device)


# %%
# loss


edgeKernelSize = 3


def calcLoss(img, pred, deltaSigma):
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
    coef = 1 / (deltaSigma**2 + 1)
    loss = torch.sum(mse * coef)
    return loss


# %%
# train
deltaSigmaScale = 0.25


def train(datatuple):
    img, cls, clsIdx = datatuple
    batch_size, channel, height, width = img.shape
    sigmaLower = torch.rand([batch_size, 1, 1, 1])
    sigmaDelta = torch.rand([batch_size, 1, 1, 1]) * deltaSigmaScale
    sigmaUpper = sigmaLower + sigmaDelta
    img = DiffusionUtil.noisedImg(img, sigma=sigmaLower)
    noisedimg = DiffusionUtil.noisedImg(img, sigma=sigmaDelta)
    pred = model(
        noisedimg.to(device),
        cls.to(device),
        sigmaLower.squeeze(dim=[-1, -2, -3]).to(device),
        sigmaUpper.squeeze(dim=[-1, -2, -3]).to(device),
    )
    loss = calcLoss(
        img.to(device), pred.to(device), sigmaLower.squeeze(dim=[-1, -2, -3]).to(device)
    )
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
            sigmaLower = torch.rand([1, 1, 1]) * 0
            sigmaDelta = torch.rand([1, 1, 1]) * 0.25
            sigmaUpper = sigmaLower + sigmaDelta
            img = DiffusionUtil.noisedImg(img, sigma=sigmaLower)
            noisedimg = DiffusionUtil.noisedImg(img, sigma=sigmaDelta)
            pred = (
                model(
                    noisedimg.unsqueeze(0).to(device),
                    cls.unsqueeze(0).to(device),
                    sigmaLower.squeeze(dim=[-1, -2, -3]).unsqueeze(0).to(device),
                    sigmaUpper.squeeze(dim=[-1, -2, -3]).unsqueeze(0).to(device),
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(img))
            sigmaLower=sigmaLower.item()
            plt.title(f"{clsIdx=}, {sigmaLower=:.3f}")
            
            self.mpp.toNextPlot()
            sigmaUpper=sigmaUpper.item()
            plt.imshow(tensorimg2ndarray(noisedimg))
            plt.title(f"{sigmaUpper=:.3f}")
            
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(pred))

    pltShape = np.array([6, 2])
    mppShape = pltShape * [1, 3]

    md(model, mppShape, np.prod(pltShape)).do()


def viewDenoiseIter():
    pltShape = np.array([6, 1])
    iterNum = 20
    mppShape = pltShape * [2, (1 + 1 + iterNum) // 2]
    datasetUsing = training_data

    class md(ModelDemo):

        def iterWork(self, i):
            img, noiimg, sig, cls, clsIdx = datasetUsing.getPureNoiseImg(
                np.random.randint(0, len(datasetUsing))
            )
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(img))
            plt.title(f"{clsIdx=}")
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(noiimg))
            plt.title(f"{sig=:.3f}")
            noiimg = noiimg.unsqueeze(0).to(device)
            cls = cls.unsqueeze(0).to(device)
            for j in range(iterNum):
                pred = model(datasetUsing.img2NoisedImg(noiimg)[1], cls)
                noiimg = pred
                self.mpp.toNextPlot()
                plt.imshow(tensorimg2ndarray(pred.squeeze(0).cpu().numpy()))

    md(model, mppShape, np.prod(pltShape)).do()


view()
