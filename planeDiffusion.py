# %%
"""
import os
import sys
def installLib(installpath, gitpath):
    projPath = os.path.join(installpath, os.path.splitext(os.path.basename(gitpath))[0])
    if not os.path.exists(projPath):
        os.system(rf"git clone {gitpath} {projPath}")
    else:
        %cd {projPath}
        os.system(rf"git pull")
    if projPath not in sys.path:
        sys.path.append(projPath)
installLib(
    r"/kaggle/working",
    "https://github.com/Ktpls/pyinclude.git",
)
installLib(r"/kaggle/working", "https://github.com/Ktpls/DLOnOpdarPlaneDetection.git")
%cd "/kaggle/working"
#!rm "/kaggle/working/DLOnOpdarPlaneDetection/nntracker.pth"
"""

# %%
from utilitypack.util_import import *

nntrackerdev_predef = import_or_reload("predef.nntrackerdev_predef")
from predef.nntrackerdev_predef import *

nntracker = import_or_reload("nntracker")
from predef.nntracker import *

from predef.planeDiffusionPredef import *

getDeviceInfo()
device = getDevice()


# %%
# model definition
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

    def forward(self, noiimg, prompt):
        imgEncoded = self.imgEncoder(noiimg)
        promptEncoded = self.promptEncoder(prompt)
        x = torch.cat([imgEncoded, promptEncoded], dim=-3)
        for i, m in enumerate(self.featureExtractor):
            x = m(x)
            if i == self.featExport0:
                x0 = x
            if i == self.featExport1:
                x1 = x
            if i == self.featExport2:
                x2 = x
        # x0 = torch.zeros_like(x0)
        x1 = torch.zeros_like(x1)
        x2 = torch.zeros_like(x2)
        x0 = self.Path0(x0)
        x1 = self.Path1(x1)
        x2 = self.Path2(x2)
        x = torch.concat([x0, self.path1topath0(x1), self.path2topath0(x2)], dim=-3)
        x = self.output(x)
        return x


# %%
# set module
modelPath = "T2IPD.pth"
model = setModule(PlaneDiffusion(), modelPath, device)

# %%
print("loading dataset")
RunOnWtUtilityEnviroment = True
if RunOnWtUtilityEnviroment:
    datasetroot = r"dataset/"
else:
    datasetroot = r"/kaggle/input/nntrackerle2renh"
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
# loss


def calcLoss(img, pred, device, sigma=1):
    edgeKernelSize = 5

    def calcEdge(img):
        return img - torch.nn.functional.conv2d(
            img,
            torch.ones([1, 3, edgeKernelSize, edgeKernelSize]).to(device)
            / (3 * edgeKernelSize**2),
            padding="same",
        )

    def x2px(x):
        return x**2  # + torch.abs(x)

    lossRebuild = x2px(img - pred)
    lossClearness = 0
    mse = torch.mean(lossRebuild + lossClearness, dim=(1, 2, 3))
    coef = 1 / (sigma**2 + 1)
    loss = torch.sum(mse * coef)
    return loss


# %%
# train
def train(datatuple):
    spl, lbl, pi = datatuple
    batch, channel, height, width = spl.shape
    sigma = torch.rand([batch, 1, 1, 1]) * 0.25
    noisedSpl = DiffusionUtil.noisedImg(spl)
    spl, noisedSpl, pi = spl.to(device), noisedSpl.to(device), pi.to(device)
    pred = model(noisedSpl, pi)
    loss = calcLoss(spl, pred, device=device, sigma=sigma.to(device))
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
            noisedSpl = DiffusionUtil.noisedImg(spl, sigma=torch.rand([1, 1, 1]) * 0.25)
            pred = (
                model(
                    noisedSpl.unsqueeze(0).to(device),
                    pi.unsqueeze(0).to(device),
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(spl))
            self.mpp.toNextPlot()
            plt.imshow(np.clip(tensorimg2ndarray(noisedSpl), 0, 1))
            self.mpp.toNextPlot()
            plt.imshow(np.clip(tensorimg2ndarray(pred), 0, 1))

    pltShape = np.array([6, 2])
    mppShape = pltShape * [1, 3]

    md(model, mppShape, np.prod(pltShape)).do()


def viewGeneration():
    pltShape = np.array([6, 1])
    iterNum = 6
    mppShape = pltShape * [1, (1 + 1 + iterNum)]
    datasetUsing = train_data

    class md(ModelDemo):

        def iterWork(self, i):
            spl, lbl, pi = datasetUsing[np.random.randint(0, len(datasetUsing))]
            splNoise = torch.clip(DiffusionUtil.noiseLike(spl, 0.155, 0.639), 0, 1)
            self.mpp.toNextPlot()
            plt.imshow(planeInfo2Lbl(pi.numpy(), stdShape), cmap="gray")
            self.mpp.toNextPlot()
            plt.imshow(tensorimg2ndarray(splNoise))
            x = splNoise.unsqueeze(0).to(device)
            pi = pi.unsqueeze(0).to(device)
            for j in range(iterNum):
                pred = model(DiffusionUtil.noisedImg(x), pi.unsqueeze(0).to(device))
                self.mpp.toNextPlot()
                plt.imshow(tensorimg2ndarray(pred.squeeze(0).cpu().numpy()))

    md(model, mppShape, np.prod(pltShape)).do()


def viewone():
    with torch.no_grad():
        spl, lbl, pi = train_data[np.random.randint(0, len(train_data))]
        noisedSpl = DiffusionUtil.noisedImg(spl, sigma=torch.rand([1, 1, 1]) * 0.25)
        pred = (
            model(
                noisedSpl.unsqueeze(0).to(device),
                pi.unsqueeze(0).to(device),
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
    plt.imshow(tensorimg2ndarray(pred))
    plt.show()


view()
