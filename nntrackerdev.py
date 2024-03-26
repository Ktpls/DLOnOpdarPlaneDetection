"""
import os
installPath=r'/kaggle/working/'
projPath=os.path.join(installPath, 'DLOnOpdarPlaneDetection')
if not os.path.exists(projPath):
    !git clone https://github.com/Ktpls/DLOnOpdarPlaneDetection.git
%cd {projPath}
!git pull
"""

# %% basics

RunOnWtUtilityEnviroment = True
if RunOnWtUtilityEnviroment:
    from predef.nntracker import *

    datasetroot = r"dataset/"
else:
    from predef.nntracker import *

    datasetroot = r"/kaggle/input/nntrackerle2renh"

# %%  nn def
device = getDevice()
modelpath = r"nntracker.pth"
model = getmodel(
    # nntracker_pi(),
    nntracker_respi(
        frozenLayers=(
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            # "layer4",
        ),
    ),
    modelpath,
    device,
)

# %% dataset


@dataclasses.dataclass
class NnTrackerDataset:
    path: str
    sel: str
    datasettype: str


print("loading dataset")
datasetname = "largeEnoughToRecon"
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
train_data = datasets[datasetname]
train_data = labeldataset().init(
    os.path.join(datasetroot, train_data.path),
    os.path.join(datasetroot, train_data.sel),
    8192,
    train_data.datasettype,
    None,
    stdShape,
    augSteps=[
        labeldataset.AugSteps.affine,
        labeldataset.AugSteps.randLine,
        labeldataset.AugSteps.autoaug,
        labeldataset.AugSteps.gausNoise,
    ],
)
test_data = datasets["largeEnoughToRecon"]
test_data = labeldataset().init(
    os.path.join(datasetroot, test_data.path),
    os.path.join(datasetroot, test_data.sel),
    32,
    test_data.datasettype,
    None,
    stdShape,
    augSteps=[
        # labeldataset.AugSteps.gausNoise,
    ],
)
bad_data = labeldataset().init(
    None,
    None,
    8192,
    None,
    None,
    stdShape,
    augSteps=[],
)
print("load finished")

# %%  dataloader
# for easier modify batchsize without reloading all samples
batch_size = 2
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=32, num_workers=0)


# %% lossFunc


def calclose(pi, pihat):
    (
        isObj,
        meanX,
        meanY,
        wingSpan,
    ) = (
        pi[:, 0],
        pi[:, 1],
        pi[:, 2],
        pi[:, 3],
    )

    coef = torch.ones_like(
        pi,
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    # enphasize wing span
    coef[:, 3] = 1.5
    # dont estimate pos and size when is no object
    coef[isObj != 1, 1:] = 0
    coef[isObj != 1, 0] = 4.5
    dist = ((pihat - pi) ** 2) * coef
    loss = torch.sum(dist)
    return loss


# %% train


def trainmainprogress(batch, datatuple):
    model.train()
    src, lbl, pi = datatuple
    src = src.to(device)
    lbl = lbl.to(device)
    pi = pi.to(device)
    pihat = model.forward(src)
    loss = calclose(pi, pihat)
    return loss


def onoutput(batch, aveerr):
    # return
    with torch.no_grad():
        model.eval()
        lossTotal = 0
        numTotal = 0
        for src, lbl, pi in test_dataloader:
            src = src.to(device)
            lbl = lbl.to(device)
            pi = pi.to(device)
            pihat = model.forward(src)
            lossTotal += calclose(pi, pihat).item()
            numTotal += batchsizeof(src)
            break
    print(f"testaveerr: {lossTotal/numTotal}")
    # writer.add_scalar("aveerr", aveerr, batch)


trainpipe.train(
    train_dataloader,
    torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    ),
    trainmainprogress,
    epochnum=10,
    customSubOnOutput=onoutput,
)


# %% save
savemodel(model, modelpath)

# %%
if __name__ == "__main__":
    exit()


# %% view effect
viewmodel(model, device, datasetusing=bad_data)


# %%
os.system("pause")


# %%
def rmModel():
    os.remove(modelpath)


rmModel()
