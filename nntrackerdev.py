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
# basics
from predef.nntrackerdev_predef import *

nnt = import_or_reload("predef.nntracker")
from predef.nntracker import *

RunOnWtUtilityEnviroment = True
if RunOnWtUtilityEnviroment:
    datasetroot = r"dataset/"
else:
    datasetroot = r"/kaggle/input/nntrackerle2renh"

# %%
# nn def
device = getDevice()
modelpath = r"nntracker.pth"
model = getmodel(
    # nntracker_pi(),
    nntracker_respi_MPn(
        freeLayers=(
            # "features.0",
            # "features.1",
            # "features.2",
            # "features.3",
            # "features.4",
            # "features.5",
            # "features.6",
            # "features.7",
            # "features.8",
            # "features.9",
            # "features.10",
            # "features.11",
            # "features.12",
            "features.13",
            "features.14",
            "features.15",
            "features.16",
        ),
        dropout=0.5,
    ),
    modelpath,
    device,
)

# %%
# dataset


print("loading dataset")
datasets = {
    "LE2RE": NnTrackerDataset(r"LE2RE", r"LE2RE/all.xlsx"),
    "smallAug": NnTrackerDataset(r"smallAug", r"smallAug/all.xlsx"),
    "affined": NnTrackerDataset(r"affined", r"affined/all.xlsx"),
}
train_data = datasets["affined"]
train_data = labeldataset().init(
    os.path.join(datasetroot, train_data.path),
    os.path.join(datasetroot, train_data.sel),
    8192,
    None,
    stdShape,
    augSteps=[
        # labeldataset.AugSteps.affine,
        # labeldataset.AugSteps.rndln,
        # labeldataset.AugSteps.autoaug,
        labeldataset.AugSteps.gausNoise,
    ],
) 
test_data = datasets["affined"]
test_data = train_data
print("load finished")

# %%
# dataloader
batch_size = 2
num_workers = 0
train_dataloader = DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers
)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# %%
# lossFunc


def calcloss(pi, pihat):
    batch, dimPi = pi.shape
    isObj = pi[:, 0].unsqueeze(1)
    isObjHat = pihat[:, 0].unsqueeze(1)

    coef = torch.tensor(
        [1, 1, 1.5], dtype=torch.float32, device=device, requires_grad=False
    ).unsqueeze(0)
    detailMask = isObj
    detailMse = (pihat - pi)[:, 1:] ** 2
    detailLoss = detailMask * coef * detailMse

    boolIsObj = isObj > 0.5
    alpha = 1 - 52 / 64
    alphat = torch.where(boolIsObj, alpha, 1 - alpha)
    pt = torch.where(boolIsObj, isObjHat, 1 - isObjHat)
    gamma = 2
    classLoss = -alphat * (1 - pt) ** gamma * torch.log(pt)

    loss = torch.sum(detailLoss) + torch.sum(classLoss)
    return loss


# %%
# train


def trainmainprogress(datatuple):
    model.train()
    src, lbl, pi = datatuple
    pihat = model.forward(src.to(device))
    loss = calcloss(pi.to(device), pihat)
    return loss


def onoutput(batch, aveerr):
    # return
    with torch.no_grad():
        model.eval()
        lossTotal = 0
        numTotal = 0
        for src, lbl, pi in test_dataloader:
            pihat = model.forward(src.to(device))
            lossTotal += calcloss(pi.to(device), pihat).item()
            numTotal += batchsizeof(src)
            break
    print(f"testaveerr: {lossTotal/numTotal}")
    # writer.add_scalar("aveerr", aveerr, batch)


trainpipe.train(
    train_dataloader,
    torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    ),
    trainmainprogress,
    epochnum=10,
    outputperbatchnum=1000,
    customSubOnOutput=onoutput,
)


# %%
# save
savemodel(model, modelpath)


# %%
# view effect
ModelEvaluation(
    model=model,
    device=device,
    dataset=test_data,
    calcloss=calcloss,
).viewmodel()


# %%
os.system("pause")


# %%
