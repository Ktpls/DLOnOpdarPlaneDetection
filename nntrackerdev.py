# %%
# installing
import os
import sys


def installLib(installpath, gitpath):
    projPath = os.path.join(installpath, os.path.splitext(os.path.basename(gitpath))[0])
    if not os.path.exists(projPath):
        os.system(rf"git clone {gitpath} {projPath}")
    else:
        cwd = os.getcwd()
        os.chdir(projPath)
        os.system(rf"git pull")
        os.chdir(cwd)
    if projPath not in sys.path:
        sys.path.append(projPath)


installLib(r"/kaggle/working", "https://github.com/Ktpls/pyinclude.git")
installLib(r"/kaggle/working", "https://github.com/Ktpls/DLOnOpdarPlaneDetection.git")
os.chdir(r"/kaggle/working")
#!rm "/kaggle/working/DLOnOpdarPlaneDetection/nntracker.pth"

# %%
# basics
from predef.nntrackerdev_predef import *

import_or_reload("predef.nntracker")
from predef.nntracker import *

RunOnWtUtilityEnviroment = True
if RunOnWtUtilityEnviroment:
    datasetroot = r"dataset/"
else:
    datasetroot = r"/kaggle/input/planedetectiondataset"

# %%
# nn def
modelpath = r"nntracker.pth"
model: nntracker_respi_MPn = getmodel(
    # nntracker_pi(),
    nntracker_respi_MPn(
        dropout=0.5,
    ),
    modelpath,
    device,
)
setModuleFree(
    model.backbone,
    (
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
)
os.chdir()
# %%
# dataset


print("loading dataset")
datasets = {
    "LE2RE": NnTrackerDataset(r"LE2RE", r"LE2RE/all.xlsx"),
    "smallAug": NnTrackerDataset(r"smallAug", r"smallAug/all.xlsx"),
    "affined": NnTrackerDataset(r"affined", r"affined/all.xlsx"),
}
train_data = datasets["smallAug"]
train_data = labeldataset().init(
    path=os.path.join(datasetroot, train_data.path),
    selection=os.path.join(datasetroot, train_data.sel),
    size=8192,
    stdShape=stdShape,
    device=device,
    augSteps=[
        # labeldataset.AugSteps.affine,
        # labeldataset.AugSteps.rndln,
        # labeldataset.AugSteps.autoaug,
        labeldataset.AugSteps.gausNoise,
    ],
)
print("load finished")

# %%
# dataloader
batch_size = 2
num_workers = 0
train_dataloader = DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers
)

# %%
# train


trainpipe.train(
    train_dataloader,
    torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    ),
    lambda *p: model.trainprogress(*p),
    epochnum=10,
    outputperbatchnum=1000,
)


# %%
# save
model.save(modelpath)


# %%
# demo
model.demo(train_data)


# %%
os.system("pause")


# %%
